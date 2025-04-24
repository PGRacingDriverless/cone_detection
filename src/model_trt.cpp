#include "cone_detection/model_trt.hpp"

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as
    // https://github.com/gabime/spdlog For the sake of this tutorial, will just
    // log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

Model::Model(const ModelParams& params) : m_config(params) {
    m_config.enginePath = params.enginePath;
    m_config.classes = params.classes;
    m_config.onnxModelPath = params.onnxModelPath;
    m_config.rect_confidence_threshold = params.rect_confidence_threshold;
    m_config.useFP16 = true; 
    if (!loadEngine()) {
        if (!buildEngine()) {
            throw std::runtime_error("Failed to build and load engine.");
        }
    }
    // Create the cuda stream that will be used for inference
    ConeUtil::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
}
Model::~Model() {
    // Free the GPU buffers
    clearGpuBuffers();
    // Destroy the cuda stream
    ConeUtil::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    // Destroy the runtime and engine
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
}

bool Model::loadEngine() {
    std::cout << "Loading engine from: " << m_config.enginePath << std::endl;
    std::ifstream infile(m_config.enginePath);
    if (!infile.good()) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Engine file does not exist or cannot be opened: %s", m_config.enginePath.c_str());
        return false;
    }
    std::ifstream file(m_config.enginePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Unable to read engine file: %s", m_config.enginePath.c_str());
        return false;
    }

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a
    // particular invocation
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    clearGpuBuffers();
    m_buffers.resize(m_engine->getNbIOTensors());

    m_outputLengths.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_IOTensorNames.clear();

    // Create a cuda stream
    cudaStream_t stream;
    ConeUtil::checkCudaErrorCode(cudaStreamCreate(&stream));

    // Allocate GPU memory for input and output buffers
    m_outputLengths.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const nvinfer1::TensorIOMode tensorType = m_engine->getTensorIOMode(tensorName);
        const nvinfer1::Dims tensorShape = m_engine->getTensorShape(tensorName);
        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            // The implementation currently only supports inputs of type float
            if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Error, the implementation currently only supports float inputs");
                return false;
            }

            // Store the input dims for later use
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            m_inputBatchSize = tensorShape.d[0];
        } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
            
            // The binding is an output
            uint32_t outputLength = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that
                // into account when sizing the buffer
                outputLength *= tensorShape.d[j];
            }

            m_outputLengths.push_back(outputLength);
            cudaMallocAsync(&m_buffers[i], outputLength * sizeof(float), stream);
        } else {
            RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Error, IO Tensor is neither an input or output!");
            return false;
        }
    }

    // Synchronize and destroy the cuda stream
    ConeUtil::checkCudaErrorCode(cudaStreamSynchronize(stream));
    ConeUtil::checkCudaErrorCode(cudaStreamDestroy(stream));
    return true;
}

bool Model::buildEngine() {
    std::cout << "Building engine from: " << m_config.onnxModelPath << std::endl;
    // Create our engine builder.
    std::unique_ptr<nvinfer1::IBuilder> builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch size is deprecated).
    // More info here:
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    // Batch size const at 1
    // uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }
    // We are going to first read the onnx file into memory, then pass that buffer
    // to the parser.
    std::ifstream file(m_config.onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Unable to read engine file: %s", m_config.onnxModelPath.c_str());
        return false;
    }

    // Parse the buffer we read into memory.
    bool parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const int32_t numInputs = network->getNbInputs();
    if (numInputs < 1) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Error, model needs at least 1 input!");
        return false;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const nvinfer1::ITensor* input = network->getInput(i);
        const char* inputName = input->getName();
        const nvinfer1::Dims inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile
        
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                  nvinfer1::Dims4(1, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                  nvinfer1::Dims4(1, inputC, inputH, inputW));
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                  nvinfer1::Dims4(1, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(optProfile);
    if (m_config.useFP16) config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // CUDA stream used for profiling by the builder.
    ConeUtil::checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to
    // kVERBOSE and try rebuilding the engine. Doing so will provide you with more
    // information on why exactly it is failing.
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    
    // Write the engine to disk
    std::ofstream outfile(m_config.enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    outfile.close();

    ConeUtil::checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return loadEngine();
}

std::vector<cv::cuda::GpuMat> Model::preprocess(const cv::cuda::GpuMat &gpuImg) {
    const std::vector<nvinfer1::Dims3> &inputDims = getInputDims();

    // Convert the image from BGR to RGB
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (rgbMat.rows != inputDims[0].d[1] || rgbMat.cols != inputDims[0].d[2]) {
        rgbMat = resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2], cv::Scalar(0, 0, 0));
    }

    preprocessedInputs.clear();
    preprocessedInputs.push_back(rgbMat);

    return preprocessedInputs;
}

void Model::postprocessDetect(std::vector<float> &m_featureVector) {
    const std::vector<nvinfer1::Dims> &outputDims = getOutputDims();
    int32_t numChannels = outputDims[0].d[1];
    int32_t numAnchors = outputDims[0].d[2];

    size_t numClasses = m_config.classes.size();

    output = cv::Mat(numChannels, numAnchors, CV_32F, m_featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
        float* rowPtr = output.row(i).ptr<float>();
        float* bboxesPtr = rowPtr;
        float* scoresPtr = rowPtr + 4;
        float* maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > m_config.rect_confidence_threshold) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            int label = maxSPtr - scoresPtr;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, m_config.rect_confidence_threshold, m_config.iou_threshold, indices);
    ModelResult result{};
    // Choose the top k detections
    int cnt = 0;
    for (int &chosenIdx : indices) {

        result.confidence = scores[chosenIdx];
        result.class_id = labels[chosenIdx];
        result.box = bboxes[chosenIdx];
        ret.push_back(result);

        cnt += 1;
    }
    scores.clear();
    labels.clear();
    indices.clear();
    bboxes.clear();
}

std::vector<ModelResult> Model::detect(const cv::Mat& inputImage) {
    // std::cout << "==================================================" << std::endl;
    try {
        // auto uploadStart = std::chrono::high_resolution_clock::now();
        gpuImg.upload(inputImage);
        // auto uploadEnd = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> uploadElapsed = uploadEnd - uploadStart;
        // std::cout << "Image upload time: " << uploadElapsed.count() << " ms" << std::endl;
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Error uploading image to GPU: %s", e.what());
    }

    // auto preprocessStart = std::chrono::high_resolution_clock::now();
    input = preprocess(gpuImg);
    // auto preprocessEnd = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> preprocessElapsed = preprocessEnd - preprocessStart;
    // std::cout << "Preprocessing time: " << preprocessElapsed.count() << " ms" << std::endl;

    // auto inferenceStart = std::chrono::high_resolution_clock::now();
    bool succ = runInference(input, m_featureVector);
    // auto inferenceEnd = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> inferenceElapsed = inferenceEnd - inferenceStart;
    // std::cout << "Inference time: " << inferenceElapsed.count() << " ms" << std::endl;

    if (!succ) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Error: Unable to run inference.");
    }

    // auto postprocessStart = std::chrono::high_resolution_clock::now();
    ret.clear();
    postprocessDetect(m_featureVector);
    // auto postprocessEnd = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> postprocessElapsed = postprocessEnd - postprocessStart;
    // std::cout << "Postprocessing time: " << postprocessElapsed.count() << " ms" << std::endl;

    return ret;
}

std::string Model::get_class_by_id(int class_id) {
    if (class_id < 0 || static_cast<size_t>(class_id) >= m_config.classes.size()) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Invalid class_id: %d", class_id);
    }

    return m_config.classes[class_id];
}

void Model::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of outputs
        const int numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            ConeUtil::checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
        }
        m_buffers.clear();
    }
}


cv::cuda::GpuMat Model::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
    const cv::Scalar &bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

bool Model::runInference(const std::vector<cv::cuda::GpuMat> &inputs,
std::vector<float> &m_featureVector) {
    // Error checking
    if (inputs.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Provided input vector is empty!");
        return false;
    }

    const nvinfer1::Dims3 &dims = m_inputDims[0];
    cv::cuda::GpuMat input = inputs[0];
    if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2]) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Input does not have correct size!");
        return false;
    }

    nvinfer1::Dims4 inputDims = {1, dims.d[0], dims.d[1], dims.d[2]};
    m_context->setInputShape(m_IOTensorNames[0].c_str(), inputDims);

    // Convert NHWC to NCHW and preprocess
    // auto blobStart = std::chrono::high_resolution_clock::now();
    blobFromGpuMats(inputs, m_normalize, m_subVals, m_divVals);
    // auto blobEnd = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> blobElapsed = blobEnd - blobStart;
    // std::cout << "Blob creation time: " << blobElapsed.count() << " ms" << std::endl;
    m_buffers[0] = blob.ptr<void>();

    // Ensure all dynamic bindings are defined
    if (!m_context->allInputDimensionsSpecified()) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Error, not all required dimensions specified.");
    }

    // Set tensor addresses
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        if (!m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i])) {
            return false;
        }
    }

    // Run inference
    if (!m_context->enqueueV3(inferenceCudaStream)) {
        return false;
    }

    // Copy outputs back to CPU asynchronously
    m_featureVector.clear();
    for (int32_t outputBinding = 1; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
        uint32_t outputLength = m_outputLengths[outputBinding - 1];
        m_featureVector.resize(outputLength);
        cudaMemcpyAsync(m_featureVector.data(),
                        static_cast<char *>(m_buffers[outputBinding]),
                        outputLength * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream);
    }

    // Synchronize the CUDA stream
    ConeUtil::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    return true;
}

void Model::blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, bool normalize, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals) {
    cv::cuda::split(batchInput[0], input_channels); // HWC -> CHW
    if (normalize) {
        // [0.f, 1.f]
        gpu_dst.convertTo(blob, CV_32FC3, 1.f / 255.f);
    } else {
        // [0.f, 255.f]
        gpu_dst.convertTo(blob, CV_32FC3);
    }

    // Apply scaling and mean subtraction
    cv::cuda::subtract(blob, cv::Scalar(subVals[0], subVals[1], subVals[2]), blob, cv::noArray(), -1);
    cv::cuda::divide(blob, cv::Scalar(divVals[0], divVals[1], divVals[2]), blob, 1, -1);
}