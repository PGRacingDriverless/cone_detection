#include "cone_detection/model_trt.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <algorithm>

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
    m_config.useFP16 = true; // Default to false, can be adjusted as needed

    if (!loadEngine()) {
        if (!buildEngine()) {
            throw std::runtime_error("Failed to build and load engine.");
        }
    }
}

bool Model::loadEngine() {
    std::ifstream file(m_config.enginePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
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
    //checkCudaErrorCode(cudaStreamCreate(&stream));
    cudaStreamCreate(&stream);

    // Allocate GPU memory for input and output buffers
    m_outputLengths.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const auto tensorType = m_engine->getTensorIOMode(tensorName);
        const auto tensorShape = m_engine->getTensorShape(tensorName);
        const auto tensorDataType = m_engine->getTensorDataType(tensorName);
        // The binding is an output
        uint32_t outputLength = 1;
        m_outputDims.push_back(tensorShape);
         for (int j = 1; j < tensorShape.nbDims; ++j) {
            // We ignore j = 0 because that is the batch size, and we will take that
            // into account when sizing the buffer
            outputLength *= tensorShape.d[j];
        }
        m_outputLengths.push_back(outputLength);
        // Now size the output buffer appropriately, taking into account the max
        // possible batch size (although we could actually end up using less
        // memory)
        //checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLength * m_options.maxBatchSize * sizeof(float), stream));
        cudaMallocAsync(&m_buffers[i], outputLength * sizeof(float), stream);
    }

    // Synchronize and destroy the cuda stream
    // checkCudaErrorCode(cudaStreamSynchronize(stream));
    // checkCudaErrorCode(cudaStreamDestroy(stream));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return true;
}

bool Model::buildEngine() {
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Define an explicit batch size and then create the network (implicit batch
    // size is deprecated). More info here:
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }
    // We are going to first read the onnx file into memory, then pass that buffer
    // to the parser. Had our onnx model file been encrypted, this approach would
    // allow us to first decrypt the buffer.
    std::ifstream file(m_config.onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    // Parse the buffer we read into memory.
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const auto numInputs = network->getNbInputs();
    if (numInputs < 1) {
        throw std::runtime_error("Error, model needs at least 1 input!");
    }
    const auto input0Batch = network->getInput(0)->getDimensions().d[0];
    for (int32_t i = 1; i < numInputs; ++i) {
        if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
            throw std::runtime_error("Error, the model has multiple inputs, each "
                                     "with differing batch sizes!");
        }
    }


    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Register a single optimization profile
    nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        // Must specify dimensions for all the inputs the model expects.
        const auto input = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        // Specify the optimization profile`
        
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
    cudaStream_t profileStream;
    //checkCudaErrorCode(cudaStreamCreate(&profileStream));
    cudaStreamCreate(&profileStream);
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

    std::cout << "Success, saved engine to " << m_config.enginePath << std::endl;

    //checkCudaErrorCode(cudaStreamDestroy(profileStream));
    cudaStreamDestroy(profileStream);
    return loadEngine();
}

std::vector<std::vector<cv::cuda::GpuMat>> Model::preprocess(const cv::cuda::GpuMat &gpuImg) {
    // Populate the input vectors
    const auto &inputDims = getInputDims();

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
}

std::vector<ModelResult> Model::postprocess(const std::vector<float>& output) {
    int numAnchors = m_outputSize / (4 + m_config.classes.size());

    std::vector<ModelResult> results;
    for (int i = 0; i < numAnchors; ++i) {
        const float* row = output.data() + i * (4 + m_config.classes.size());
        float x = row[0], y = row[1], w = row[2], h = row[3];
        float maxScore = 0;
        int classId = -1;
        for (size_t j = 0; j < m_config.classes.size(); ++j) {
            if (row[4 + j] > maxScore) {
                maxScore = row[4 + j];
                classId = static_cast<int>(j);
            }
        }
        if (maxScore > m_config.rect_confidence_threshold) {
            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            results.push_back({classId, maxScore, cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0)});
        }
    }
    return results;
}

std::vector<ModelResult> Model::detect(const cv::Mat& inputImage) {
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImage);
    auto batchedInput = preprocess(gpuImg);

    cudaMemcpy(m_gpuInput, batchedInput[0][0].ptr<float>(), sizeof(float) * m_inputDims.d[1] * m_inputDims.d[2] * m_inputDims.d[3], cudaMemcpyDeviceToDevice);

    m_context->setBindingDimensions(0, m_inputDims);
    void* bindings[] = {m_gpuInput, m_gpuOutput};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    m_context->enqueueV2(bindings, stream, nullptr);

    std::vector<float> output(m_outputSize);
    cudaMemcpyAsync(output.data(), m_gpuOutput, sizeof(float) * m_outputSize, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return postprocess(output);
}

std::string Model::get_class_by_id(int class_id) {
    if (class_id < 0 || static_cast<size_t>(class_id) >= m_config.classes.size()) {
        throw std::out_of_range("Invalid class_id");
    }

    return m_config.classes[class_id];
}

void Model::clearGpuBuffers() {
    if (!m_buffers.empty()) {
        // Free GPU memory of outputs
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            //checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
            cudaFree(m_buffers[outputBinding]);
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