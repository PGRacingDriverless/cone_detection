#include "cone_detection/model.hpp"

Model::Model(const SessionOptions& options, const ModelParams& params) {
    try {
        // Set options used to construct session
        Ort::SessionOptions session_options;
        if (options.cuda_enable) {
            // And for CUDA if enabled
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );
        session_options.SetIntraOpNumThreads(options.intra_op_num_threads);
        session_options.SetLogSeverityLevel(options.log_severity_level);

        // Set passed params to class member params_
        params_ = params;
        // Create OrtEnv object with setted logging level
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Model");
        // Create session from a model file with options
        session_ = std::make_unique<Ort::Session>(
            env_,
            params_.model_path.c_str(),
            session_options
        );
        // Create RunOptions object
        options_ = Ort::RunOptions{ nullptr };

        // Allocate memory for blob
        blob = std::make_unique<float[]>(params_.img_size.at(0) * params_.img_size.at(1) * 3);

        get_node_names();
        warm_up();
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Model: %s", e.what());
    }
}

std::vector<ModelResult> Model::detect(const cv::Mat& img) {
    std::vector<ModelResult> res{};

    try {
    auto start = std::chrono::high_resolution_clock::now();
        // Preprocess image
        cv::Mat processed_img = letterboxing(img);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = 
    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Preprocess time: " << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
        blob_from_image(processed_img);
    end = std::chrono::high_resolution_clock::now();
    duration = 
    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "blob time: " << duration.count() << std::endl;    

    start = std::chrono::high_resolution_clock::now();
        auto output_tensors = create_tensor_and_run();
    end = std::chrono::high_resolution_clock::now();
    duration = 
    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "create_tensor_and_run time: " << duration.count() << std::endl;   

    start = std::chrono::high_resolution_clock::now();
        res = process_output_tensors(output_tensors);
    end = std::chrono::high_resolution_clock::now();
    duration = 
    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "process_output_tensors time: " << duration.count() << std::endl;   
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("cone_detection"), "Model: %s", e.what());
    }

    return res;
}

void Model::blob_from_image(const cv::Mat& img) {
    int channels = img.channels();
    int height = img.rows;
    int width = img.cols;
    // Pointer to the raw data of the img
    const uint8_t* img_data = img.data;

    // The order matters because of cv::Mat memory layout!
    // Changing the order of loops slow down this function!
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // Direct access to the pixel data
                blob[c * width * height + h * width + w] =
                    img_data[(h * width + w) * channels + c] / 255.0f;
            }
        }
    }
}

std::vector<Ort::Value> Model::create_tensor_and_run() {
    // Create a vector of tensor dimensions of the input node
    std::vector<int64_t> input_node_dims =
        { 1, 3, params_.img_size.at(0), params_.img_size.at(1) };
    
    // Create a tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
            blob.get(),
            3 * params_.img_size.at(0) * params_.img_size.at(1),
            input_node_dims.data(),
            input_node_dims.size()
        );

    // Run the model in a session with our options and data
    auto output_tensors = session_->Run(
        options_,
        input_node_names_.data(),
        &input_tensor,
        1,
        output_node_names_.data(),
        output_node_names_.size()
    );

    return output_tensors;
}

void Model::get_node_names() {
    // Memory allocator with default options
    Ort::AllocatorWithDefaultOptions allocator;

    // Get number of input and output nodes
    size_t input_nodes_num = session_->GetInputCount();
    size_t output_nodes_num = session_->GetOutputCount();

    // Get input node names
    for (size_t node_num = 0; node_num < input_nodes_num; node_num++) {
        Ort::AllocatedStringPtr input_node_name =
            session_->GetInputNameAllocated(node_num, allocator);
        /**
         * @todo We need "Array of null terminated UTF8 encoded
         * strings of the output names" for ONNX Runtime
         * 
         * case 1:
         * input_node_names_.push_back(input_node_name.get());
         * The memory is cleared after leaving input_node_name
         * go out of scope.
         * 
         * case 2:
         * std::string temp_str(input_node_name.get());
         * input_node_names_.push_back(temp_str.c_str());
         * c_str() return pointer on string buffer.
         * No string = no memory.
         * 
         * case 3:
         * char* temp_buf = new char[50]; <- we set buffer size here!!!
         * This memory never cleared, so it works.
         */
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        input_node_names_.push_back(temp_buf);
    }

    // Get output node names
    for (size_t node_num = 0; node_num < output_nodes_num; node_num++) {
        Ort::AllocatedStringPtr output_node_name =
            session_->GetOutputNameAllocated(node_num, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, output_node_name.get());
        output_node_names_.push_back(temp_buf);
    }
}

cv::Mat Model::letterboxing(const cv::Mat& img) {
    cv::Mat res_img;

    // Convert to RGB color space
    if (img.channels() == 3) {
        res_img = img.clone();
        cv::cvtColor(res_img, res_img, cv::COLOR_BGR2RGB);
    }
    else {
        cv::cvtColor(img, res_img, cv::COLOR_GRAY2RGB);
    }

    // Letter-boxing itself
    if (img.cols >= img.rows) {
        resize_scale_ = img.cols / (float)params_.img_size.at(0);
        cv::resize(
            res_img,
            res_img,
            cv::Size(params_.img_size.at(0), int(img.rows / resize_scale_))
        );
    }
    else {
        resize_scale_ = img.rows / (float)params_.img_size.at(1);
        cv::resize(
            res_img,
            res_img,
            cv::Size(int(img.cols / resize_scale_), params_.img_size.at(1))
        );
    }

    cv::Mat temp_img =
        cv::Mat::zeros(params_.img_size.at(0), params_.img_size.at(1), CV_8UC3);
    res_img.copyTo(temp_img(cv::Rect(0, 0, res_img.cols, res_img.rows)));
    res_img = std::move(temp_img);
    // Letter-boxing end

    return res_img;
}

std::vector<ModelResult> Model::process_output_tensors(
    std::vector<Ort::Value>& output_tensors
) const {
    // Get the type and shape of output tensors
    Ort::TypeInfo type_info = output_tensors.front().GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    // Create a vector of tensor dimensions of the output node and get shape data
    std::vector<int64_t> output_node_dims = tensor_info.GetShape();
    // Return a non-const pointer to a tensor contained buffer
    auto output = output_tensors.front().GetTensorMutableData<float>();
    
    //8400
    int stride_num = output_node_dims[1];
    //84
    int signal_result_num = output_node_dims[2];

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    cv::Mat raw_data = cv::Mat(stride_num, signal_result_num, CV_32F, output);
    /**
     * @note Ultralytics add transpose operator to the output of yolov8
     * model.which make yolov8/v5/v7 has same shape.
     * @see https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
     */
    raw_data = raw_data.t(); // added
    float* data = (float*)raw_data.data;

    // strideNum -> signalResultNum
    for (int i = 0; i < signal_result_num; ++i) {
        float* classes_scores = data + 4;
        cv::Mat scores(1, params_.classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
        if (max_class_score > params_.rect_confidence_threshold) {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id.x);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * resize_scale_);
            int top = int((y - 0.5 * h) * resize_scale_);

            int width = int(w * resize_scale_);
            int height = int(h * resize_scale_);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        // signalResultNum -> strideNum
        data += stride_num;
    }

    // Performs non maximum suppression given boxes and corresponding scores
    std::vector<int> nms_res;
    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        params_.rect_confidence_threshold,
        params_.iou_threshold,
        nms_res
    );
    
    std::vector<ModelResult> results{};

    for (size_t i = 0; i < nms_res.size(); ++i) {
        int idx = nms_res[i];
        ModelResult res;
        res.class_id = class_ids[idx];
        res.confidence = confidences[idx];
        res.box = boxes[idx];
        results.push_back(res);
    }

    return results;
}

void Model::warm_up() {
    // Create "image" of our sizes with 3 channels
    cv::Mat img =
        cv::Mat(cv::Size(params_.img_size.at(0), params_.img_size.at(1)), CV_8UC3);
    // Preprocess "image"
    cv::Mat processed_img = letterboxing(img);

    blob_from_image(processed_img);

    create_tensor_and_run();
}

std::string Model::get_class_by_id(int class_id) {
    if (class_id < 0 || static_cast<size_t>(class_id) >= params_.classes.size()) {
        throw std::out_of_range("Invalid class_id");
    }

    return params_.classes[class_id];
}
