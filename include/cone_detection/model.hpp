#ifndef MODEL_HPP
#define MODEL_HPP

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

/**
 * Options for ONNX Runtime Ort::SessionOptions.
 * All parameters can be changed in `params.yaml`.
 * @param cuda_enable responsible for starting with/without CUDA.
 * @param intra_op_num_threads the total number of INTRA threads to use
 * to run the model.
 * @param log_severity_level log severity level. Severity levels:
 * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
 */
typedef struct {
    bool cuda_enable = false;
    int intra_op_num_threads = 1;
    int log_severity_level = 3;
} SessionOptions;

/**
 * Model parameters. All parameters can be changed in `params.yaml`.
 * @param model_path path to the selected model.
 * @param classes vector of classes of the selected model.
 * @param img_size the image size that the selected model works with.
 * @param iou_threshold intersection-over-union threshold for deciding
 * whether boxes overlap too much with respect to IOU.
 * @param rect_confidence_threshold minimum score that the model will
 * consider the prediction to be a true prediction.
 */
typedef struct {
    std::string model_path;
    std::vector<std::string> classes;
    std::vector<int> img_size = { 640, 640 };
    float iou_threshold = 0.5;
    float rect_confidence_threshold = 0.6;
} ModelParams;

/**
 * Structure storing the result of model detection.
 * @param class_id class id from those defined in the
 * `ModelParams.classes`.
 * @param confidence model confidence in prediction.
 * @param box OpenCV rectangle (`Rect(x, y, width, height)`).
 * @note The structure stores only one result and is used when
 * returning a vector of results!
 */
typedef struct {
    int class_id;
    float confidence;
    cv::Rect box;
} ModelResult;

/**
 * The class that is used to work with the model, which includes things
 * like parameter setting, model warm-up, image preprocessing and
 * detection using the model.
 * @note Works only for detecting FP32 models!
 * @todo It may be necessary to take image preprocessing out of this
 * class in the future.
 */
class Model {
public:
    /**
     * Constructor.
     * Model initialization. Creates an Ort::Session with the passed
     * options and params. Gets names of input and output nodes
     * (`get_node_names()`) and warms up (`warm_up()`) the model inside
     * the function.
     * @param options Options to customize the session.
     * @param params Model parameters.
     * @note params.model_path can only include English letters and numbers.
     */
    Model(const SessionOptions& options, const ModelParams& params);

    /**
     * Detects objects in the image.
     * The image is pre-processed (`letterboxing()`), then the input tensor
     * is created (`blob_from_image()`),and then the model is run in session
     * and the output tensors are processed (`create_tensor_and_run()`).
     * @param img The image for object detection.
     * @return The vector of `ModelResult`, i.e. detected objects.
     */
    std::vector<ModelResult> detect(const cv::Mat& img);

    /**
     * Gets class name from `Model.classes_` by id.
     * @param class_id id of the class.
     * @return Returns the `std::string` class name under the given id.
     * @note It is public because we are now using it to display class
     * names in RViz2.
     */
    std::string get_class_by_id(int class_id);
private:
    /** A blob of the image will be stored here. */
    std::unique_ptr<float[]> blob;

    /**
     * Creates a blob (binary large object) from the image and writes
     * to class member `blob`.
     * @param img image from which we will make a blob.
     * @todo Idk why we don't use blobfromImage() from the cv2 library.
     */
    void blob_from_image(const cv::Mat& img);

    /**
     * Creates a tensor from the `blob` and runs the model (in a session)
     * with this tensor as input.
     * @return A vector of output tensors.
     */
    std::vector<Ort::Value> create_tensor_and_run();

    /**
     * Gets names of input and output nodes from the model using a session
     * and writes them to `input_node_names_` and `output_node_names_`
     * class variables.
     */
    void get_node_names();

    /**
     * Processes images using the letter-boxing method.
     * @param img image to process.
     * @return letterboxing processed image.
     * @note Most common computer models like YOLO prefer a square
     * sized input.
     */
    cv::Mat letterboxing(const cv::Mat& img);

    /**
     * Gets the detection results from the output tensors.
     * @param output_tensors vector of output tensors.
     * @return vector of results of detection.
     * @todo const reference for output_tensors.
     */
    std::vector<ModelResult> process_output_tensors(
        std::vector<Ort::Value>& output_tensors
    ) const;

    /**
     * Warm-up function that runs once after a session is created because
     * the first model run is slower.
     */
    void warm_up();

    /** Model parameters. */
    ModelParams params_;
    /** Contains the process-global ONNX Runtime environment. */
    Ort::Env env_;
    /** Loads and runs the model. */
    std::unique_ptr<Ort::Session> session_;
    /** A set of configurations for inference run behavior. */
    Ort::RunOptions options_;
    //** Input names obtained from the model in the function `get_node_names()` */
    std::vector<const char*> input_node_names_;
    //** Output names obtained from the model in the function `get_node_names()` */
    std::vector<const char*> output_node_names_;
    /** Letter-boxing scale. */
    float resize_scale_;
};

#endif
