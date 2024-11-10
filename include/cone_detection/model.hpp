#ifndef MODEL_HPP
#define MODEL_HPP

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

typedef struct
{
    bool cuda_enable = false;
    int intra_op_num_threads = 1;
    int log_severity_level = 3;
} SessionOptions;

typedef struct
{
    std::string model_path;
    std::vector<std::string> classes;
    std::vector<int> img_size = { 640, 640 };
    float iou_threshold = 0.5;
    float rect_confidence_threshold = 0.6;
} ModelParams;

typedef struct
{
    int class_id;
    float confidence;
    cv::Rect box;
} ModelResult;

/**
 * @note Works only for detecting FP32 models.
 */
class Model
{
public:
    Model();

    /**
     * Clears the memory allocated for the session.
     */
    ~Model();

    /**
     * Model initialization. Creates an Ort::Session with the passed options,
     * gets classes from the file, gets names of input and output nodes, warms up
     * the model.
     * @param options Options to customize the session.
     * @param params Model parameters.
     * @note params.model_path can only include English letters and numbers.
     * @return True on success, false on failure.
     */
    bool init(const SessionOptions& options, const ModelParams& params);

    /**
     * Detects objects in the image. The image is processed, then the input tensor
     * is created, runs the model in session and processes the output tensors.
     * @param img The image for object detection.
     * @return The vector of results, i.e. detected objects.
     */
    std::vector<ModelResult> detect(const cv::Mat& img);

    /**
     * Get class from classes_ by ID.
     * @param class_id ID of the class.
     * @return Returns the class under the given ID.
     */
    std::string get_class_by_id(int class_id);
private:
    /**
     * Creates a blob (binary large object) from the image.
     * @param img The image from which we will make a blob.
     * @param blob Reference to the blob pointer. A blob of the image will be here.
     * @note Idk why we don't use blobfromImage() from the cv2 library.
     */
    void blob_from_image(const cv::Mat& img, float*& blob);

    /**
     * Creates a tensor from the blob and runs the model (in a session) with this
     * tensor as input.
     * @param blob Reference to the blob pointer. A blob data of an image is
     * stored here.
     * @return A vector of output tensors.
     */
    std::vector<Ort::Value> create_tensor_and_run(float*& blob);

    /**
     * Gets names of input and output nodes from the model using a session and
     * writes them to input_node_names_ and output_node_names_ class variables.
     */
    void get_node_names();

    /**
     * Processes images using the letter-boxing method.
     * @param img Image to process.
     * @return Letterboxing processed image.
     * @note Most common computer models like YOLO prefer a square sized input.
     */
    cv::Mat letterboxing(const cv::Mat& img);

    /**
     * Gets the detection results from the output tensors.
     * @param output_tensors A vector of output tensors.
     * @return The vector of results of detection.
     * @todo output_tensors const
     */
    std::vector<ModelResult> process_output_tensors(
        std::vector<Ort::Value>& output_tensors
    ) const;

    /**
     * Warm-up function that runs once after a session is created because
     * the first model run is slower.
     */
    void warm_up();

    ModelParams params_;
    Ort::Env env_;
    Ort::Session* session_;
    Ort::RunOptions options_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    // Letter-boxing scale
    float resize_scale_;
};

#endif
