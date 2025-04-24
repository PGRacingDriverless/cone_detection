#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <fstream>
#include <iostream>

// Utility methods
namespace ConeUtil {
    inline bool doesFileExist(const std::string &filepath) {
        std::ifstream f(filepath.c_str());
        return f.good();
    }
    
    inline void checkCudaErrorCode(cudaError_t code) {
        if (code != 0) {
            std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                                 "), with message: " + cudaGetErrorString(code);
            std::cout << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
    }
}
typedef struct {
    int class_id;
    float confidence;
    cv::Rect box;
} ModelResult;

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
};

typedef struct {
    std::string enginePath;
    std::string onnxModelPath;
    std::vector<std::string> classes;
    std::vector<int> img_size = { 1280, 1280 };
    float iou_threshold = 0.5;
    float rect_confidence_threshold = 0.6;
    bool useFP16 = false;
} ModelParams;

class Model {
public:
    Model(const ModelParams& config);
    ~Model();
    std::vector<ModelResult> detect(const cv::Mat& inputImage);
    std::string get_class_by_id(int class_id);
    [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };
private:
    bool buildEngine();
    bool loadEngine();
    std::vector<cv::cuda::GpuMat> preprocess(const cv::cuda::GpuMat& gpuImg);
    void postprocessDetect(std::vector<float> &featureVector);
    bool runInference(const std::vector<cv::cuda::GpuMat> &inputs, std::vector<float> &featureVector);
    void blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, bool normalize, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals);
    cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width, const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));
    void clearGpuBuffers();

    // Configuration and engine-related members
    ModelParams m_config;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    // GPU buffers and streams
    void* m_gpuInput = nullptr;
    void* m_gpuOutput = nullptr;
    size_t m_outputSize = 0;
    Logger m_logger;
    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;
    cudaStream_t inferenceCudaStream;
    cudaStream_t profileStream;

    float m_imgWidth = 0;
    float m_imgHeight = 0;
    float m_ratio = 1.0f;
    bool m_normalize = true;
    std::array<float, 3> m_subVals{0.f, 0.f, 0.f};
    std::array<float, 3> m_divVals{1.f, 1.f, 1.f};
    int imgH = 1280;
    int imgW = 1280;
    cv::cuda::GpuMat gpu_dst = cv::cuda::GpuMat(1, imgH * imgW, CV_8UC3);
    size_t width = 1280 * 1280;
    std::vector<cv::cuda::GpuMat> input_channels{
        cv::cuda::GpuMat(imgW, imgH, CV_8U, &(gpu_dst.ptr()[0])),
        cv::cuda::GpuMat(imgW, imgH, CV_8U, &(gpu_dst.ptr()[width])),
        cv::cuda::GpuMat(imgW, imgH, CV_8U, &(gpu_dst.ptr()[width * 2]))
    };

    // Preallocated data structures for performance
    cv::cuda::GpuMat rgbMat;                     // For color conversion
    cv::cuda::GpuMat gpuImg;                     // For input image upload
    cv::cuda::GpuMat blob;                   // For blob 
    cv::Mat output;                              // For postprocessing output
    std::vector<cv::cuda::GpuMat> input;             // For input image
    std::vector<ModelResult> ret;                // Detection results
    std::vector<cv::cuda::GpuMat> preprocessedInputs; // Preprocessed inputs
    std::vector<float> m_featureVector;          // Inference output

    std::vector<cv::Rect> bboxes;
    cv::Rect_<float> bbox;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    
};
