#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h"
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_fp16.h>
#include "NvInfer.h"



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
    std::vector<ModelResult> detect(const cv::Mat& inputImage);
    std::string get_class_by_id(int class_id);
    [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };
private:
    bool buildEngine();
    bool loadEngine();
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat& gpuImg);
    std::vector<ModelResult> postprocessDetect(std::vector<float> &featureVector);

    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs, std::vector<float> &featureVector);

    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, bool normalize, const std::array<float, 3>& subVals, const std::array<float, 3>& divVals);

    cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
        const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));

    ModelParams m_config;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    void clearGpuBuffers();

    float m_imgWidth = 0;
    float m_imgHeight = 0;
    float m_ratio = 1.0f;

    std::vector<float> m_featureVector;

    std::array<float, 3> m_subVals{0.f, 0.f, 0.f};
    std::array<float, 3> m_divVals{1.f, 1.f, 1.f};
    bool m_normalize = true;

    void* m_gpuInput = nullptr;
    void* m_gpuOutput = nullptr;
    size_t m_outputSize = 0;
    Logger m_logger;

    // Holds pointers to the input and output GPU buffers
    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;
};
