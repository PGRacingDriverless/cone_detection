#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>
#include <string>
#include <array>
#include <fstream>
#include <iostream>

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
    std::vector<int> img_size = { 640, 640 };
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
private:
    bool buildEngine();
    bool loadEngine();
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat& gpuImg);
    std::vector<ModelResult> postprocess(const std::vector<float>& output);

    ModelParams m_config;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    void clearGpuBuffers();

    float m_imgWidth = 0;
    float m_imgHeight = 0;
    float m_ratio = 1.0f;

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
