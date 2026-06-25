#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

// =========================================================================
// TRT 10.x ILogger — 输出警告及以上的日志
// =========================================================================
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << msg << std::endl;
    }
};

// =========================================================================
// TensorRT 引擎封装
// =========================================================================
class TensorRTEngine {
public:
    TensorRTEngine()  = default;
    ~TensorRTEngine() { release(); }

    // ---- 禁止拷贝 ----
    TensorRTEngine(const TensorRTEngine&)            = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    // ---- 允许移动 ----
    TensorRTEngine(TensorRTEngine&& other) noexcept;
    TensorRTEngine& operator=(TensorRTEngine&& other) noexcept;

    // ---- 加载 .engine 文件 + 分配 buffer ----
    bool load(const std::string& engine_path);

    // ---- 执行推理（输入/输出都在 GPU 上）----
    bool infer(cudaStream_t stream = nullptr);

    // ---- 获取 I/O buffer 指针 ----
    float* inputPtr()  const { return static_cast<float*>(buffers_[input_idx_]); }
    float* outputPtr() const { return static_cast<float*>(buffers_[output_idx_]); }

    // ---- 查询 ----
    int numInputs()  const { return num_inputs_; }
    int numOutputs() const { return num_outputs_; }
    bool isLoaded()  const { return engine_ != nullptr; }

    // ---- 获取某个 I/O tensor 的维度 ----
    std::vector<int64_t> getTensorDims(int io_idx) const;

private:
    void release();

    // ---- TRT 对象 ----
    TRTLogger                    logger_;
    nvinfer1::IRuntime*          runtime_  = nullptr;
    nvinfer1::ICudaEngine*       engine_   = nullptr;
    nvinfer1::IExecutionContext* context_  = nullptr;

    // ---- I/O 信息 ----
    std::vector<void*>   buffers_;       // GPU buffer 指针，和 tensor 编号一一对应
    std::vector<size_t>  buffer_sizes_;  // 每个 buffer 的字节数
    int num_inputs_  = 0;
    int num_outputs_ = 0;
    int input_idx_   = -1;              // 第一个输入的 tensor 索引
    int output_idx_  = -1;              // 第一个输出的 tensor 索引
};
