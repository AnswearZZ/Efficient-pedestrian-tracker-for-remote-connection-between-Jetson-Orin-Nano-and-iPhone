// =============================================================================
// TensorRT 引擎封装 — 加载 .engine 文件、管理 GPU buffer、执行推理
// =============================================================================

#include "trt_engine.hpp"
#include <fstream>
#include <cstring>
#include <sstream>

// =========================================================================
// 工具：将 Dims 转为字符串（调试用）
// =========================================================================
static std::string dimsToString(const nvinfer1::Dims& d) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < d.nbDims; i++) {
        if (i > 0) oss << ", ";
        oss << d.d[i];
    }
    oss << ")";
    return oss.str();
}

// =========================================================================
// 工具：计算 tensor 字节大小
// =========================================================================
static size_t tensorBytes(nvinfer1::Dims dims, nvinfer1::DataType dtype) {
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; i++) count *= dims.d[i];
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return count * 4;
        case nvinfer1::DataType::kHALF:  return count * 2;
        case nvinfer1::DataType::kINT8:  return count * 1;
        case nvinfer1::DataType::kINT32: return count * 4;
        case nvinfer1::DataType::kBOOL:  return count * 1;
        default: return 0;
    }
}

// =========================================================================
// 移动构造 / 移动赋值
// =========================================================================
TensorRTEngine::TensorRTEngine(TensorRTEngine&& other) noexcept
    : runtime_(other.runtime_),
      engine_(other.engine_),
      context_(other.context_),
      buffers_(std::move(other.buffers_)),
      buffer_sizes_(std::move(other.buffer_sizes_)),
      num_inputs_(other.num_inputs_),
      num_outputs_(other.num_outputs_),
      input_idx_(other.input_idx_),
      output_idx_(other.output_idx_)
{
    // 清空源对象，防止析构时 double-free
    other.runtime_     = nullptr;
    other.engine_      = nullptr;
    other.context_     = nullptr;
    other.num_inputs_  = 0;
    other.num_outputs_ = 0;
    other.input_idx_   = -1;
    other.output_idx_  = -1;
}

TensorRTEngine& TensorRTEngine::operator=(TensorRTEngine&& other) noexcept {
    if (this != &other) {
        release();
        runtime_     = other.runtime_;
        engine_      = other.engine_;
        context_     = other.context_;
        buffers_     = std::move(other.buffers_);
        buffer_sizes_ = std::move(other.buffer_sizes_);
        num_inputs_  = other.num_inputs_;
        num_outputs_ = other.num_outputs_;
        input_idx_   = other.input_idx_;
        output_idx_  = other.output_idx_;

        other.runtime_     = nullptr;
        other.engine_      = nullptr;
        other.context_     = nullptr;
        other.num_inputs_  = 0;
        other.num_outputs_ = 0;
        other.input_idx_   = -1;
        other.output_idx_  = -1;
    }
    return *this;
}

// =========================================================================
// 释放所有资源
// =========================================================================
void TensorRTEngine::release() {
    for (void* ptr : buffers_) {
        if (ptr) cudaFree(ptr);
    }
    buffers_.clear();
    buffer_sizes_.clear();

    delete context_;
    delete engine_;
    delete runtime_;

    context_  = nullptr;
    engine_   = nullptr;
    runtime_  = nullptr;
    input_idx_   = -1;
    output_idx_  = -1;
    num_inputs_  = 0;
    num_outputs_ = 0;
}

// =========================================================================
// 加载 engine 文件 + 分配 I/O buffer
// =========================================================================
bool TensorRTEngine::load(const std::string& engine_path) {
    // 1. 读取 .engine 文件到内存
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[TensorRT] Failed to open: " << engine_path << std::endl;
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    if (!file.read(blob.data(), size)) {
        std::cerr << "[TensorRT] Failed to read: " << engine_path << std::endl;
        return false;
    }
    file.close();
    std::cout << "[TensorRT] Loaded " << size / 1024.0 / 1024.0 << " MiB from "
              << engine_path << std::endl;

    // 2. 创建 IRuntime
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        std::cerr << "[TensorRT] createInferRuntime failed" << std::endl;
        return false;
    }

    // 3. 反序列化 engine
    engine_ = runtime_->deserializeCudaEngine(blob.data(), size);
    if (!engine_) {
        std::cerr << "[TensorRT] deserializeCudaEngine failed" << std::endl;
        release();
        return false;
    }

    // 4. 创建 ExecutionContext
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "[TensorRT] createExecutionContext failed" << std::endl;
        release();
        return false;
    }

    // 5. 遍历所有 I/O tensor，分配 GPU buffer
    int nb_tensors = engine_->getNbIOTensors();
    std::cout << "[TensorRT] Engine has " << nb_tensors << " I/O tensor(s):" << std::endl;

    buffers_.resize(nb_tensors, nullptr);
    buffer_sizes_.resize(nb_tensors, 0);
    num_inputs_  = 0;
    num_outputs_ = 0;

    for (int i = 0; i < nb_tensors; i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode loc = engine_->getTensorIOMode(name);
        nvinfer1::Dims     dims = engine_->getTensorShape(name);
        nvinfer1::DataType dtype = engine_->getTensorDataType(name);

        size_t bytes = tensorBytes(dims, dtype);
        buffer_sizes_[i] = bytes;

        cudaError_t err = cudaMalloc(&buffers_[i], bytes);
        if (err != cudaSuccess) {
            std::cerr << "[TensorRT] cudaMalloc failed for tensor '" << name
                      << "': " << cudaGetErrorString(err) << std::endl;
            release();
            return false;
        }

        // TRT 10.x: 用 setTensorAddress 注册 buffer
        context_->setTensorAddress(name, buffers_[i]);

        const char* io_tag = (loc == nvinfer1::TensorIOMode::kINPUT) ? "IN " : "OUT";
        if (loc == nvinfer1::TensorIOMode::kINPUT) {
            num_inputs_++;
            if (input_idx_ < 0) input_idx_ = i;
        } else {
            num_outputs_++;
            if (output_idx_ < 0) output_idx_ = i;
        }

        std::cout << "  [" << io_tag << "] #" << i << " '" << name << "' "
                  << dimsToString(dims) << " " << bytes / 1024.0f << " KiB" << std::endl;
    }

    std::cout << "[TensorRT] Engine loaded OK — "
              << num_inputs_ << " input(s), " << num_outputs_ << " output(s)" << std::endl;
    return true;
}

// =========================================================================
// 执行推理
// =========================================================================
bool TensorRTEngine::infer(cudaStream_t stream) {
    if (!context_) {
        std::cerr << "[TensorRT] infer: engine not loaded" << std::endl;
        return false;
    }
    return context_->enqueueV3(stream);
}

// =========================================================================
// 查询某个 I/O tensor 的维度
// =========================================================================
std::vector<int64_t> TensorRTEngine::getTensorDims(int io_idx) const {
    if (!engine_ || io_idx < 0 || io_idx >= engine_->getNbIOTensors())
        return {};
    const char* name = engine_->getIOTensorName(io_idx);
    nvinfer1::Dims d = engine_->getTensorShape(name);
    return std::vector<int64_t>(d.d, d.d + d.nbDims);
}
