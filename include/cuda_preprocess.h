#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace cuda_cv {

// =========================================================================
// 常量：YOLO 输入尺寸
// =========================================================================
constexpr int kYoloInputW = 640;
constexpr int kYoloInputH = 640;
constexpr int kYoloInputC = 3;

// =========================================================================
// 预分配 GPU buffer（程序启动时调用一次，帧循环中复用）
// =========================================================================
struct PreprocessBuffers {
    uint8_t* d_src  = nullptr;  // 源帧 GPU buffer，大小 = max_h * max_w * 3
    float*   d_dst  = nullptr;  // 输出 (1,3,640,640) float32 [0,1] CHW
    int      max_w  = 0;
    int      max_h  = 0;
};

// 分配：传入视频帧可能的最大尺寸（建议 1920×1080 足够）
PreprocessBuffers allocate_preprocess_buffers(int max_width, int max_height);

// 释放
void free_preprocess_buffers(PreprocessBuffers& buf);

// =========================================================================
// 核心：上传源帧 + letterbox 预处理，一步完成
// =========================================================================
// h_src   : CPU 端 BGR 帧数据 uint8*（cv::Mat::data）
// src_w/h : 帧尺寸
// buf     : 预分配的 GPU buffer
// stream  : CUDA stream，传 nullptr 用默认流
void preprocess_yolo(
    const uint8_t* h_src,
    int src_width, int src_height,
    PreprocessBuffers& buf,
    cudaStream_t stream = nullptr
);

}  // namespace cuda_cv
