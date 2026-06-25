// =============================================================================
// CUDA GPU 预处理 — letterbox + BGR→RGB + 归一化 + HWC→CHW 单次 kernel 完成
// =============================================================================

#include "cuda_preprocess.h"
#include <cstdio>

namespace {

// =========================================================================
// 双线性插值：从源图指定通道采样一个浮点值，同时归一化到 [0,1]
// src       : GPU 端 BGR uint8 (HWC, interleaved)
// w, h      : 源图尺寸
// step      : 源图一行字节数 = w * 3
// x, y      : 源图上的浮点坐标
// channel   : 0=B, 1=G, 2=R
// =========================================================================
__device__ float bilinear_sample(
    const uint8_t* src, int w, int h, int step,
    float x, float y, int channel)
{
    // 边界 clamp（防止越界）
    x = fminf(fmaxf(x, 0.0f), w - 1.0f - 1e-5f);
    y = fminf(fmaxf(y, 0.0f), h - 1.0f - 1e-5f);

    int   x0 = (int)floorf(x);
    int   y0 = (int)floorf(y);
    int   x1 = x0 + 1;
    int   y1 = y0 + 1;
    float wx = x - (float)x0;
    float wy = y - (float)y0;

    // 读 4 个邻居像素的指定通道值
    // BGR 交错存储：每像素 3 字节，offset = y * step + x * 3 + channel
    float v00 = (float)src[y0 * step + x0 * 3 + channel];
    float v10 = (float)src[y0 * step + x1 * 3 + channel];
    float v01 = (float)src[y1 * step + x0 * 3 + channel];
    float v11 = (float)src[y1 * step + x1 * 3 + channel];

    // 水平 + 垂直混合 + 归一化
    float top    = v00 * (1.0f - wx) + v10 * wx;
    float bottom = v01 * (1.0f - wx) + v11 * wx;
    return (top * (1.0f - wy) + bottom * wy) / 255.0f;
}

// =========================================================================
// 主 kernel：letterbox resize + BGR→RGB + 归一化 + HWC→CHW，一次完成
// =========================================================================
__global__ void preprocess_kernel(
    const uint8_t* src,     // GPU 端 BGR uint8, HWC
    int src_w, int src_h, int src_step,
    float* dst,             // 输出 (3, dst_h, dst_w) float32 CHW
    int dst_w, int dst_h,
    float scale, int left, int top, int new_w, int new_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int plane_size = dst_w * dst_h;
    float* r_plane = dst;                      // channel 0 = Red
    float* g_plane = dst + plane_size;         // channel 1 = Green
    float* b_plane = dst + plane_size * 2;     // channel 2 = Blue

    float r, g, b;

    // 判断该像素在 letterbox 有效区域还是 padding 区域
    if (x >= left && x < left + new_w && y >= top && y < top + new_h) {
        // 映射回源图坐标（对齐 OpenCV 的 half-pixel center 约定）
        float src_x = ((float)(x - left) + 0.5f) / scale - 0.5f;
        float src_y = ((float)(y - top)  + 0.5f) / scale - 0.5f;

        // BGR→RGB 采样
        r = bilinear_sample(src, src_w, src_h, src_step, src_x, src_y, 2);
        g = bilinear_sample(src, src_w, src_h, src_step, src_x, src_y, 1);
        b = bilinear_sample(src, src_w, src_h, src_step, src_x, src_y, 0);
    } else {
        // 灰边填充 (114/255, 114/255, 114/255)
        r = 114.0f / 255.0f;
        g = 114.0f / 255.0f;
        b = 114.0f / 255.0f;
    }

    int idx = y * dst_w + x;
    r_plane[idx] = r;
    g_plane[idx] = g;
    b_plane[idx] = b;
}

}  // anonymous namespace


namespace cuda_cv {

// =========================================================================
// Buffer 管理
// =========================================================================
PreprocessBuffers allocate_preprocess_buffers(int max_w, int max_h) {
    PreprocessBuffers buf;
    buf.max_w = max_w;
    buf.max_h = max_h;

    size_t src_bytes = (size_t)max_w * max_h * 3;              // uint8 BGR
    size_t dst_bytes = (size_t)kYoloInputC * kYoloInputW
                       * kYoloInputH * sizeof(float);           // float32 CHW

    cudaError_t err;
    err = cudaMalloc(&buf.d_src, src_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[cuda_preprocess] cudaMalloc d_src failed: %s\n",
                     cudaGetErrorString(err));
        return buf;
    }
    err = cudaMalloc(&buf.d_dst, dst_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[cuda_preprocess] cudaMalloc d_dst failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(buf.d_src);
        buf.d_src = nullptr;
    }
    return buf;
}

void free_preprocess_buffers(PreprocessBuffers& buf) {
    if (buf.d_src) { cudaFree(buf.d_src); buf.d_src = nullptr; }
    if (buf.d_dst) { cudaFree(buf.d_dst); buf.d_dst = nullptr; }
}

// =========================================================================
// 预处理入口：上传帧 + 启动 kernel
// =========================================================================
void preprocess_yolo(
    const uint8_t* h_src,
    int src_width, int src_height,
    PreprocessBuffers& buf,
    cudaStream_t stream)
{
    // --- 1. 计算 letterbox 参数（CPU 端，计算量可忽略）---
    float scale = fminf((float)kYoloInputW / (float)src_width,
                        (float)kYoloInputH / (float)src_height);
    int new_w = (int)((float)src_width  * scale);
    int new_h = (int)((float)src_height * scale);
    int left   = (kYoloInputW - new_w) / 2;
    int top    = (kYoloInputH - new_h) / 2;
    int src_step = src_width * 3;  // BGR 交错，每行 src_width * 3 字节

    // --- 2. 上传源帧到 GPU ---
    size_t src_bytes = (size_t)src_height * src_step;
    cudaMemcpyAsync(buf.d_src, h_src, src_bytes,
                    cudaMemcpyHostToDevice, stream);

    // --- 3. 启动预处理 kernel ---
    dim3 block(16, 16);
    dim3 grid((kYoloInputW + block.x - 1) / block.x,
              (kYoloInputH + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        buf.d_src,
        src_width, src_height, src_step,
        buf.d_dst,
        kYoloInputW, kYoloInputH,
        scale, left, top, new_w, new_h
    );
}

}  // namespace cuda_cv
