#pragma once

#include "frame_queue.hpp"
#include "cuda_preprocess.h"
#include "trt_engine.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <vector>

// =========================================================================
// 一个检测结果
// =========================================================================
struct Detection {
    float x1, y1, x2, y2;   // 边界框（像素坐标，对应原始帧尺寸）
    float conf;              // 置信度
    int   class_id = 0;

    Detection() = default;
    Detection(float x1_, float y1_, float x2_, float y2_, float conf_, int cls = 0)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), conf(conf_), class_id(cls) {}
};

// =========================================================================
// 流水线中流转的一帧
// =========================================================================
struct Frame {
    int      id = 0;
    cv::Mat  cpu_img;                       // 原始帧（BGR uint8）— 用于最终显示
    int      src_w = 0, src_h = 0;          // 原始帧尺寸

    cuda_cv::PreprocessBuffers pp_buf;      // 本帧专用的 GPU 预处理 buffer
    float* d_output = nullptr;              // GPU buffer: 推理输出（大小 = 引擎输出字节数）
    size_t d_output_bytes = 0;

    ~Frame() { if (d_output) cudaFree(d_output); }
    Frame() = default;
    Frame(const Frame&) = delete;
    Frame(Frame&& other) noexcept
        : id(other.id), cpu_img(std::move(other.cpu_img)),
          src_w(other.src_w), src_h(other.src_h),
          pp_buf(other.pp_buf), d_output(other.d_output),
          d_output_bytes(other.d_output_bytes),
          detections(std::move(other.detections)) {
        other.d_output = nullptr;
        other.d_output_bytes = 0;
        other.pp_buf.d_src = nullptr;
        other.pp_buf.d_dst = nullptr;
    }

    std::vector<Detection> detections;      // 解析后的检测结果
};

using FramePtr = std::shared_ptr<Frame>;

// =========================================================================
// 对象池：预分配 N 个 Frame，流水线中循环使用
// =========================================================================
class FramePool {
public:
    FramePool(int count, int max_w, int max_h, size_t output_bytes = 0);
    FramePtr acquire();            // 借一个（阻塞）
    void     release(FramePtr f);  // 还回去

private:
    std::queue<FramePtr>      free_list_;
    std::mutex                mtx_;
    std::condition_variable   cv_;
    int                       total_;
};

// =========================================================================
// 四个工作线程的任务函数
// =========================================================================

// Thread 1: 视频帧抓取
void captureThread(const std::string& video_url,
                   FrameQueue<FramePtr>& out_queue,
                   FramePool& pool,
                   std::atomic<bool>& running);

// Thread 2: CUDA 预处理
void preprocessThread(FrameQueue<FramePtr>& in_queue,
                      FrameQueue<FramePtr>& out_queue,
                      std::atomic<bool>& running);

// Thread 3: TensorRT 推理
void inferThread(FrameQueue<FramePtr>& in_queue,
                 FrameQueue<FramePtr>& out_queue,
                 TensorRTEngine& engine,
                 std::atomic<bool>& running);

// Thread 4: 后处理 + 显示
void displayThread(FrameQueue<FramePtr>& in_queue,
                   FramePool& pool,
                   std::atomic<bool>& running,
                   float conf_threshold = 0.5f);

// =========================================================================
// 后处理：解析 YOLO 输出 + NMS
// =========================================================================
std::vector<Detection> parseYoloOutput(
    const float* output,     // GPU 数据 (1, 5, 8400) — 需先拷到 CPU
    int num_anchors,         // 8400
    int num_classes,         // 1（你的行人检测模型）
    int src_w, int src_h,    // 原始帧尺寸（用于坐标缩放）
    float conf_threshold = 0.5f,
    float nms_threshold = 0.45f
);
