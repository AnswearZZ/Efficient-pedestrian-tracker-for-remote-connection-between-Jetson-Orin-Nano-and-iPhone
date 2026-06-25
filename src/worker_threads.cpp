#include "worker_threads.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>

// =========================================================================
// FramePool
// =========================================================================
FramePool::FramePool(int count, int max_w, int max_h, size_t output_bytes)
    : total_(count)
{
    for (int i = 0; i < count; i++) {
        auto f = std::make_shared<Frame>();
        f->id  = i;
        f->pp_buf = cuda_cv::allocate_preprocess_buffers(max_w, max_h);
        f->cpu_img = cv::Mat(max_h, max_w, CV_8UC3);  // 预分配 CPU 内存

        // 分配推理输出 buffer（GPU）
        if (output_bytes > 0) {
            f->d_output_bytes = output_bytes;
            cudaError_t err = cudaMalloc(&f->d_output, output_bytes);
            if (err != cudaSuccess) {
                std::cerr << "[FramePool] cudaMalloc output failed: "
                          << cudaGetErrorString(err) << std::endl;
            }
        }

        free_list_.push(f);
    }
    std::cout << "[FramePool] Allocated " << count << " frames ("
              << max_w << "x" << max_h << ", output "
              << output_bytes / 1024.0f << " KiB each)" << std::endl;
}

FramePtr FramePool::acquire() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() { return !free_list_.empty(); });
    auto f = free_list_.front();
    free_list_.pop();
    return f;
}

void FramePool::release(FramePtr f) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        free_list_.push(f);
    }
    cv_.notify_one();
}

// =========================================================================
// 工具函数：sigmoid
// =========================================================================
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// =========================================================================
// Thread 1: 视频帧抓取
// =========================================================================
void captureThread(const std::string& video_url,
                   FrameQueue<FramePtr>& out_queue,
                   FramePool& pool,
                   std::atomic<bool>& running)
{
    cv::VideoCapture cap(video_url);
    if (!cap.isOpened()) {
        std::cerr << "[Capture] Failed to open: " << video_url << std::endl;
        running = false;
        out_queue.stop();
        return;
    }
    std::cout << "[Capture] Video stream opened: " << video_url << std::endl;

    while (running) {
        auto frame = pool.acquire();      // 从池里借一个 Frame

        if (!cap.read(frame->cpu_img)) {
            std::cerr << "[Capture] Read failed, stopping" << std::endl;
            pool.release(frame);
            break;
        }

        // CPU 端旋转 90°（和 Python 版本保持一致）
        cv::rotate(frame->cpu_img, frame->cpu_img, cv::ROTATE_90_CLOCKWISE);
        frame->src_w = frame->cpu_img.cols;
        frame->src_h = frame->cpu_img.rows;

        out_queue.push(frame);            // 交给预处理线程
    }

    cap.release();
    out_queue.stop();                     // 通知下游：没有新帧了
    std::cout << "[Capture] Stopped" << std::endl;
}

// =========================================================================
// Thread 2: CUDA 预处理
// =========================================================================
void preprocessThread(FrameQueue<FramePtr>& in_queue,
                      FrameQueue<FramePtr>& out_queue,
                      std::atomic<bool>& running)
{
    std::cout << "[Preprocess] Started" << std::endl;

    while (running) {
        FramePtr frame;
        if (!in_queue.pop(frame)) break;  // 队列已 stop 且空 → 退出

        // 执行 GPU 预处理（上传 + letterbox + BGR→RGB + 归一化 + CHW）
        cuda_cv::preprocess_yolo(
            frame->cpu_img.data,
            frame->src_w, frame->src_h,
            frame->pp_buf
        );

        out_queue.push(frame);            // 交给推理线程
    }

    out_queue.stop();
    std::cout << "[Preprocess] Stopped" << std::endl;
}

// =========================================================================
// Thread 3: TensorRT 推理
// =========================================================================
void inferThread(FrameQueue<FramePtr>& in_queue,
                 FrameQueue<FramePtr>& out_queue,
                 TensorRTEngine& engine,
                 std::atomic<bool>& running)
{
    // 创建独立 CUDA stream，避免 TRT 用 default stream 触发额外同步
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "[Infer] Started" << std::endl;

    // FPS 计时
    auto infer_begin = std::chrono::steady_clock::now();
    int  infer_count = 0;

    while (running) {
        FramePtr frame;
        if (!in_queue.pop(frame)) break;

        // 拷贝预处理结果 → engine input buffer
        size_t in_bytes = 1 * cuda_cv::kYoloInputC
                            * cuda_cv::kYoloInputW
                            * cuda_cv::kYoloInputH * sizeof(float);
        cudaMemcpyAsync(engine.inputPtr(), frame->pp_buf.d_dst,
                        in_bytes, cudaMemcpyDeviceToDevice, stream);

        // 执行推理
        if (!engine.infer(stream)) {
            std::cerr << "[Infer] Inference failed on frame " << frame->id << std::endl;
            continue;
        }

        // 把引擎输出拷到 Frame 的专属 buffer（避免下一帧覆盖）
        cudaMemcpyAsync(frame->d_output, engine.outputPtr(),
                        frame->d_output_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        out_queue.push(frame);            // 交给后处理线程

        // 每秒打印推理 FPS
        infer_count++;
        auto now = std::chrono::steady_clock::now();
        float sec = std::chrono::duration<float>(now - infer_begin).count();
        if (sec >= 2.0f) {
            float infer_fps = infer_count / sec;
            float infer_ms  = 1000.0f / infer_fps;
            std::cout << "[Infer] " << infer_count << " frames in "
                      << sec << "s → " << infer_fps << " FPS ("
                      << infer_ms << " ms/frame)" << std::endl;
            infer_begin = now;
            infer_count = 0;
        }
    }

    cudaStreamDestroy(stream);
    out_queue.stop();
    std::cout << "[Infer] Stopped" << std::endl;
}

// =========================================================================
// Thread 4: 后处理 + 显示
// =========================================================================
void displayThread(FrameQueue<FramePtr>& in_queue,
                   FramePool& pool,
                   std::atomic<bool>& running,
                   float conf_threshold)
{
    // FPS 统计
    auto fps_start = std::chrono::steady_clock::now();
    int  fps_count = 0;
    float fps_avg   = 0.0f;

    cv::namedWindow("Pedestrian Tracker - C++", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Pedestrian Tracker - C++", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    std::cout << "[Display] Started" << std::endl;

    while (running) {
        FramePtr frame;
        if (!in_queue.tryPop(frame)) {
            // 暂时没有帧，等 1ms 再试
            if (!running) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // --- 1. 从 Frame 的 GPU 输出 buffer 拷到 CPU ---
        // output0 shape: (1, 5, 8400) for 1-class model
        const int num_anchors = 8400;
        const int num_outputs = 5;       // cx, cy, w, h, conf
        size_t out_count = num_anchors * num_outputs;
        std::vector<float> h_out(out_count);
        cudaMemcpy(h_out.data(), frame->d_output,
                   out_count * sizeof(float), cudaMemcpyDeviceToHost);

        // --- 2. 解析 YOLO 输出 + NMS ---
        frame->detections = parseYoloOutput(
            h_out.data(), num_anchors, 1,  // 1 class
            frame->src_w, frame->src_h,
            conf_threshold
        );

        // --- 3. 画框 ---
        for (const auto& det : frame->detections) {
            cv::rectangle(frame->cpu_img,
                          cv::Point((int)det.x1, (int)det.y1),
                          cv::Point((int)det.x2, (int)det.y2),
                          cv::Scalar(0, 255, 0), 2);
            char label[64];
            std::snprintf(label, sizeof(label), "person %.2f", det.conf);
            cv::putText(frame->cpu_img, label,
                        cv::Point((int)det.x1, (int)det.y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 2);
        }

        // --- 4. FPS 计算 ---
        fps_count++;
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - fps_start).count();
        if (elapsed >= 1.0f) {
            fps_avg = (float)fps_count / elapsed;
            fps_count = 0;
            fps_start = now;
        }
        cv::putText(frame->cpu_img,
                    "FPS: " + std::to_string((int)fps_avg),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);

        // --- 5. 保持宽高比缩放 + 黑边填充到 1024x600 ---
        const int kDispW = 1024, kDispH = 600;
        cv::Mat display_img(kDispH, kDispW, CV_8UC3, cv::Scalar(0, 0, 0));

        int iw = frame->cpu_img.cols;
        int ih = frame->cpu_img.rows;
        float scale = std::min((float)kDispW / iw, (float)kDispH / ih);
        int nw = (int)(iw * scale);
        int nh = (int)(ih * scale);

        cv::Mat resized;
        cv::resize(frame->cpu_img, resized, cv::Size(nw, nh));

        // 居中放置
        int x = (kDispW - nw) / 2;
        int y = (kDispH - nh) / 2;
        resized.copyTo(display_img(cv::Rect(x, y, nw, nh)));

        cv::imshow("Pedestrian Tracker - C++", display_img);

        // --- 6. 归还 Frame 到池里 ---
        pool.release(frame);

        // 按 'q' 退出
        if (cv::waitKey(1) == 'q') {
            running = false;
            break;
        }
    }

    cv::destroyAllWindows();
    std::cout << "[Display] Stopped" << std::endl;
}

// =========================================================================
// YOLO 输出解析 + NMS
//
// 输出格式说明（YOLO11, 1 class, raw detection head）：
//   output shape = (1, 5, 8400)
//   layout = [cx, cy, w, h, conf] * 8400
//   cx,cy,w,h 为 640×640 输入空间下的像素坐标（已解码）
//   conf 为原始 logit，需过 sigmoid
//
// 如果你的模型实际输出格式不同（比如坐标是归一化的、或者
// 需要 anchor 解码），请按实际观察调整下面的解析逻辑。
// =========================================================================
std::vector<Detection> parseYoloOutput(
    const float* output,
    int num_anchors,
    int num_classes,
    int src_w, int src_h,
    float conf_threshold,
    float nms_threshold)
{
    std::vector<Detection> raw;

    // 缩放因子：640×640 输入 → 原始帧尺寸
    float scale_x = (float)src_w / 640.0f;
    float scale_y = (float)src_h / 640.0f;

    // YOLO 输出是 CHW 布局: (5, num_anchors)
    // 每 channel 连续存储 num_anchors 个值，而不是 5 个值一组交错排列
    const float* cx_ptr   = output;                      // channel 0
    const float* cy_ptr   = output + num_anchors;        // channel 1
    const float* w_ptr    = output + num_anchors * 2;    // channel 2
    const float* h_ptr    = output + num_anchors * 3;    // channel 3
    const float* conf_ptr = output + num_anchors * 4;    // channel 4

    for (int i = 0; i < num_anchors; i++) {
        float cx  = cx_ptr[i];
        float cy  = cy_ptr[i];
        float w   = w_ptr[i];
        float h   = h_ptr[i];
        float raw_conf = conf_ptr[i];

        // 置信度：ONNX 导出已内置 sigmoid，直接使用
        float conf = raw_conf;

        if (conf < conf_threshold) continue;

        // 转为 corner 格式，缩放到原始帧尺寸
        float x1 = (cx - w * 0.5f) * scale_x;
        float y1 = (cy - h * 0.5f) * scale_y;
        float x2 = (cx + w * 0.5f) * scale_x;
        float y2 = (cy + h * 0.5f) * scale_y;

        // 裁剪到图像范围内
        x1 = std::max(0.0f, std::min(x1, (float)src_w - 1.0f));
        y1 = std::max(0.0f, std::min(y1, (float)src_h - 1.0f));
        x2 = std::max(0.0f, std::min(x2, (float)src_w - 1.0f));
        y2 = std::max(0.0f, std::min(y2, (float)src_h - 1.0f));

        if (x2 <= x1 || y2 <= y1) continue;

        raw.emplace_back(x1, y1, x2, y2, conf, 0);
    }

    // --- NMS (使用 OpenCV 内置) ---
    std::vector<Detection> result;
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float>  scores;

    boxes.reserve(raw.size());
    scores.reserve(raw.size());
    for (const auto& d : raw) {
        boxes.emplace_back((int)d.x1, (int)d.y1,
                           (int)(d.x2 - d.x1), (int)(d.y2 - d.y1));
        scores.push_back(d.conf);
    }

    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        result.push_back(raw[idx]);
    }

    return result;
}
