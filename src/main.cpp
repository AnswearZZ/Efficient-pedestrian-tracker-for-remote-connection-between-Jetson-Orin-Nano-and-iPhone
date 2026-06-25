// =============================================================================
// PedestrianTracker — 实时行人检测流水线入口
// =============================================================================
// 架构概览：
//   本文件负责解析命令行参数、加载 TensorRT 引擎、创建 FramePool 和
//   FrameQueue，然后启动 4 个流水线线程：
//
//   Thread 1 (capture)    : HTTP 视频流帧抓取     → q1
//   Thread 2 (preprocess) : CUDA GPU 预处理        → q2
//   Thread 3 (infer)      : TensorRT 推理          → q3
//   Thread 4 (display)    : 后处理 + OpenCV 显示
//
//   所有 GPU/CPU 内存在 FramePool 中预分配，运行时零动态分配。
//
//   退出方式：按 'q' 键 或 Ctrl+C
// =============================================================================

#include "worker_threads.hpp"
#include "trt_engine.hpp"
#include <csignal>

// 全局运行标志，Ctrl+C → signalHandler → g_running = false → 所有线程退出
static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);

    // =========================================================================
    // 命令行参数（均可通过命令行覆盖，未提供时使用默认值）
    // =========================================================================
    // 默认视频源：手机 IP 摄像头 HTTP 流（请替换为你的手机 IP）
    const char* video_url   = "http://192.168.50.127:4747/video";
    // 默认模型路径：YOLO11n INT8 engine 文件
    const char* engine_path = "best_int8.engine";
    // 默认置信度阈值：INT8 量化后置信度偏低，使用 0.45
    float       conf_thr    = 0.45f;

    if (argc >= 2) video_url   = argv[1];
    if (argc >= 3) engine_path = argv[2];
    if (argc >= 4) conf_thr    = std::atof(argv[3]);

    // 打印启动信息
    std::cout << "==============================================" << std::endl;
    std::cout << "  Pedestrian Tracker (C++ Pipeline)" << std::endl;
    std::cout << "  Video : " << video_url << std::endl;
    std::cout << "  Engine: " << engine_path << std::endl;
    std::cout << "  Conf  : " << conf_thr << std::endl;
    std::cout << "==============================================" << std::endl;

    // =========================================================================
    // 步骤 1：加载 TensorRT 引擎
    // =========================================================================
    TensorRTEngine engine;
    if (!engine.load(engine_path)) {
        std::cerr << "[Main] Engine load failed" << std::endl;
        return 1;
    }

    // 根据引擎输出 tensor 维度，计算每个 Frame 需要的输出 buffer 大小
    // 例：YOLO11n 输出 (1, 5, 8400) → out_bytes = 1 × 5 × 8400 × 4 = 168 KB
    auto out_dims = engine.getTensorDims(engine.numInputs());
    size_t out_bytes = 1;
    for (auto d : out_dims) out_bytes *= d;
    out_bytes *= sizeof(float);

    // =========================================================================
    // 步骤 2：创建流水线组件
    // =========================================================================
    // FramePool：预分配 4 个 Frame（每个含 GPU 预处理 buffer + GPU 输出 buffer
    //           + CPU 图像 Mat），流水线中循环复用，避免运行时 cudaMalloc
    const int kPoolSize  = 4;
    // FrameQueue：每级队列容量 2，满时自动丢旧帧 → 背压控制，保证实时性
    const int kQueueSize = 2;
    // 支持的最大视频帧尺寸（超出会自动裁剪或缩放）
    const int kMaxW = 1920, kMaxH = 1080;

    FramePool pool(kPoolSize, kMaxW, kMaxH, out_bytes);

    // 三级队列：q1(抓取→预处理), q2(预处理→推理), q3(推理→显示)
    FrameQueue<FramePtr> q1(kQueueSize);
    FrameQueue<FramePtr> q2(kQueueSize);
    FrameQueue<FramePtr> q3(kQueueSize);

    // =========================================================================
    // 步骤 3：启动 4 个流水线线程
    // =========================================================================
    std::cout << "\n[Main] Starting pipeline...\n" << std::endl;

    std::thread t1(captureThread,     video_url,
                   std::ref(q1), std::ref(pool), std::ref(g_running));
    std::thread t2(preprocessThread,  std::ref(q1), std::ref(q2),
                   std::ref(g_running));
    std::thread t3(inferThread,       std::ref(q2), std::ref(q3),
                   std::ref(engine), std::ref(g_running));
    std::thread t4(displayThread,     std::ref(q3), std::ref(pool),
                   std::ref(g_running), conf_thr);

    std::cout << "[Main] Pipeline running. Press 'q' or Ctrl+C to exit.\n" << std::endl;

    // =========================================================================
    // 步骤 4：等待退出
    // display 线程退出（用户按 'q' 或 Ctrl+C）后，依次停止所有线程
    // =========================================================================
    t4.join();              // 等待显示线程退出
    g_running = false;      // 通知所有线程停止
    q1.stop(); q2.stop(); q3.stop();  // 唤醒所有阻塞在队列上的线程
    t1.join(); t2.join(); t3.join();  // 等待其他线程退出

    std::cout << "\n[Main] Pipeline stopped cleanly." << std::endl;
    return 0;
}
