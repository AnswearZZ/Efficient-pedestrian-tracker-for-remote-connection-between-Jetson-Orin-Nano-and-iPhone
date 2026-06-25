#include "worker_threads.hpp"
#include "trt_engine.hpp"
#include <csignal>

static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);

    // ---- 参数（可通过命令行覆盖）----1
    const char* video_url   = "http://192.168.50.127:4747/video";
    const char* engine_path = "/home/zhouco/human_test/yolo11n_human/best_int8.engine";
    float       conf_thr    = 0.45f;   // INT8 量化后置信度偏低，阈值降低
    if (argc >= 2) video_url   = argv[1];
    if (argc >= 3) engine_path = argv[2];
    if (argc >= 4) conf_thr    = std::atof(argv[3]);

    std::cout << "==============================================" << std::endl;
    std::cout << "  Pedestrian Tracker (C++ Pipeline)" << std::endl;
    std::cout << "  Video : " << video_url << std::endl;
    std::cout << "  Engine: " << engine_path << std::endl;
    std::cout << "  Conf  : " << conf_thr << std::endl;
    std::cout << "==============================================" << std::endl;

    // ---- 1. 加载引擎 ----
    TensorRTEngine engine;
    if (!engine.load(engine_path)) {
        std::cerr << "[Main] Engine load failed" << std::endl;
        return 1;
    }

    // 计算输出 buffer 大小
    auto out_dims = engine.getTensorDims(engine.numInputs());
    size_t out_bytes = 1;
    for (auto d : out_dims) out_bytes *= d;
    out_bytes *= sizeof(float);

    // ---- 2. 创建流水线组件 ----
    const int kPoolSize  = 4;
    const int kQueueSize = 2;
    const int kMaxW = 1920, kMaxH = 1080;

    FramePool pool(kPoolSize, kMaxW, kMaxH, out_bytes);

    FrameQueue<FramePtr> q1(kQueueSize);
    FrameQueue<FramePtr> q2(kQueueSize);
    FrameQueue<FramePtr> q3(kQueueSize);

    // ---- 3. 启动 4 线程 ----
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

    // ---- 4. 等待退出 ----
    t4.join();
    g_running = false;
    q1.stop(); q2.stop(); q3.stop();
    t1.join(); t2.join(); t3.join();

    std::cout << "\n[Main] Pipeline stopped cleanly." << std::endl;
    return 0;
}
