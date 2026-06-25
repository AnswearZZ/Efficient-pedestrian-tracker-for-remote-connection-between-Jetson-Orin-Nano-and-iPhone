<p align="center">
  <img src="https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-12-green?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-10-red?logo=nvidia" alt="TensorRT">
  <img src="https://img.shields.io/badge/OpenCV-4-purple?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/platform-Jetson%20Orin%20Nano-brightgreen?logo=nvidia" alt="Jetson Orin Nano">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
</p>

<h1 align="center">🚶 Efficient Pedestrian Tracker</h1>
<h3 align="center">高效行人检测器</h3>

<p align="center">
  <b>Remote Connection Between Jetson Orin Nano and iPhone</b><br>
  Jetson Orin Nano 与 iPhone 远程连接
</p>

<p align="center">
  Real-time pedestrian detection pipeline powered by <b>TensorRT + CUDA</b><br>
  基于 <b>TensorRT + CUDA</b> 的实时行人检测流水线<br>
  Built for NVIDIA Jetson Orin Nano edge computing platform<br>
  专为 NVIDIA Jetson Orin Nano 边缘计算平台设计
</p>

<p align="center">
  <img src="show/demo-screenshot.jpg" alt="Demo Screenshot" width="800">
</p>
<p align="center">
  <i>Screenshot: Pedestrian detection with real-time FPS overlay &nbsp;|&nbsp; 运行截图：行人检测 + 实时 FPS 叠加显示</i><br>
  <i>Jetson Orin Nano ← HTTP stream ← iPhone camera &nbsp;|&nbsp; Jetson Orin Nano ← HTTP 视频流 ← iPhone 摄像头</i>
</p>

---

## 🎬 Demo / 演示

<p align="center">
  <a href="https://github.com/user-attachments/assets/924217ed-41b0-4310-9d15-6977ca28ff25">
    <img src="show/demo-screenshot.jpg" alt="Demo Video" width="640">
  </a>
</p>

<p align="center">
  <a href="https://github.com/user-attachments/assets/924217ed-41b0-4310-9d15-6977ca28ff25">
    <b>▶️ Click to Play Demo Video &nbsp;/&nbsp; 点击播放演示视频</b>
  </a>
  <br>
  <sub>Jetson Orin Nano + iPhone · HTTP video stream · Real-time detection <b>~60 Hz</b></sub>
  <br>
  <sub>Jetson Orin Nano + iPhone · HTTP 视频流 · 实时检测 <b>~60 Hz</b></sub>
</p>

---

## ✨ Features / 特性

| EN | 中文 |
|---|---|
| 🚀 **4-Thread Pipeline** — Capture, CUDA preprocess, TensorRT inference, and display run in parallel | 🚀 **四线程流水线** — 视频抓取、CUDA 预处理、TensorRT 推理、后处理显示并行执行 |
| 🎯 **YOLO11n INT8 Model** — Single-class pedestrian detection with maximum throughput | 🎯 **YOLO11n INT8 量化** — 单类行人检测，极致推理速度 |
| ⚡ **Zero-copy GPU Preprocess** — Letterbox + BGR→RGB + Normalize + CHW fused in a single CUDA kernel | ⚡ **零拷贝 GPU 预处理** — letterbox + BGR→RGB + 归一化 + CHW 转换在单次 CUDA kernel 中完成 |
| 🔄 **FramePool Reuse** — All GPU/CPU buffers pre-allocated at startup, zero runtime allocation | 🔄 **对象池复用** — 启动时预分配全部 GPU/CPU 内存，运行时零分配 |
| 📊 **Real-time FPS** — End-to-end frame rate overlaid on display | 📊 **实时 FPS 统计** — 端到端帧率叠加显示在画面 |
| 🎮 **Plug & Play** — HTTP video stream (phone IP camera), configurable via CLI | 🎮 **即插即用** — 支持 HTTP 视频流（手机 IP 摄像头），命令行参数配置 |

---

## 🏗️ Architecture / 系统架构

```
                        ┌───────────────────────────────────────────┐
                        │              FramePool / 对象池             │
                        │       4 Frames pre-allocated / 预分配       │
                        └──────┬──────────────────────┬─────────────┘
                               │ acquire()            │ release()
                               ▼                      ▲
┌──────────┐    q1     ┌──────────┐    q2    ┌──────────┐    q3    ┌──────────┐
│ Thread 1 │ ────────► │ Thread 2 │ ───────► │ Thread 3 │ ───────► │ Thread 4 │
│ Capture  │          │Preprocess │          │  Infer   │          │ Display  │
│ 视频抓取  │          │ CUDA预处理 │          │ TRT 推理  │          │ 后处理显示 │
└──────────┘          └──────────┘          └──────────┘          └──────────┘
     │                     │                     │                     │
     │ HTTP stream         │ GPU: letterbox     │ GPU: enqueueV3     │ CPU: NMS + draw
     │ cv::VideoCapture   │ + BGR→RGB          │ D2D memcpy         │ cv::imshow
     │ CPU rotate 90°     │ + Normalize + CHW  │ + inference        │ + FPS counter
     ▼                     ▼                     ▼                     ▼
┌──────────┐          ┌──────────┐          ┌──────────┐          ┌──────────┐
│  Phone   │          │  CUDA    │          │ TensorRT │          │  Screen  │
│  手机摄像头│          │  Kernel  │          │  Engine  │          │  显示器   │
└──────────┘          └──────────┘          └──────────┘          └──────────┘
```

### Pipeline Stages / 流水线阶段

| Thread / 线程 | Function / 函数 | Input / 输入 | Output / 输出 | Task / 工作内容 |
|---|---|---|---|---|
| **T1** Capture | `captureThread` | HTTP stream | → q1 | Open stream, read frames, CPU rotate 90° |
| **T2** Preprocess | `preprocessThread` | q1 → | → q2 | H2D upload, CUDA kernel preprocessing |
| **T3** Infer | `inferThread` | q2 → | → q3 | D2D copy input, `enqueueV3` inference |
| **T4** Display | `displayThread` | q3 → | → screen | Parse output → NMS → Draw boxes → Show |

---

## 📋 Prerequisites / 前置依赖

| Library / 库 | Version / 版本 | Purpose / 用途 |
|---|---|---|
| **CUDA** | ≥ 11.4 | GPU memory management, CUDA Stream / GPU 内存管理、CUDA Stream |
| **TensorRT** | ≥ 10.x | Model loading & inference / 模型加载与推理 (`libnvinfer`, `libnvinfer_plugin`) |
| **OpenCV** | ≥ 4.5 | Video capture, display, NMS / 视频流读取、图像显示、NMS |
| **CMake** | ≥ 3.18 | Build system / 构建系统 |
| **GCC/G++** | ≥ 9 (C++17) | Compiler / 编译器 |
| **pthread** | built-in | Multithreading / 多线程 |

> Recommended: JetPack 6.x on Jetson Orin Nano includes all dependencies above.<br>
> 推荐使用 JetPack 6.x，已预装上述所有依赖。

### Model Preparation / 模型准备

A TensorRT `.engine` file is required. Default config targets **YOLO11n single-class (person) INT8**:<br>
需要 TensorRT `.engine` 文件。默认配置适用于 **YOLO11n 单类行人检测 INT8 模型**：

| Parameter / 参数 | Value / 值 |
|---|---|
| Model | YOLO11n (nano) |
| Classes / 类别 | 1 (person) |
| Quantization / 量化 | INT8 |
| Input / 输入 | 3×640×640 float32 |
| Output / 输出 | 1×5×8400 float32 |

```bash
# Export via Ultralytics / 使用 Ultralytics 导出
yolo export model=best.pt format=engine device=0 int8=True
```

---

## 🔧 Build / 构建

```bash
# 1. Clone / 克隆
git clone https://github.com/AnswearZZ/Efficient-pedestrian-tracker-for-remote-connection-between-Jetson-Orin-Nano-and-iPhone.git
cd Efficient-pedestrian-tracker-for-remote-connection-between-Jetson-Orin-Nano-and-iPhone

# 2. Configure (Release mode) / 配置（Release 模式）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. Compile / 编译
cmake --build build -j$(nproc)
```

Build artifacts / 编译产物：
- `build/PedestrianTracker` — Main program / 主程序
- `build/Benchmark` — Performance benchmark / 性能基准测试

---

## 🚀 Usage / 使用

### Quick Start / 基本启动

```bash
./build/PedestrianTracker
```

Default parameters / 默认参数：
- Video source / 视频源：`http://192.168.50.127:4747/video`（phone IP camera）
- Model / 模型：`best_int8.engine`

### Custom Parameters / 自定义参数

```bash
./build/PedestrianTracker <video-url> <engine-path> <conf-threshold>

# Example / 示例
./build/PedestrianTracker \
  "http://192.168.1.100:4747/video" \
  "best_int8.engine" \
  0.4
```

### Controls / 操作

| Action / 操作 | Method / 方法 |
|---|---|
| **Exit / 退出** | Press `q` or `Ctrl+C` / 按 `q` 键或 `Ctrl+C` |

### Phone Camera Setup / 手机摄像头设置

1. Install an IP camera app on your phone (e.g. **IP Webcam** for Android, **DroidCam** for iPhone)<br>
   在手机上安装 IP 摄像头 App（Android: IP Webcam，iPhone: DroidCam）
2. Start the app and note the HTTP stream URL (typically `http://<phone-ip>:4747/video`)<br>
   启动 App，记录 HTTP 视频流地址
3. Ensure Jetson and phone are on the same local network<br>
   确保 Jetson 和手机在同一局域网

---

## ⚡ Performance / 性能

> Test platform / 测试平台：NVIDIA Jetson Orin Nano, YOLO11n, 640×640

### C++ Pipeline / C++ 流水线（本项目）

| Stage / 阶段 | Latency / 耗时 | Notes / 说明 |
|---|---|---|
| Video capture | ~5-10 ms | Network-dependent / 受网络延迟影响 |
| GPU preprocess | ~1-2 ms | CUDA kernel parallel / CUDA kernel 并行 |
| TRT inference | ~8-15 ms | INT8 Tensor Core |
| Postprocess + display | ~3-8 ms | CPU NMS |
| **End-to-end / 端到端** | **~17 ms** | **Measured ~60 Hz / 实测约 60 Hz** 🚀 |

### Python Baseline / Python 基准（单线程 PyTorch CUDA）

Run `baseline_pytorch.py` as a performance floor reference:<br>
运行 `baseline_pytorch.py` 作为性能下限参考：

| Stage / 阶段 | Latency / 耗时 | Notes / 说明 |
|---|---|---|
| Capture + CPU preprocess | ~50-70 ms | ultralytics built-in CPU resize/normalize |
| PyTorch inference | ~20-40 ms | FP16 CUDA (no INT8, no TRT) |
| Postprocess + display | ~5-10 ms | ultralytics built-in plot |
| **End-to-end / 端到端** | **~100 ms** | **Measured ~10 Hz / 实测约 10 Hz** 🐢 |

### Comparison / 对比总结

| Dimension / 维度 | C++ Pipeline | Python Baseline (`baseline_pytorch.py`) |
|---|---|---|
| **Architecture / 架构** | 4-thread parallel / 四线程并行 | Single-thread serial / 单线程串行 |
| **Inference Engine** | TensorRT INT8 | PyTorch CUDA FP16 |
| **Preprocess / 预处理** | GPU CUDA kernel | CPU (ultralytics) |
| **Memory / 内存** | FramePool pre-allocated, zero runtime allocation | Per-frame dynamic allocation |
| **Measured FPS / 实测帧率** | **~60 Hz** 🚀 | **~10 Hz** 🐢 |
| **Speedup / 加速比** | **6×** | 1× (baseline / 基准) |
| **Deploy / 部署** | `./build/PedestrianTracker` | `python baseline_pytorch.py` |
| **Dependencies / 依赖** | TensorRT + OpenCV runtime only | ultralytics + PyTorch |

```bash
# Run Python baseline for comparison / 运行 Python 基准对比
python baseline_pytorch.py \
  "http://192.168.50.127:4747/video" \
  "best.pt" \
  0.5
```

### Optimization Techniques / 性能优化技术

| Technique / 技术 | Location / 位置 | Effect / 效果 |
|---|---|---|
| **4-thread pipeline** | Global | Overlap compute & I/O, maximize GPU utilization |
| **CUDA GPU preprocess** | Thread 2 | Avoid D2H→H2D roundtrip, fuse 5 steps into 1 kernel |
| **Dedicated CUDA Stream** | Thread 3 | Avoid TRT default stream global sync |
| **INT8 quantization** | Model | Halve model size, 40-60% inference speedup |
| **FramePool** | Global | One-time allocation, zero runtime GPU malloc |
| **Bounded queue + frame drop** | FrameQueue | Auto-drop stale frames, real-time guarantee |
| **Non-blocking display** | Thread 4 | `tryPop` without waiting, sleep 1ms when idle |

---

## 📁 Project Structure / 项目结构

```
PedestrianTracker/
├── CMakeLists.txt                  # Build config / 构建配置
├── README.md                       # This file / 本文件 (bilingual)
├── LICENSE                         # MIT License
├── baseline_pytorch.py             # Python single-thread baseline / 单线程基准
├── show/
│   ├── demo-screenshot.jpg         # Screenshot / 运行截图
│   └── demo-video.mp4              # Demo video / 演示视频
├── src/
│   ├── main.cpp                    # Entry point / 程序入口
│   ├── trt_engine.cpp              # TensorRT engine wrapper / TRT 引擎封装
│   ├── worker_threads.cpp          # Worker threads + YOLO postprocess / 工作线程+后处理
│   └── cuda_preprocess.cu          # CUDA preprocess kernel / CUDA 预处理 kernel
├── include/
│   ├── trt_engine.hpp              # TRT engine header
│   ├── worker_threads.hpp          # Frame/Detection structs, thread functions
│   ├── frame_queue.hpp             # Thread-safe bounded queue / 线程安全有界队列
│   └── cuda_preprocess.h           # CUDA preprocess API + buffer struct
└── .vscode/
    ├── settings.json               # CMake config
    ├── launch.json                 # GDB debug config / GDB 调试配置
    └── tasks.json                  # Build task (make) / 构建任务
```

~1,200 lines of code, clean modular design / 约 1200 行代码，模块清晰

---

## 🧩 Core Modules / 核心模块

### `FrameQueue<T>` — Bounded Thread-safe Queue / 有界线程安全队列

- Generic template, any type / 模板化设计，可存放任意类型
- Auto-drops oldest frame when full (non-blocking push for real-time) / 容量满时自动丢弃最旧帧
- Dual condition variables: consumer wait + producer wait / 双条件变量
- `stop()` graceful shutdown, `droppedCount()` statistics / `stop()` 优雅关闭，丢帧统计

### `TensorRTEngine` — TRT 10.x Inference Engine / 推理引擎

- RAII resource management, move semantics / RAII 资源管理，移动语义
- Dynamic tensor binding via `setTensorAddress` / 遍历 I/O tensor 动态绑定
- Dedicated CUDA stream to avoid global sync / 独立 CUDA stream 避免全局同步
- Full logging: model size, tensor name/dimensions/size / 完整日志

### `cuda_preprocess` — GPU Preprocessing / GPU 预处理

- Bilinear interpolation sampling (BGR HWC → RGB CHW) / 双线性插值采样
- Letterbox resize + gray border fill (114, 114, 114) / letterbox 缩放 + 灰边填充
- Normalize to [0, 1] / 归一化到 [0, 1]
- `16×16` thread blocks, `40×40` grid (640×640 target) / `16×16` 线程块，`40×40` 网格

### `worker_threads` — Pipeline Threads / 流水线工作线程

- **FramePool**: N pre-allocated frames, cyclic reuse / 预分配 N 帧，循环复用
- **parseYoloOutput**: CHW layout parsing, center→corner conversion, OpenCV NMS
- **FPS**: Inference FPS printed every 2s, end-to-end FPS overlaid on display every 1s

---

## 🎯 Design Decisions / 设计决策

| Decision / 决策 | Reason / 原因 |
|---|---|
| **GPU Preprocess / GPU 预处理** | Avoid D2H→H2D roundtrip; kernel parallelism 409,600 pixels |
| **INT8 Quantization / INT8 量化** | Jetson Orin Ampere GPU has dedicated INT8 Tensor Core acceleration |
| **Conf threshold 0.45 / 置信度阈值 0.45** | INT8 causes naturally lower conf; lower threshold to prevent missed detections |
| **90° Rotation / 90° 旋转** | Phone outputs landscape by default; rotate for portrait view |
| **FramePool = 4** | At most 4 frames in-flight (capture, preprocess, infer, display) |
| **QueueSize = 2** | Small capacity backpressure to prevent memory buildup |

---

## 🔍 Data Flow / 数据流

```
Raw frame (CPU) → [T1: Capture + Rotate] → q1
  → [T2: H2D → GPU letterbox → CHW float32] → q2
    → [T3: D2D copy → enqueueV3 → D2D copy] → q3
      → [T4: D2H → Parse + NMS → Draw → imshow] → Return to FramePool
```

Every GPU/CPU buffer is pre-allocated in FramePool — zero dynamic allocation at runtime.<br>
所有 GPU/CPU buffer 均在 FramePool 中预分配 —— 运行时零动态分配。

---

## 📄 License / 许可证

MIT License · Copyright (c) 2025

---

## 🤝 Contributing / 贡献

Issues and Pull Requests welcome! / 欢迎提交 Issue 和 Pull Request！

---

> 💡 **Note / 提示**：Designed for Jetson Orin Nano, but works on any CUDA + TensorRT GPU (adjust `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt).<br>
> 本项目专为 Jetson Orin Nano 设计，但可在任何支持 CUDA + TensorRT 的 GPU 上运行（调整 CMakeLists.txt 中的 `CMAKE_CUDA_ARCHITECTURES`）。
