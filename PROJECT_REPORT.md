# 🚶 PedestrianTracker 项目完整分析报告

> 生成日期：2026-06-25

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [技术栈与依赖](#2-技术栈与依赖)
3. [系统架构](#3-系统架构)
4. [模块详解](#4-模块详解)
5. [数据流分析](#5-数据流分析)
6. [构建系统](#6-构建系统)
7. [VS Code 开发环境](#7-vs-code-开发环境)
8. [关键设计决策](#8-关键设计决策)
9. [性能特点](#9-性能特点)
10. [运行方式](#10-运行方式)

---

## 1. 项目概述

**PedestrianTracker** 是一个高性能的 **实时行人检测** C++ 流水线系统，专为 **NVIDIA Jetson Orin Nano** 边缘计算平台设计。

| 属性 | 值 |
|---|---|
| 项目名称 | PedestrianTracker |
| 语言 | C++17 + CUDA |
| 目标平台 | NVIDIA Jetson Orin Nano (aarch64) |
| GPU 架构 | sm_87 (Ampere) |
| 核心功能 | 实时视频流行人检测 |
| AI 模型 | YOLO11n (nano)，单类（person），INT8 量化 |
| 推理引擎 | TensorRT 10.x |
| 视频输入 | HTTP 视频流（如手机摄像头） |
| 模型路径 | `best_int8.engine`（已编译为 TensorRT engine） |

---

## 2. 技术栈与依赖

```
┌─────────────────────────────────────────────────┐
│                  应用层 (C++17)                    │
├─────────────────────────────────────────────────┤
│  OpenCV 4.x     │  TensorRT 10.x  │  CUDA 12.x  │
│  (视频I/O/显示)  │  (模型推理)      │  (GPU加速)   │
├─────────────────────────────────────────────────┤
│              NVIDIA Jetson Orin Nano              │
│              GPU: Ampere (sm_87)                  │
└─────────────────────────────────────────────────┘
```

| 库 | 用途 | 链接方式 |
|---|---|---|
| **OpenCV** | 视频流读取 (`cv::VideoCapture`)、图像显示、NMS（非极大值抑制） | 动态链接 |
| **TensorRT 10.x** (`libnvinfer`) | 加载 `.engine` 文件、创建执行上下文、GPU 推理 | 动态链接 |
| **TensorRT Plugin** (`libnvinfer_plugin`) | TensorRT 插件支持 | 动态链接 |
| **CUDA Runtime** (`libcudart`) | GPU 内存管理 (`cudaMalloc`/`cudaFree`/`cudaMemcpy`)、CUDA Stream | 动态链接 |
| **pthread** | 多线程支持 | 系统库 |

---

## 3. 系统架构

### 3.1 整体架构图

```
                        ┌───────────────────────────────────────────┐
                        │              FramePool (对象池)             │
                        │         预分配 4 个 Frame，循环复用           │
                        └──────┬──────────────────────┬─────────────┘
                               │ acquire()            │ release()
                               ▼                      ▲
┌──────────┐    q1     ┌──────────┐    q2    ┌──────────┐    q3    ┌──────────┐
│ Thread 1 │ ────────► │ Thread 2 │ ───────► │ Thread 3 │ ───────► │ Thread 4 │
│  视频抓取 │          │ CUDA预处理 │          │ TRT 推理  │          │ 后处理显示 │
│ Capture  │          │Preprocess │          │  Infer   │          │ Display  │
└──────────┘          └──────────┘          └──────────┘          └──────────┘
     │                     │                     │                     │
     │  HTTP 视频流         │  GPU: letterbox    │  GPU: enqueueV3     │  CPU: NMS + 画框
     │  cv::VideoCapture   │  + BGR→RGB         │  D2D memcpy         │  cv::imshow
     │  CPU 旋转 90°       │  + 归一化 + CHW     │  + 推理             │  + FPS 统计
     ▼                     ▼                     ▼                     ▼
┌──────────┐          ┌──────────┐          ┌──────────┐          ┌──────────┐
│ 手机摄像头 │          │  CUDA    │          │  TensorRT │          │  显示器    │
│ 192.168. │          │  Kernel  │          │  Engine  │          │  1024x600 │
│ 50.127   │          │  16×16   │          │  INT8    │          │ 全屏窗口  │
└──────────┘          └──────────┘          └──────────┘          └──────────┘
```

### 3.2 四线程流水线

每个线程独立运行，通过 `FrameQueue` 传递数据：

| 线程 | 函数 | 输入 | 输出 | 主要工作 |
|---|---|---|---|---|
| **T1. 视频抓取** | `captureThread` | HTTP 视频流 URL | → q1 | 打开视频流，逐帧读取，CPU 旋转 90°，借 Frame |
| **T2. CUDA 预处理** | `preprocessThread` | q1 → | → q2 | 上传帧到 GPU，CUDA kernel 做 letterbox + BGR→RGB + 归一化 |
| **T3. TensorRT 推理** | `inferThread` | q2 → | → q3 | D2D 拷贝输入，调用 `enqueueV3` 推理，拷贝输出 |
| **T4. 后处理显示** | `displayThread` | q3 → | → 显示器 | 解析输出 → NMS → 画框 → 显示 → 归还 Frame |

---

## 4. 模块详解

### 4.1 `main.cpp` — 程序入口

**文件位置**: `src/main.cpp` (76 行)

**职责**：
1. 注册 `SIGINT` 信号处理器（Ctrl+C 优雅退出）
2. 解析命令行参数（视频 URL、engine 路径、置信度阈值）
3. 加载 TensorRT engine
4. 创建 FramePool（4 个预分配帧）和 3 个 FrameQueue（容量各 2）
5. 启动 4 个工作线程
6. 等待 display 线程退出（按 'q' 或 Ctrl+C）
7. 清理资源，停止所有线程

**命令行参数**：
```
./PedestrianTracker [视频URL] [engine路径] [置信度阈值]
./PedestrianTracker http://192.168.50.127:4747/video /path/to/model.engine 0.45
```

**关键参数**：
| 参数 | 默认值 | 含义 |
|---|---|---|
| `video_url` | `http://192.168.50.127:4747/video` | 手机摄像头 HTTP 流 |
| `engine_path` | `/home/zhouco/human_test/yolo11n_human/best_int8.engine` | TensorRT 模型 |
| `conf_thr` | `0.45` | 检测置信度阈值 |
| `kPoolSize` | `4` | Frame 池大小 |
| `kQueueSize` | `2` | 每级队列容量 |
| `kMaxW/H` | `1920×1080` | 最大帧尺寸 |

---

### 4.2 `frame_queue.hpp` — 线程安全队列

**文件位置**: `include/frame_queue.hpp` (88 行)

**设计特点**：
- **泛型模板** `FrameQueue<T>`，可存放任意类型
- **有界队列**：超过 `max_size_` 时自动丢弃最旧帧（非阻塞 push，保证实时性）
- **双条件变量**：`cond_`（消费者等待非空）、`cond_full_`（生产者等待非满）
- **优雅停止**：`stop()` 唤醒所有等待者，`pop()` 返回 `false` 表示结束
- **丢帧统计**：`dropped_` 计数器追踪丢弃帧数

**关键 API**：
```cpp
void push(const T& item);      // 阻塞推入（满则丢弃最旧）
void push(T&& item);           // 移动语义推入
bool pop(T& item);             // 阻塞取出（队列空且未 stop → 等待）
bool tryPop(T& item);          // 非阻塞取出（空则立即返回 false）
void stop();                   // 标记停止，唤醒所有等待者
```

---

### 4.3 `trt_engine.hpp/cpp` — TensorRT 引擎封装

**文件位置**: `include/trt_engine.hpp` (72 行) + `src/trt_engine.cpp` (219 行)

**核心类**: `TensorRTEngine`

**设计特点**：
- **RAII 资源管理**：析构函数自动释放所有 GPU buffer
- **禁止拷贝，允许移动**：使用移动语义安全转移所有权
- **TRT 10.x API**：使用 `enqueueV3`（而非旧版 `enqueueV2`）
- **动态 I/O 绑定**：遍历所有 tensor，按名称绑定 buffer（`setTensorAddress`）
- **完整日志**：模型大小、每个 I/O tensor 的名称/维度/大小

**加载流程**：
```
1. 读取 .engine 文件到内存 (二进制)
2. createInferRuntime() 创建 IRuntime
3. deserializeCudaEngine() 反序列化
4. createExecutionContext() 创建执行上下文
5. 遍历所有 I/O tensor → cudaMalloc + setTensorAddress
```

**模型输出信息**：
| 属性 | 值 |
|---|---|
| 模型 | YOLO11n (nano)，单类检测 |
| 量化 | INT8 |
| 输出 Shape | `(1, 5, 8400)` |
| 输出含义 | cx, cy, w, h, conf × 8400 个 anchor |
| 输出大小 | 8400 × 5 × 4B = 168 KB |

---

### 4.4 `cuda_preprocess.cu/h` — GPU 预处理

**文件位置**: `src/cuda_preprocess.cu` (163 行) + `include/cuda_preprocess.h` (45 行)

**核心功能**：在 GPU 上一次完成全部预处理，避免 CPU-GPU 往返。

**预处理流程**：
```
原始帧 (CPU BGR uint8, HWC)
    │
    ▼ cudaMemcpyAsync (HostToDevice)
GPU 内存 (BGR uint8, HWC)
    │
    ▼ preprocess_kernel (CUDA kernel, 16×16 线程块)
    │  ├─ 计算 letterbox 缩放 + 居中参数
    │  ├─ 双线性插值采样（BGR → RGB，含 half-pixel 对齐）
    │  ├─ 归一化到 [0, 1]
    │  ├─ 灰边填充 (114, 114, 114) / 255
    │  └─ HWC → CHW 布局转换
    │
    ▼
输出 (GPU float32, CHW, 3×640×640)
```

**CUDA Kernel 参数**：
| 参数 | 值 |
|---|---|
| 线程块大小 | 16 × 16 = 256 线程 |
| 网格大小 | ⌈640/16⌉ × ⌈640/16⌉ = 40 × 40 |
| 总线程数 | 40 × 40 × 256 = 409,600 |
| 目标尺寸 | 640 × 640 × 3 |

**双线性插值** (`bilinear_sample`)：
- 从 GPU BGR 图像中按浮点坐标采样单个通道值
- 取 4 个邻居像素 (x0,y0), (x1,y0), (x0,y1), (x1,y1)
- 水平插值 → 垂直插值 → 归一化 (/255.0)
- 边界 clamp 防止越界

---

### 4.5 `worker_threads.cpp/hpp` — 工作线程 + 后处理

**文件位置**: `src/worker_threads.cpp` (377 行) + `include/worker_threads.hpp` (111 行)

#### 4.5.1 FramePool — 对象池

预分配 N 个 `Frame` 对象，流水线循环使用，避免频繁 `cudaMalloc`/`cudaFree`：

```cpp
FramePool pool(4, 1920, 1080, output_bytes);
// 预分配 4 个 Frame
// 每个 Frame 包含：
//   - GPU 预处理 buffer (1920×1080×3 uint8 + 3×640×640 float)
//   - CPU 图像 Mat (1080×1920×3 uint8)
//   - GPU 推理输出 buffer (~168KB float)
```

#### 4.5.2 YOLO 输出解析 + NMS

**`parseYoloOutput()` 函数**（约 75 行）：

```
输入: float output[1][5][8400] — GPU 数据已拷到 CPU
输出: std::vector<Detection> — 最终检测框

处理步骤:
  1. 按 CHW 布局解析 (非交错存储):
     channel 0 = cx (8400 个值)
     channel 1 = cy
     channel 2 = w
     channel 3 = h
     channel 4 = confidence

  2. 将中心点格式转为角点格式: (cx,cy,w,h) → (x1,y1,x2,y2)

  3. 缩放到原始帧尺寸: coord × (src_w/640, src_h/640)

  4. 裁剪到图像范围内

  5. 按置信度阈值过滤

  6. OpenCV NMS (nms_threshold=0.45)
```

#### 4.5.3 Thread 4 显示逻辑

```
1. 从 q3 获取推理完成的帧 (非阻塞 tryPop)
2. 从 GPU 拷贝输出到 CPU
3. 解析 YOLO 输出 + NMS
4. 画绿色检测框 + "person X.XX" 标签
5. 计算并显示端到端 FPS
6. 保持宽高比缩放 + 黑边填充 → 1024×600 固定窗口
7. cv::imshow 显示
8. 归还 Frame 到池中
9. 检测按键 'q' 退出
```

---

## 5. 数据流分析

### 5.1 Frame 结构体（流水线中流转的数据单元）

```
┌─────────────────────────────────────────────┐
│                  Frame                       │
├─────────────────────────────────────────────┤
│  id: int                   帧编号            │
│  cpu_img: cv::Mat          CPU 图像 (BGR)    │
│  src_w / src_h: int        原始帧宽高         │
│  pp_buf: PreprocessBuffers GPU 预处理 buffer  │
│    ├─ d_src: uint8*        源帧 (HWC)        │
│    └─ d_dst: float*        预处理后 (CHW)     │
│  d_output: float*          GPU 推理输出       │
│  d_output_bytes: size_t    输出字节数         │
│  detections: vector        CPU 检测结果       │
└─────────────────────────────────────────────┘
```

### 5.2 逐级数据变换

```
[Thread 1 - Capture]
  HTTP 视频流 → cv::Mat (BGR, HWC, uint8, 变化尺寸)
  → rotate 90° clockwise
  → push(FramePtr) → q1

[Thread 2 - Preprocess]
  q1 → pop(FramePtr)
  → cudaMemcpy H2D: cpu_img → d_src
  → CUDA kernel: d_src (BGR HWC) → d_dst (RGB CHW, 3×640×640, float32 [0,1])
  → push(FramePtr) → q2

[Thread 3 - Infer]
  q2 → pop(FramePtr)
  → cudaMemcpy D2D: pp_buf.d_dst → engine input tensor
  → engine.enqueueV3(stream): 推理
  → cudaMemcpy D2D: engine output → frame.d_output
  → push(FramePtr) → q3

[Thread 4 - Display]
  q3 → tryPop(FramePtr)
  → cudaMemcpy D2H: frame.d_output → h_out (1×5×8400 float)
  → parseYoloOutput() + NMS → detections
  → cv::rectangle + cv::putText 画框
  → 缩放 + 填充 → 1024×600
  → cv::imshow
  → pool.release(FramePtr)
```

### 5.3 内存管理

```
预分配 (启动时):
  4 × Frame = 4 × (
    1920×1080×3 B    GPU 源帧 buffer  (~6 MB)
    3×640×640×4 B    GPU 输出 buffer   (~4.9 MB)
    1×5×8400×4 B     GPU 推理输出       (~168 KB)
    1920×1080×3 B    CPU 图像 Mat       (~6 MB)
  )
  总计 ≈ 4 × ~17 MB ≈ 68 MB GPU + 24 MB CPU

运行时: 零分配 (全部复用预分配 buffer)
```

---

## 6. 构建系统

### 6.1 CMakeLists.txt 解析

```
cmake_minimum_required(VERSION 3.18)
project(PedestrianTracker LANGUAGES CXX CUDA)

C++ 标准: C++17
CUDA 标准: C++17
GPU 架构: sm_87 (Jetson Orin Nano)

依赖查找:
  - OpenCV (find_package)
  - CUDA Toolkit (find_package)
  - TensorRT (find_path/find_library, 指定 aarch64 路径)

源文件:
  C++: main.cpp, trt_engine.cpp, worker_threads.cpp
  CUDA: cuda_preprocess.cu

链接库:
  OpenCV + TensorRT(nvinfer, nvinfer_plugin) + CUDA Runtime + pthread

优化:
  Release: -O3 -DNDEBUG
  CUDA: --use_fast_math
```

### 6.2 构建命令

```bash
# 配置
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build build -j$(nproc)

# 生成的可执行文件
./build/PedestrianTracker
# 或
./build/Benchmark
```

### 6.3 已编译产物

从 build 目录可见项目已经成功编译：
- `build/PedestrianTracker` — 主程序可执行文件
- `build/Benchmark` — 性能基准测试可执行文件
- 目标文件已生成（`.o` 文件）

---

## 7. VS Code 开发环境

### 7.1 配置文件

| 文件 | 用途 |
|---|---|
| `.vscode/settings.json` | CMake 配置路径、调试器设置 |
| `.vscode/launch.json` | GDB 调试启动配置 |
| `.vscode/tasks.json` | `make -C build -j$(nproc)` 构建任务 |

### 7.2 调试配置

```
调试器: GDB (gdb-multiarch for ARM64 cross-debug)
程序路径: ${workspaceFolder}/build/PedestrianTracker
前置任务: CMake Build (make -j$(nproc))
断点: 入口处不停止 (stopAtEntry: false)
```

---

## 8. 关键设计决策

### 8.1 为什么使用四线程流水线？

| 优势 | 说明 |
|---|---|
| **计算与 I/O 重叠** | 抓取一帧的同时，前一帧在做预处理，再前一帧在做推理 |
| **GPU 利用率最大化** | CUDA 预处理和 TRT 推理可以在不同 stream 上并发 |
| **低延迟** | 流水线填满后，每帧延迟 ≈ max(各阶段耗时)，而非 sum(各阶段耗时) |
| **背压控制** | FrameQueue 容量为 2，自动丢弃旧帧，防止内存爆炸 |

### 8.2 为什么在 GPU 上做预处理？

- **避免 D2H→H2D 往返**：如果 CPU 预处理，需要 GPU→CPU→GPU，浪费带宽
- **CUDA kernel 并行度高**：640×640 = 409,600 像素，40×40 个线程块并行处理
- **letterbox + resize + BGR→RGB + normalize + CHW 一次完成**：5 步合并为 1 个 kernel

### 8.3 为什么使用 INT8 量化？

- YOLO11n INT8 模型比 FP16 体积减半，推理速度提升约 40-60%
- Jetson Orin Nano 的 Ampere GPU 对 INT8 有专门的 Tensor Core 加速
- 代价是置信度略微降低（代码注释中已将阈值从 0.5 降至 0.45）

### 8.4 为什么使用 FramePool 对象池？

- `cudaMalloc` 是昂贵的同步操作（会隐式同步整个 GPU）
- 启动时一次性分配，运行时零 GPU 内存分配
- 循环使用 4 个 Frame，流水线中同时最多 4 帧在途

### 8.5 为什么选择 90° 旋转？

```cpp
cv::rotate(frame->cpu_img, frame->cpu_img, cv::ROTATE_90_CLOCKWISE);
```
手机摄像头通常以横屏方向输出，旋转 90° 后获得正确的竖直画面。

---

## 9. 性能特点

### 9.1 各阶段估算（基于 YOLO11n INT8, 640×640, Jetson Orin Nano）

| 阶段 | 线程 | 预计耗时 | 瓶颈 |
|---|---|---|---|
| 视频抓取 | T1 | ~5-10 ms | 网络延迟 |
| GPU 预处理 | T2 | ~1-2 ms | GPU 带宽 |
| TRT 推理 | T3 | ~8-15 ms | GPU 计算 |
| 后处理+显示 | T4 | ~3-8 ms | CPU (NMS) |
| **端到端** | — | **~15-30 ms** | **约 30-60 FPS** |

### 9.2 性能优化技术

| 技术 | 位置 | 效果 |
|---|---|---|
| CUDA Stream 异步 | inferThread | 避免 TRT 用 default stream 触发全局同步 |
| D2D 拷贝 | inferThread | 预处理输出→引擎输入，引擎输出→Frame buffer，零 CPU 参与 |
| 非阻塞 tryPop | displayThread | 显示线程不等待推理完成，无帧时 sleep(1ms) |
| 丢帧策略 | FrameQueue push | 队列满时自动丢旧帧，保证实时性 |
| fast math | CUDA 编译选项 | `--use_fast_math` 加速浮点运算 |

### 9.3 FPS 统计

代码中有两处 FPS 统计：
- **T3 inferThread**：每 2 秒打印推理 FPS（纯推理速率）
- **T4 displayThread**：每 1 秒更新显示 FPS（端到端流水线速率，画在图像上）

---

## 10. 运行方式

### 10.1 基本运行

```bash
# 在 Jetson Orin Nano 上
./build/PedestrianTracker

# 自定义参数
./build/PedestrianTracker \
  "http://192.168.1.100:4747/video" \
  "/home/user/models/best_int8.engine" \
  0.4
```

### 10.2 退出方式

- 按键盘 **'q'** 键
- 按 **Ctrl+C** (SIGINT)

### 10.3 前置条件

1. Jetson Orin Nano 已安装 JetPack（含 CUDA + TensorRT + OpenCV）
2. 手机运行 IP 摄像头 App（如 IP Webcam），提供 HTTP 视频流
3. `.engine` 模型文件已就位

---

## 附：文件清单

```
PedestrainTracker/
├── CMakeLists.txt                  # CMake 构建配置
├── build/
│   ├── PedestrianTracker           # 编译好的主程序
│   ├── Benchmark                   # 编译好的性能测试
│   └── ...                         # CMake 生成文件
├── src/
│   ├── main.cpp                    # 程序入口 (76 行)
│   ├── trt_engine.cpp              # TensorRT 引擎封装 (219 行)
│   ├── worker_threads.cpp          # 工作线程 + YOLO 后处理 (377 行)
│   └── cuda_preprocess.cu          # CUDA 预处理 kernel (163 行)
├── include/
│   ├── trt_engine.hpp              # TRT 引擎头文件 (72 行)
│   ├── worker_threads.hpp          # 工作线程头文件 (111 行)
│   ├── frame_queue.hpp             # 线程安全队列 (88 行)
│   └── cuda_preprocess.h           # CUDA 预处理头文件 (45 行)
└── .vscode/
    ├── settings.json               # VS Code 设置
    ├── launch.json                 # 调试启动配置
    └── tasks.json                  # 构建任务
```

**总代码量**：约 1150 行（不含注释），结构清晰，模块分明。

---

> 📌 **总结**：这是一个设计精良的实时行人检测系统，充分利用了 NVIDIA Jetson Orin Nano 的 GPU 加速能力。通过四线程流水线 + GPU 预处理 + INT8 量化 + 对象池复用，在边缘设备上实现了高效、低延迟的行人检测。
