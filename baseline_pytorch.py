#!/usr/bin/env python3
# =============================================================================
# baseline_pytorch.py — 单线程 PyTorch 基准对比
# =============================================================================
# 本脚本作为 PedestrianTracker C++ 流水线的性能对比基准：
#
#   C++ 流水线                    本脚本（Python 基准）
#   ────────────────────────      ──────────────────────────
#   4 线程并行                    单线程串行
#   TensorRT INT8 engine          PyTorch FP16/FP32 PT
#   GPU 预处理 (CUDA kernel)      CPU 预处理 (ultralytics 内置)
#   FramePool 零动态分配          每帧动态分配内存
#   端到端 ~30-60 FPS             端到端 ~10-20 FPS
#
# 用法：
#   python baseline_pytorch.py [视频URL] [模型路径] [置信度阈值]
#   python baseline_pytorch.py
#   python baseline_pytorch.py http://192.168.1.100:4747/video best.pt 0.5
# =============================================================================

import sys
import time
import cv2
from ultralytics import YOLO

# =============================================================================
# 命令行参数
# =============================================================================
video_url   = sys.argv[1] if len(sys.argv) >= 2 else "http://192.168.50.127:4747/video"
model_path  = sys.argv[2] if len(sys.argv) >= 3 else "best.pt"
conf_thr    = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.5

print("==============================================")
print("  Pedestrian Tracker — Baseline (PyTorch)")
print(f"  Video : {video_url}")
print(f"  Model : {model_path}")
print(f"  Conf  : {conf_thr}")
print("==============================================")

# =============================================================================
# 打开视频流
# =============================================================================
cap = cv2.VideoCapture(video_url)
if not cap.isOpened():
    print("[Error] Cannot connect to video stream")
    sys.exit(1)

# =============================================================================
# 加载 YOLO 模型（PyTorch CUDA）
# =============================================================================
model = YOLO(model_path)
model.to("cuda")

# =============================================================================
# 显示窗口
# =============================================================================
cv2.namedWindow("Baseline - PyTorch CUDA (Single Thread)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Baseline - PyTorch CUDA (Single Thread)", 480, 640)

# =============================================================================
# FPS 统计变量
# =============================================================================
fps_avg         = 0.0
fps_frame_count = 0
fps_start       = time.time()
prev_time       = fps_start

# =============================================================================
# 主循环（串行：抓取 → 推理 → 显示）
# =============================================================================
print("\n[Baseline] Running. Press 'q' to exit.\n")

while True:
    # ---- 抓取一帧 ----
    ret, frame = cap.read()
    if not ret:
        print("[Error] Video stream interrupted")
        break

    # 修正手机画面方向：顺时针旋转 90°
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # ---- YOLO 推理（GPU 加速）----
    # 注意：ultralytics 内部会在 CPU 上做预处理（resize/normalize），
    # 与 C++ 版本的 GPU 预处理形成对比
    t0 = time.time()
    results = model(frame, conf=conf_thr)
    infer_ms = (time.time() - t0) * 1000

    # ---- 画检测框 ----
    frame = results[0].plot()

    # ---- FPS 计算 ----
    fps_frame_count += 1
    now = time.time()
    elapsed = now - fps_start
    instant_fps = 1.0 / (now - prev_time) if now != prev_time else 0
    prev_time = now

    # 每秒更新一次平均 FPS
    if elapsed >= 1.0:
        fps_avg = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_start = now

    # ---- 叠加 FPS 信息 ----
    cv2.putText(frame, f"FPS: {instant_fps:.1f} (avg: {fps_avg:.1f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Infer: {infer_ms:.0f}ms",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Baseline - PyTorch CUDA (Single Thread)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================================================================
# 清理
# =============================================================================
cap.release()
cv2.destroyAllWindows()
print("[Baseline] Stopped.")
