cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PedestrianTracker LANGUAGES CXX CUDA)# 声明同时使用C++和CUDA语言

# 1.设置C++和CUDA的编译标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2.设置 Jetson Orin Nano 的 GPU 架构（Orin 架构为 sm_87）
set(CMAKE_CUDA_ARCHITECTURES 87)

# 允许 CUDA 的 __host__ __device__ 函数跨 .cu 文件调用（如果不需要可删除）
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 3.寻找依赖库
# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "  version: ${OpenCV_VERSION}")
message(STATUS "  libraries: ${OpenCV_LIBS}")
message(STATUS "  include path: ${OpenCV_INCLUDE_DIRS}")

# Find CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA toolkit status:")
message(STATUS "  version: ${CUDAToolkit_VERSION}")
message(STATUS "  include: ${CUDAToolkit_INCLUDE_DIRS}")

# Find TensorRT (Jetson 上通常在系统默认路径)
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu
    NO_DEFAULT_PATH
)
find_library(TENSORRT_NVINFER
    NAMES nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH
)
find_library(TENSORRT_NVINFER_PLUGIN
    NAMES nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH
)
message(STATUS "TensorRT status:")
message(STATUS "  include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  nvinfer: ${TENSORRT_NVINFER}")
message(STATUS "  nvinfer_plugin: ${TENSORRT_NVINFER_PLUGIN}")

# 4.指定头文件搜索路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 5.收集源文件
set(SOURCES
    src/main.cpp
    src/trt_engine.cpp
    src/worker_threads.cpp
)

set(CUDA_SOURCES
    src/cuda_preprocess.cu
)

# 6.生成可执行程序
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${CUDA_SOURCES}
)

# 7.链接依赖库
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        # OpenCV
        ${OpenCV_LIBS}
        # TensorRT
        ${TENSORRT_NVINFER}
        ${TENSORRT_NVINFER_PLUGIN}
        # CUDA
        CUDA::cudart
        # 系统库
        pthread
)

# 8.编译优化选项
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# 设置默认 build 类型为 Release（没有 -DCMAKE_BUILD_TYPE 时生效）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()
