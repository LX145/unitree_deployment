cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# ============================================================
# 自动识别系统架构 (Auto-detect Architecture)
# ============================================================
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    # Jetson Orin (Arm64)
    message(STATUS "Target Architecture: AArch64 (Jetson/Arm)")
    set(ONNX_DIR_NAME "onnxruntime-linux-aarch64-1.22.0")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    # PC (Intel/AMD)
    message(STATUS "Target Architecture: x86_64 (PC)")
    set(ONNX_DIR_NAME "onnxruntime-linux-x64-1.22.0")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 设置 ONNX 的根路径方便后续引用
set(ONNX_ROOT_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/${ONNX_DIR_NAME}")

# ============================================================
# 头文件路径
# ============================================================
include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${ONNX_ROOT_DIR}/include  # <--- 使用自动选择的路径
  ${PROJECT_SOURCE_DIR}/../../include/
)

# ============================================================
# 链接库
# ============================================================
link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNX_ROOT_DIR}/lib/libonnxruntime.so.1.22.0 # <--- 使用自动选择的路径
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)