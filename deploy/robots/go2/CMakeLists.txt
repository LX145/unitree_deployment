cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# ---- RealSense2 (optional, via pkg-config) ----
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(realsense2 QUIET realsense2)
endif()
if(realsense2_FOUND)
    message(STATUS "RealSense2 found: depth camera support enabled")
    set(HAS_REALSENSE ON)
    add_compile_definitions(HAS_REALSENSE)
    include_directories(${realsense2_INCLUDE_DIRS})
else()
    message(STATUS "RealSense2 NOT found: depth camera support disabled")
    set(HAS_REALSENSE OFF)
endif()

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
set(COMMON_LIBS
  unitree_sdk2 ddsc ddscxx rt pthread
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNX_ROOT_DIR}/lib/libonnxruntime.so.1.22.0
)

link_libraries(${COMMON_LIBS})

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

# shared sensor sources (only when RealSense is available)
if(HAS_REALSENSE)
    file(GLOB_RECURSE SENSOR_SRC_LIST
      ${PROJECT_SOURCE_DIR}/../../src/sensors/*.cpp
    )
    link_libraries(${realsense2_LIBRARIES})
endif()

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST} ${SENSOR_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)

# ---- OpenCV (optional, for depth camera test display) ----
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: depth viewer enabled")
    add_executable(test_depth_dds test_depth_dds.cpp)
    target_link_libraries(test_depth_dds ${COMMON_LIBS} ${OpenCV_LIBS})
    target_include_directories(test_depth_dds PRIVATE ${OpenCV_INCLUDE_DIRS})
else()
    message(STATUS "OpenCV NOT found: DDS depth verifier disabled")
endif()

# standalone depth camera test (only when RealSense is available)
if(HAS_REALSENSE)
    add_executable(test_depth_camera test_depth_camera.cpp ${SENSOR_SRC_LIST})
    target_link_libraries(test_depth_camera ${COMMON_LIBS} ${realsense2_LIBRARIES})
    if(OpenCV_FOUND)
        target_link_libraries(test_depth_camera ${OpenCV_LIBS})
        target_include_directories(test_depth_camera PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(test_depth_camera PRIVATE HAS_OPENCV)
    endif()

    # Read-only shadow inference: LowState DDS + RealSense + depth policy.
    # This target intentionally has no LowCmd publisher/control path.
    add_executable(test_depth_action test_depth_action.cpp)
    target_link_libraries(test_depth_action ${PROJECT_NAME}_lib ${COMMON_LIBS} ${realsense2_LIBRARIES})
    target_compile_definitions(test_depth_action PRIVATE GO2_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()
