# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)

project(sampleYOLOV8)

add_compile_options(-std=c++11)
find_package(OpenMP)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
set(COMMON_INC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../common/include")

find_path(ACL_INC_PATH acl/acl.h
    PATHS
        "${TOOLKIT_ROOT}/aarch64-linux/include"
        "${TOOLKIT_ROOT}/acllib/include"
        "${TOOLKIT_ROOT}/runtime/include"
    NO_DEFAULT_PATH
)

find_library(ASCENDCL_LIBRARY ascendcl
    PATHS
        "${TOOLKIT_ROOT}/aarch64-linux/lib64"
        "${TOOLKIT_ROOT}/aarch64-linux/devlib"
        "${TOOLKIT_ROOT}/acllib/lib64"
        "${TOOLKIT_ROOT}/runtime/lib64"
    NO_DEFAULT_PATH
)

find_library(ACL_DVPP_LIBRARY acl_dvpp
    PATHS
        "${TOOLKIT_ROOT}/aarch64-linux/lib64"
        "${TOOLKIT_ROOT}/aarch64-linux/devlib"
        "${TOOLKIT_ROOT}/acllib/lib64"
        "${TOOLKIT_ROOT}/runtime/lib64"
    NO_DEFAULT_PATH
)

find_library(ACLLITE_LIBRARY acllite
    PATHS
        "${TOOLKIT_ROOT}/aarch64-linux/thirdpart/lib"
        "${TOOLKIT_ROOT}/thirdpart/lib"
    NO_DEFAULT_PATH
)

find_path(ACLLITE_INCLUDE_ROOT acllite/AclLiteUtils.h
    PATHS
        "${TOOLKIT_ROOT}/aarch64-linux/thirdpart/include"
        "${TOOLKIT_ROOT}/thirdpart/include"
    NO_DEFAULT_PATH
)

find_path(OPENCV_INC_PATH opencv2/opencv.hpp
    PATHS
        "/usr/local/include"
        "/usr/include/opencv4"
        "/usr/include"
)

find_library(OPENCV_CORE_LIBRARY opencv_core
    PATHS
        "/usr/local/lib64"
        "/usr/local/lib"
        "/usr/lib64"
        "/usr/lib/aarch64-linux-gnu"
        "/usr/lib"
)

find_library(OPENCV_IMGPROC_LIBRARY opencv_imgproc
    PATHS
        "/usr/local/lib64"
        "/usr/local/lib"
        "/usr/lib64"
        "/usr/lib/aarch64-linux-gnu"
        "/usr/lib"
)

find_library(OPENCV_IMGCODECS_LIBRARY opencv_imgcodecs
    PATHS
        "/usr/local/lib64"
        "/usr/local/lib"
        "/usr/lib64"
        "/usr/lib/aarch64-linux-gnu"
        "/usr/lib"
)

find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h
    PATHS
        "/usr/local/include"
        "/usr/include"
)

find_library(YAML_CPP_LIBRARY yaml-cpp
    PATHS
        "/usr/local/lib64"
        "/usr/local/lib"
        "/usr/lib64"
        "/usr/lib/aarch64-linux-gnu"
        "/usr/lib"
)

if(NOT ACL_INC_PATH)
    message(FATAL_ERROR "ACL include path not found under ${TOOLKIT_ROOT}")
endif()

if(NOT ASCENDCL_LIBRARY OR NOT ACL_DVPP_LIBRARY)
    message(FATAL_ERROR "Ascend runtime libraries not found under ${TOOLKIT_ROOT}")
endif()

if(NOT ACLLITE_LIBRARY)
    message(FATAL_ERROR "libacllite.so not found under ${TOOLKIT_ROOT}")
endif()

if(NOT OPENCV_INC_PATH OR NOT OPENCV_CORE_LIBRARY OR NOT OPENCV_IMGPROC_LIBRARY OR NOT OPENCV_IMGCODECS_LIBRARY)
    message(FATAL_ERROR "OpenCV headers or libraries were not found")
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. postprocess will run in serial mode.")
endif()

get_filename_component(ACL_LIB_PATH "${ASCENDCL_LIBRARY}" DIRECTORY)
get_filename_component(THIRDPART_LIB_PATH "${ACLLITE_LIBRARY}" DIRECTORY)
get_filename_component(OPENCV_LIB_PATH "${OPENCV_CORE_LIBRARY}" DIRECTORY)
set(COMMON_RPATH "${ACL_LIB_PATH};${THIRDPART_LIB_PATH};${OPENCV_LIB_PATH}")

find_library(AVCODEC_LIBRARY avcodec PATHS "${THIRDPART_LIB_PATH}" NO_DEFAULT_PATH)
find_library(AVFORMAT_LIBRARY avformat PATHS "${THIRDPART_LIB_PATH}" NO_DEFAULT_PATH)
find_library(AVDEVICE_LIBRARY avdevice PATHS "${THIRDPART_LIB_PATH}" NO_DEFAULT_PATH)
find_library(AVUTIL_LIBRARY avutil PATHS "${THIRDPART_LIB_PATH}" NO_DEFAULT_PATH)
find_library(SWRESAMPLE_LIBRARY swresample PATHS "${THIRDPART_LIB_PATH}" NO_DEFAULT_PATH)
find_library(SWSCALE_LIBRARY swscale PATHS "${THIRDPART_LIB_PATH}" NO_DEFAULT_PATH)
find_library(AVFILTER_LIBRARY avfilter PATHS "${THIRDPART_LIB_PATH}" NO_DEFAULT_PATH)

if(NOT ACLLITE_INCLUDE_ROOT)
    message(FATAL_ERROR "AclLite headers not found under ${TOOLKIT_ROOT}")
endif()

if(NOT AVCODEC_LIBRARY OR NOT AVFORMAT_LIBRARY OR NOT AVDEVICE_LIBRARY OR NOT AVUTIL_LIBRARY OR
   NOT SWRESAMPLE_LIBRARY OR NOT SWSCALE_LIBRARY OR NOT AVFILTER_LIBRARY)
    message(FATAL_ERROR "FFmpeg libraries required by libacllite.so were not found under ${THIRDPART_LIB_PATH}")
endif()

message(STATUS "ACL_INC_PATH: ${ACL_INC_PATH}")
message(STATUS "ACL_LIB_PATH: ${ACL_LIB_PATH}")
message(STATUS "THIRDPART_LIB_PATH: ${THIRDPART_LIB_PATH}")
message(STATUS "ACLLITE_INCLUDE_ROOT: ${ACLLITE_INCLUDE_ROOT}")
message(STATUS "COMMON_INC_PATH: ${COMMON_INC_PATH}")
message(STATUS "OPENCV_INC_PATH: ${OPENCV_INC_PATH}")
message(STATUS "OPENCV_LIB_PATH: ${OPENCV_LIB_PATH}")
message(STATUS "COMMON_RPATH: ${COMMON_RPATH}")

if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
    message(STATUS "YAML_CPP_INCLUDE_DIR: ${YAML_CPP_INCLUDE_DIR}")
    message(STATUS "YAML_CPP_LIBRARY: ${YAML_CPP_LIBRARY}")
else()
    message(WARNING "yaml-cpp not found. sampleYOLOV8Cls/sampleYOLOV8Obb/sampleYOLOV8Pose will be skipped.")
endif()

set(CMAKE_BUILD_RPATH "${COMMON_RPATH}")
set(CMAKE_INSTALL_RPATH "${COMMON_RPATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(
    ${ACL_INC_PATH}
    ${ACLLITE_INCLUDE_ROOT}
    ${ACLLITE_INCLUDE_ROOT}/acllite
    ${COMMON_INC_PATH}
    ${OPENCV_INC_PATH}
)

if(YAML_CPP_INCLUDE_DIR)
    include_directories(${YAML_CPP_INCLUDE_DIR})
endif()

link_directories(
    ${ACL_LIB_PATH}
    ${THIRDPART_LIB_PATH}
    ${OPENCV_LIB_PATH}
)

set(BASE_LINK_LIBS
    ${ASCENDCL_LIBRARY}
    ${ACL_DVPP_LIBRARY}
    stdc++
    ${ACLLITE_LIBRARY}
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVDEVICE_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWRESAMPLE_LIBRARY}
    ${SWSCALE_LIBRARY}
    ${AVFILTER_LIBRARY}
    ${OPENCV_CORE_LIBRARY}
    ${OPENCV_IMGPROC_LIBRARY}
    ${OPENCV_IMGCODECS_LIBRARY}
    dl
    rt
    pthread
)

if(OpenMP_CXX_FOUND AND OpenMP_CXX_LIBRARIES)
    list(APPEND BASE_LINK_LIBS ${OpenMP_CXX_LIBRARIES})
endif()

set(ALL_TARGETS
    sampleYOLOV8
)

function(add_yolov8_target target_name source_file)
    add_executable(${target_name} ${source_file})
    if(DEFINED target AND target STREQUAL "Simulator_Function")
        target_link_libraries(${target_name} funcsim)
    else()
        target_link_libraries(${target_name} ${BASE_LINK_LIBS} ${ARGN})
    endif()
endfunction()

add_yolov8_target(sampleYOLOV8 sampleYOLOV8.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sampleYOLO26.cpp")
    list(APPEND ALL_TARGETS sampleYOLO26)
    add_yolov8_target(sampleYOLO26 sampleYOLO26.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sampleYOLOV8_npu.cpp")
    list(APPEND ALL_TARGETS sampleYOLOV8_npu)
    add_yolov8_target(sampleYOLOV8_npu sampleYOLOV8_npu.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sampleYOLOV8_npu-rtsp.cpp")
    list(APPEND ALL_TARGETS sampleYOLOV8_npu_rtsp)
    add_yolov8_target(sampleYOLOV8_npu_rtsp sampleYOLOV8_npu-rtsp.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sampleYOLOV8_npu-rtsp-thread.cpp")
    list(APPEND ALL_TARGETS sampleYOLOV8_npu_rtsp_thread)
    add_yolov8_target(sampleYOLOV8_npu_rtsp_thread sampleYOLOV8_npu-rtsp-thread.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sampleYOLOV8_npu-rtsp-process.cpp")
    list(APPEND ALL_TARGETS sampleYOLOV8_npu_rtsp_process)
    add_yolov8_target(sampleYOLOV8_npu_rtsp_process sampleYOLOV8_npu-rtsp-process.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sampleYOLOV8_1280.cpp")
    list(APPEND ALL_TARGETS sampleYOLOV8_1280)
    add_yolov8_target(sampleYOLOV8_1280 sampleYOLOV8_1280.cpp)
endif()

if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
    set(YAML_LINK_LIB "${YAML_CPP_LIBRARY}")
    list(APPEND ALL_TARGETS
        sampleYOLOV8Seg
        sampleYOLOV8Cls
        sampleYOLOV8Obb
        sampleYOLOV8Pose
    )
    add_yolov8_target(sampleYOLOV8Seg sampleYOLOV8Seg.cpp ${YAML_LINK_LIB})
    add_yolov8_target(sampleYOLOV8Cls sampleYOLOV8Cls.cpp ${YAML_LINK_LIB})
    add_yolov8_target(sampleYOLOV8Obb sampleYOLOV8Obb.cpp ${YAML_LINK_LIB})
    add_yolov8_target(sampleYOLOV8Pose sampleYOLOV8Pose.cpp ${YAML_LINK_LIB})
endif()

install(TARGETS ${ALL_TARGETS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
