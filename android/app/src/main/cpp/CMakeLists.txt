cmake_minimum_required(VERSION 3.18)
project(docscan_jni CXX)

# the pipeline sources are shared with the desktop build in cpp/
set(DOCSCAN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp/src)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d flann calib3d photo)

add_library(docscan_jni SHARED
  jni.cpp
  ${DOCSCAN_SRC}/common.cpp ${DOCSCAN_SRC}/camera.cpp ${DOCSCAN_SRC}/matching.cpp
  ${DOCSCAN_SRC}/surface.cpp ${DOCSCAN_SRC}/render.cpp ${DOCSCAN_SRC}/pipeline.cpp
  ${DOCSCAN_SRC}/gpu_sift.cpp)

# GPU SIFT: shaders compiled with the NDK's glslc, Vulkan from the NDK sysroot
include(${DOCSCAN_SRC}/../shaders.cmake)
find_program(GLSLC glslc HINTS ${ANDROID_NDK}/shader-tools/linux-x86_64 REQUIRED)
docscan_add_shaders(docscan_jni ${GLSLC})
target_compile_definitions(docscan_jni PRIVATE DOCSCAN_GPU)
set_target_properties(docscan_jni PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(docscan_jni PRIVATE ${DOCSCAN_SRC} ${EIGEN_DIR})
target_link_libraries(docscan_jni PRIVATE ${OpenCV_LIBS} vulkan log android)
# 16 KB page alignment (required for new Play uploads, harmless otherwise)
target_link_options(docscan_jni PRIVATE -Wl,-z,max-page-size=16384 -Wl,--gc-sections)
