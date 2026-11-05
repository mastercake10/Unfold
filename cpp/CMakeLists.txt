cmake_minimum_required(VERSION 3.16)
project(docscan CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d flann calib3d photo)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# the pipeline as a library (for the Android app via JNI) ...
add_library(docscan STATIC
  src/common.cpp src/camera.cpp src/matching.cpp src/surface.cpp src/render.cpp src/pipeline.cpp)
target_include_directories(docscan PUBLIC src)
target_link_libraries(docscan PUBLIC ${OpenCV_LIBS} Eigen3::Eigen)
target_compile_options(docscan PRIVATE -Wall)

# ... and a command-line tool
add_executable(docscan_cli src/main.cpp)
set_target_properties(docscan_cli PROPERTIES OUTPUT_NAME docscan)
target_link_libraries(docscan_cli PRIVATE docscan)
