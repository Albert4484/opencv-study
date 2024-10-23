cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

project(OpenCVTest)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加你的源文件
add_executable(DisplayImage DisplayImage.cpp)

# 链接OpenCV库
target_link_libraries(DisplayImage ${OpenCV_LIBS})