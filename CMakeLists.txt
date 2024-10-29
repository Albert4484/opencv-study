cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

project(OpenCVTest)

# opencv include
find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS} inc)

# My include
# 添加源文件
set(SOURCES
    DisplayImage.cpp        
    src/image_pro.cpp
    src/yolo.cpp
    src/inference.cpp
)

# 添加头文件
set(HEADERS
    inc/image_pro.hpp
    inc/yolo.hpps
    inc/inference.h
)

# 添加源文件
add_executable(DisplayImage ${SOURCES})

# 链接OpenCV库
target_link_libraries(DisplayImage ${OpenCV_LIBS})

target_include_directories(DisplayImage 
    PUBLIC inc                    
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)