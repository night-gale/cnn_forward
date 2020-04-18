cmake_minimum_required(VERSION 3.15)
project(cnn_forward)

set(CMAKE_CXX_STANDARD 14)

aux_source_directory(./src DIR_SRCS)
include_directories("D:/Drivers/eigen-3.3.7/eigen-3.3.7")
include_directories("include/")

add_executable(cnn_forward src/main.cpp src/conv.cpp src/relu.cpp src/pooling.cpp src/fully_connect.cpp src/softmax.cpp)