cmake_minimum_required(VERSION 3.17)
project(interlaced_ans)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(other)
include_directories(src)

add_executable(interlaced_ans
        src/main.cpp
        src/opencl/cl_helper.h
        src/opencl/cl_helper.cpp
        src/errors/opencl.h
        src/opencl/freq_dist.h
        src/opencl/freq_dist.cpp src/opencl/interlaced_rans64.h src/opencl/interlaced_rans64.cpp)

target_link_libraries(interlaced_ans PUBLIC rainman OpenCL)