cmake_minimum_required(VERSION 3.17)
project(interlaced_ans)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(other)
include_directories(src)

add_executable(irans
        src/main.cpp
        src/opencl/cl_helper.h
        src/opencl/cl_helper.cpp
        src/errors/opencl.h
        src/opencl/freq_dist.h
        src/opencl/freq_dist.cpp
        src/opencl/interlaced_rans64.h
        src/opencl/interlaced_rans64.cpp
        src/io/writer.h
        src/io/writer.cpp
        src/io/reader.h
        src/io/reader.cpp src/multiblob.h src/multiblob.cpp src/errors/base.h)

target_link_libraries(irans PUBLIC argparse rainman OpenCL)
target_compile_definitions(irans PUBLIC CL_HPP_ENABLE_EXCEPTIONS)