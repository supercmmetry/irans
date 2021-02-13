#include "cl_helper.h"
#include <errors/opencl.h>
#include <iostream>

#define INTERLACED_ANS_OPENCL_BUILD_OPTIONS "-cl-std=CL2.0"

using namespace interlaced_ans::opencl;

std::unordered_map<std::string, cl::Program> ProgramProvider::_program_map;
std::unordered_map<std::string, std::string> ProgramProvider::_src_map;
std::mutex ProgramProvider::_mutex;

std::vector<cl::Device> DeviceProvider::_devices;
std::mutex DeviceProvider::_mutex;
uint64_t DeviceProvider::_device_index = 0;
std::string DeviceProvider::_preferred_device_name;


cl::Program ProgramProvider::get(const std::string &kernel) {
    _mutex.lock();
    if (!_program_map.contains(kernel)) {
        _mutex.unlock();
        throw OpenCLErrors::InvalidOperationException("Failed to load unregistered OpenCL kernel");
    }
    _mutex.unlock();
    return _program_map[kernel];
}

void ProgramProvider::clear() {
    _mutex.lock();
    _program_map.clear();
    _mutex.unlock();
}

void ProgramProvider::register_program(const std::string &name, const std::string &src) {
    _mutex.lock();
    if (!_program_map.contains(name)) {

        auto device = DeviceProvider::get();
        cl::Context context(device);
        auto program = cl::Program(context, src);
        try {
            program.build(INTERLACED_ANS_OPENCL_BUILD_OPTIONS);
        } catch (cl::Error &e) {
            if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
                // Check the build status
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
                if (status != CL_BUILD_ERROR) {
                    throw e;
                }

                // Get the build log
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << "Build log " << ":" << std::endl
                          << buildlog << std::endl;
            } else {
                throw e;
            }
        }

        _program_map[name] = program;
        _src_map[name] = src;
    }
    _mutex.unlock();
}

void ProgramProvider::compile(const std::string &kernel, const cl::Device &device) {
    _mutex.lock();
    if (!_program_map.contains(kernel)) {
        throw OpenCLErrors::InvalidOperationException("Cannot set device for unregistered OpenCL kernel");
    } else {
        cl::Context context(device);
        auto program = cl::Program(context, _src_map[kernel]);
        program.build(INTERLACED_ANS_OPENCL_BUILD_OPTIONS);
        _program_map[kernel] = program;
    }
    _mutex.unlock();
}

cl::Kernel KernelProvider::get(const std::string &kernel) {
    cl::Program program = ProgramProvider::get(kernel);
    return cl::Kernel(program, "run");
}

cl::Kernel KernelProvider::get(const std::string &kernel, const std::string &name) {
    cl::Program program = ProgramProvider::get(kernel);
    return cl::Kernel(program, name.c_str());
}