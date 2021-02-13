#ifndef HZIP_CODEC_CL_HELPER_H
#define HZIP_CODEC_CL_HELPER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>
#include <CL/cl2.hpp>
#include <errors/opencl.h>

namespace interlaced_ans::opencl {
    class ProgramProvider {
    private:
        static std::unordered_map<std::string, cl::Program> _program_map;
        static std::unordered_map<std::string, std::string> _src_map;
        static std::mutex _mutex;
    public:
        static cl::Program get(const std::string &kernel);

        static void register_program(const std::string &name, const std::string &src);

        static void compile(const std::string &kernel, const cl::Device &device);

        static void clear();
    };

    class DeviceProvider {
    private:
        static std::vector<cl::Device> _devices;
        static std::mutex _mutex;
        static uint64_t _device_index;
        static std::string _preferred_device_name;
    public:
        template<uint32_t device_type = CL_DEVICE_TYPE_ALL>
        static void load_devices() {
            _mutex.lock();
            _devices.clear();
            _device_index = 0;
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);

            for (const auto &platform: platforms) {
                std::vector<cl::Device> platform_devices;
                platform.getDevices(device_type, &platform_devices);
                _devices.insert(_devices.end(), platform_devices.begin(), platform_devices.end());
            }

            ProgramProvider::clear();
            _mutex.unlock();
        }

        static cl::Device get() {
            _mutex.lock();
            if (_devices.empty()) {
                _mutex.unlock();
                throw OpenCLErrors::InvalidOperationException("No devices were loaded");
            }
            cl::Device device = _devices[_device_index];
            if (_preferred_device_name.empty()) {
                _device_index = (_device_index + 1) % _devices.size();
            }
            _mutex.unlock();
            return device;
        }

        static bool empty() {
            _mutex.lock();
            bool v = _devices.empty();
            _mutex.unlock();
            return v;
        }

        static void set_preferred_device(const std::string &dev_name) {
            _preferred_device_name = dev_name;
            bool not_found_device = true;
            _mutex.lock();
            if (_devices.empty()) {
                _mutex.unlock();
                throw OpenCLErrors::InvalidOperationException("No devices were loaded");
            }
            for (int i = 0; i < _devices.size(); i++) {
                auto device = _devices[i];
                std::string dname = device.getInfo<CL_DEVICE_NAME>();

                if (dname.find(_preferred_device_name) != std::string::npos) {
                    not_found_device = false;
                    _device_index = i;
                    break;
                }
            }

            if (not_found_device) {
                _preferred_device_name = "";
            }

            _mutex.unlock();
        }
    };

    class KernelProvider {
    public:
        static cl::Kernel get(const std::string &kernel);

        static cl::Kernel get(const std::string &kernel, const std::string &name);
    };
}

#endif