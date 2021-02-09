#include <iostream>
#include <opencl/cl_helper.h>
#include <opencl/freq_dist.h>
#include <chrono>

int main() {
    interlaced_ans::opencl::DeviceProvider::load_devices<CL_DEVICE_TYPE_GPU>();

    uint64_t size = 1048576;
    auto z = rainman::ptr<uint8_t>(size);

    for (uint64_t i = 0; i < size; i++) {
        z[i] = i & 0xff;
    }

    auto fdist = interlaced_ans::FrequencyDistribution();

    auto clock = std::chrono::high_resolution_clock();

    auto start = clock.now();
    auto res = fdist.opencl_freq_dist(z, 1048576);
    std::cout << "Speed of freq-dist: " << (size * 1000000000.0 / 1048576) / (clock.now() - start).count() << " MBps" << std::endl;

    for (int i = 0; i < 256; i++) {
        std::cout << i << ": " << res[i] << std::endl;
    }

}
