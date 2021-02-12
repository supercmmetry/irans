#include <iostream>
#include <opencl/cl_helper.h>
#include <opencl/freq_dist.h>
#include <opencl/interlaced_rans64.h>
#include <chrono>
#include <io/writer.h>
#include <io/reader.h>

int main() {
    interlaced_ans::opencl::DeviceProvider::load_devices<CL_DEVICE_TYPE_GPU>();

    uint64_t size = 10485760;
    auto data = rainman::ptr<uint8_t>(size);
    for (uint64_t i = 0; i < size; i++) {
        data[i] = i & 0xf;
    }

    auto freq_dist = interlaced_ans::FrequencyDistribution();

    auto ftable = freq_dist.opencl_freq_dist(data, data.size() / 1024);

    auto codec = interlaced_ans::Rans64Codec(ftable);
    codec.normalize();
    codec.create_ctable();

    auto output = codec.opencl_encode(data, data.size() / 1024);

    {
        interlaced_ans::Writer writer("sample.irans");

        writer.write(ftable);
        writer.write(output);
    }

    {
        interlaced_ans::Reader reader("sample.irans");

        ftable = reader.read_ftable();
        output = reader.read_encoder_output();
    }

    auto decoder = interlaced_ans::Rans64Codec(ftable);
    decoder.create_ctable();

    auto decoded = decoder.opencl_decode(output);

    for (int i = 0; i < 20; i++) {
        std::cout << (int) decoded[i] << std::endl;
    }

}
