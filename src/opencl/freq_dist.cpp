#include "freq_dist.h"
#include <iostream>


using namespace interlaced_ans;

void FrequencyDistribution::register_kernel() {
    opencl::ProgramProvider::register_program("freq_dist",

#include "freq_dist.cl"

    );
}

rainman::ptr<uint64_t> FrequencyDistribution::opencl_freq_dist(const rainman::ptr<uint8_t> &input, uint64_t stride_size) {
    register_kernel();

    auto kernel = opencl::KernelProvider::get("freq_dist");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    if (_verbose) {
        std::cout << "[OPENCL]\t\tRunning 'freq_dist.run' kernels on device: "
                  << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    uint64_t local_size = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

    uint64_t n = input.size();
    uint64_t true_size = (n / stride_size) + (n % stride_size != 0);
    uint64_t global_size = (true_size / local_size + (true_size % local_size != 0)) * local_size;

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(uint8_t), input.pointer());
    cl::Buffer buf_b(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, true_size * 256 * sizeof(uint64_t));
    auto host_ptr = rainman::ptr<uint64_t>(true_size << 8);

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_b);
    kernel.setArg(2, true_size);
    kernel.setArg(3, stride_size);
    kernel.setArg(4, input.size());

    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
    queue.enqueueBarrierWithWaitList();

    queue.enqueueReadBuffer(buf_b, CL_FALSE, 0, true_size * 256 * sizeof(uint64_t), host_ptr.pointer());
    queue.finish();

    auto result = rainman::ptr<uint64_t>(256);

    for (uint64_t i = 0; i < true_size; i++) {
        for (uint16_t j = 0; j < 256; j++) {
            result[j] += host_ptr[(i << 8) + j];
        }
    }

    return result;
}
