#include "interlaced_rans64.h"
#include <vector>
#include "cl_helper.h"

using namespace interlaced_ans;

#define RANS64_SCALE 24

void Rans64Codec::register_kernel() {
    opencl::ProgramProvider::register_kernel("interlaced_rans64",

#include "interlaced_rans64.cl"

    );
}

encoder_output Rans64Codec::opencl_encode(const rainman::ptr<uint8_t> &input, uint64_t stride_size) {
    register_kernel();

    auto kernel = opencl::KernelProvider::get("interlaced_rans64", "encode");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();
    uint64_t local_size = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

    uint64_t n = input.size();
    uint64_t true_size = (n / stride_size) + (n % stride_size != 0);
    uint64_t global_size = (true_size / local_size + (true_size % local_size != 0)) * local_size;
    uint64_t output_size = (true_size * stride_size) >> 2;

    cl::Buffer buf_input(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                         input.size() * sizeof(uint8_t), input.pointer());

    cl::Buffer buf_ftable(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                          _ftable.size() * sizeof(uint64_t), _ftable.pointer());

    cl::Buffer buf_ctable(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                          _ctable.size() * sizeof(uint64_t), _ctable.pointer());

    cl::Buffer buf_output(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                          output_size * sizeof(uint32_t));

    cl::Buffer buf_output_ns(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                             true_size * sizeof(uint64_t));

    cl::Buffer buf_input_residues(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                                  true_size * sizeof(uint64_t));


    kernel.setArg(0, buf_input);
    kernel.setArg(1, input.size());
    kernel.setArg(2, buf_ftable);
    kernel.setArg(3, buf_ctable);
    kernel.setArg(4, buf_output);
    kernel.setArg(5, buf_output_ns);
    kernel.setArg(6, buf_input_residues);
    kernel.setArg(7, output_size);
    kernel.setArg(8, true_size);
    kernel.setArg(9, stride_size);

    auto output = rainman::ptr<uint32_t>(output_size);
    auto output_ns = rainman::ptr<uint64_t>(true_size);
    auto input_residues = rainman::ptr<uint64_t>(true_size);

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
    queue.enqueueBarrierWithWaitList();

    queue.enqueueReadBuffer(buf_output, CL_FALSE, 0, output_size * sizeof(uint32_t), output.pointer());
    queue.enqueueReadBuffer(buf_output_ns, CL_FALSE, 0, true_size * sizeof(uint64_t), output_ns.pointer());
    queue.enqueueReadBuffer(buf_input_residues, CL_FALSE, 0, true_size * sizeof(uint64_t), input_residues.pointer());

    queue.finish();

    auto residual_output = encode_residues(input, input_residues, stride_size);
    return encoder_output{
        .cl_outputs = output,
        .output_ns = output_ns,
        .residual_output = residual_output,
        .input_residues = input_residues,
    };

}

rainman::ptr<uint8_t> Rans64Codec::opencl_decode(const rainman::ptr<uint8_t> &input) {
    return rainman::ptr<uint8_t>();
}

void Rans64Codec::normalize() {
    uint64_t sum = 256;
    for (int i = 0; i < 256; i++) {
        sum += _ftable[i];
    }

    uint64_t ssum = 0;
    uint64_t mul_factor = (1ull << RANS64_SCALE) - 256;

    for (int i = 0; i < 256; i++) {
        uint64_t value = 1 + (_ftable[i] + 1) * mul_factor / sum;
        ssum += value - 1;
        _ftable[i] = value;
    }

    // Disperse residues uniformly.
    ssum = mul_factor - ssum;
    for (int i = 0; ssum > 0; i++, ssum--) {
        _ftable[i]++;
    }
}

void Rans64Codec::create_ctable() {
    uint64_t bs = 0;
    _ctable = rainman::ptr<uint64_t>(256);
    _ctable[0] = 0;

    for (int i = 0; i < 255; i++) {
        bs += _ftable[i];
        _ctable[i + 1] = bs;
    }
}

rainman::ptr<uint32_t> Rans64Codec::encode_residues(
        const rainman::ptr<uint8_t> &input,
        const rainman::ptr<uint64_t> &input_residues,
        uint64_t stride_size
) {
    const uint64_t lower_bound = 1 << 31;
    const uint64_t up_prefix = (lower_bound >> RANS64_SCALE) << 32;

    uint64_t state = lower_bound;
    std::vector<uint32_t> out;

    for (uint64_t i = 0; i < input_residues.size(); i++) {
        uint64_t residue = input_residues[i];
        if (residue == 0) {
            continue;
        }

        int64_t start_index = stride_size * i;
        int64_t end_index = start_index + residue - 1;

        for (int64_t j = end_index; j >= start_index; j--) {
            auto symbol = input[j];
            auto ls = _ftable[symbol];
            auto bs = _ctable[symbol];
            uint64_t upper_bound = up_prefix * symbol;

            if (state >= upper_bound) {
                out.push_back(state);
                state >>= 32;
            }

            state = ((state / ls) << RANS64_SCALE) + bs + (state % ls);
        }
    }

    out.push_back(state);
    out.push_back(state >> 32);

    auto output = rainman::ptr<uint32_t>(out.size());
    for (uint64_t i = 0; i < out.size(); i++) {
        output[i] = out[i];
    }

    return output;
}
