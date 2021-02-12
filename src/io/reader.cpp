#include "reader.h"

using namespace interlaced_ans;

Reader::Reader(const std::string &filename) {
    _file = std::fopen(filename.c_str(), "rb");
}

Reader::~Reader() {
    std::fclose(_file);
}

uint64_t Reader::read_u64() {
    uint64_t x;
    std::fread(&x, sizeof(x), 1, _file);

    return x;
}

rainman::ptr<uint64_t> Reader::read_ftable() {
    auto ftable = rainman::ptr<uint64_t>(256);
    std::fread(ftable.pointer(), sizeof(uint64_t), ftable.size(), _file);

    return ftable;
}

encoder_output Reader::read_encoder_output() {
    auto output = encoder_output();

    uint64_t true_size{};
    uint64_t stride_size{};
    uint64_t input_size{};

    // Read true-size, stride-size and input-size
    std::fread(&true_size, sizeof(true_size), 1, _file);
    std::fread(&stride_size, sizeof(stride_size), 1, _file);
    std::fread(&input_size, sizeof(input_size), 1, _file);

    output.input_size = input_size;
    output.stride_size = stride_size;

    uint64_t u32_size = stride_size >> 2;

    output.output_ns = rainman::ptr<uint64_t>(true_size);
    output.input_residues = rainman::ptr<uint64_t>(true_size);
    output.cl_outputs = rainman::ptr<uint32_t>(true_size * u32_size);

    // Read output_ns
    std::fread(output.output_ns.pointer(), sizeof(uint64_t), output.output_ns.size(), _file);

    // Read input-residues
    std::fread(output.input_residues.pointer(), sizeof(uint64_t), output.input_residues.size(), _file);

    // Write cl_outputs
    for (uint64_t i = 0; i < true_size; i++) {
        std::fread(output.cl_outputs.pointer() + u32_size * i, sizeof(uint32_t), output.output_ns[i], _file);
    }

    // Read residual_output
    uint64_t residual_output_size;
    std::fread(&residual_output_size, sizeof(residual_output_size), 1, _file);

    output.residual_output = rainman::ptr<uint32_t>(residual_output_size);
    std::fread(output.residual_output.pointer(), sizeof(uint32_t), output.residual_output.size(), _file);

    return output;
}
