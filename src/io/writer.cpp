#include "writer.h"

using namespace interlaced_ans;

Writer::Writer(const std::string &filename) {
    remove(filename.c_str());
    _file = std::fopen(filename.c_str(), "wb");
}

void Writer::write(uint64_t x) {
    std::fwrite(&x, sizeof(x), 1, _file);
}

void Writer::write(const rainman::ptr<uint64_t> &ftable) {
    std::fwrite(ftable.pointer(), sizeof(uint64_t), ftable.size(), _file);
}

void Writer::write(const encoder_output& output) {
    uint64_t true_size = output.input_residues.size();

    // Write true-size, stride-size and input-size
    std::fwrite(&true_size, sizeof(true_size), 1, _file);
    std::fwrite(&output.stride_size, sizeof(output.stride_size), 1, _file);
    std::fwrite(&output.input_size, sizeof(output.input_size), 1, _file);

    // Write output_ns
    std::fwrite(output.output_ns.pointer(), sizeof(uint64_t), output.output_ns.size(), _file);

    // Write input-residues
    std::fwrite(output.input_residues.pointer(), sizeof(uint64_t), output.input_residues.size(), _file);

    // Write cl_outputs
    uint64_t u32_size = output.stride_size >> 2;

    for (uint64_t i = 0; i < true_size; i++) {
        std::fwrite(output.cl_outputs.pointer() + u32_size * i, sizeof(uint32_t), output.output_ns[i], _file);
    }

    // Write residual_output
    uint64_t residual_output_size = output.residual_output.size();
    std::fwrite(&residual_output_size, sizeof(residual_output_size), 1, _file);
    std::fwrite(output.residual_output.pointer(), sizeof(uint32_t), output.residual_output.size(), _file);
}

void Writer::write(const rainman::ptr<uint8_t> &data) {
    std::fwrite(data.pointer(), 1, data.size(), _file);
}

Writer::~Writer() {
    std::fclose(_file);
}
