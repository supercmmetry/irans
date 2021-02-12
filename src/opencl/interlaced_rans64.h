#ifndef INTERLACED_ANS_INTERLACED_RANS64_H
#define INTERLACED_ANS_INTERLACED_RANS64_H

#include <rainman/rainman.h>

namespace interlaced_ans {

    struct encoder_output {
        rainman::ptr<uint32_t> cl_outputs;
        rainman::ptr<uint64_t> output_ns;
        rainman::ptr<uint32_t> residual_output;
        rainman::ptr<uint64_t> input_residues;

        uint64_t stride_size;
        uint64_t input_size;
    };

    class Rans64Codec {
    private:
        rainman::ptr<uint64_t> _ftable;
        rainman::ptr<uint64_t> _ctable;

        void register_kernel();

        rainman::ptr<uint32_t> encode_residues(
                const rainman::ptr<uint8_t> &input,
                const rainman::ptr<uint64_t> &input_residues,
                uint64_t stride_size
        );

        void decode_residues(
                const rainman::ptr<uint8_t> &input,
                const rainman::ptr<uint64_t> &input_residues,
                const rainman::ptr<uint32_t> &encoded_residues,
                uint64_t stride_size
        );

        uint8_t inv_bs(uint64_t bs);

    public:
        explicit Rans64Codec(const rainman::ptr<uint64_t> &ftable) : _ftable(ftable) {}

        void normalize();

        void create_ctable();

        encoder_output opencl_encode(const rainman::ptr<uint8_t> &input, uint64_t stride_size);

        rainman::ptr<uint8_t> opencl_decode(const encoder_output &output);
    };

}

#endif
