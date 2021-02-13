#ifndef INTERLACED_ANS_MULTIBLOB_H
#define INTERLACED_ANS_MULTIBLOB_H

// Default blob size: 100MB
#define INTERLACED_ANS_DEFAULT_BLOB_SIZE 104857600

// Default kernels: 64
#define INTERLACED_ANS_DEFAULT_N_KERNELS 64

#include <cstdint>
#include <string>

namespace interlaced_ans {
    class MultiBlobCodec {
    private:
        uint64_t _blob_size;
        uint64_t _n_kernels;
        bool _verbose;
    public:
        MultiBlobCodec(
                uint64_t n_kernels = INTERLACED_ANS_DEFAULT_N_KERNELS,
                uint64_t blob_size = INTERLACED_ANS_DEFAULT_BLOB_SIZE,
                bool verbose = false
        ) : _n_kernels(n_kernels), _blob_size(blob_size), _verbose(verbose) {}

        void compress_file(const std::string &src, const std::string &dst);

        void decompress_file(const std::string &src, const std::string &dst);
    };
}

#endif
