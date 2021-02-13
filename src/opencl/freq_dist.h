#ifndef INTERLACED_ANS_FREQ_DIST_H
#define INTERLACED_ANS_FREQ_DIST_H

#include <rainman/rainman.h>
#include "cl_helper.h"

namespace interlaced_ans {
    class FrequencyDistribution {
    private:
        bool _verbose;
        static void register_kernel();

    public:
        FrequencyDistribution(bool verbose = false) : _verbose(verbose) {};

        rainman::ptr<uint64_t> opencl_freq_dist(const rainman::ptr<uint8_t> &input, uint64_t stride_size = 64);
    };
}

#endif
