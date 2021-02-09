#ifndef INTERLACED_ANS_FREQ_DIST_H
#define INTERLACED_ANS_FREQ_DIST_H

#include <rainman/rainman.h>
#include "cl_helper.h"

namespace interlaced_ans {
    class FrequencyDistribution {
    private:
        static void register_kernel();

    public:
        rainman::ptr<uint64_t> opencl_freq_dist(const rainman::ptr<uint8_t> &input, uint64_t stride_size = 64);
    };
}

#endif
