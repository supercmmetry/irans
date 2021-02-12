#ifndef INTERLACED_ANS_READER_H
#define INTERLACED_ANS_READER_H

#include <cstdio>
#include <string>
#include <rainman/rainman.h>
#include <opencl/interlaced_rans64.h>

namespace interlaced_ans {
    class Reader {
    private:
        FILE *_file;

    public:
        Reader(const std::string &filename);

        uint64_t read_u64();

        rainman::ptr<uint64_t> read_ftable();

        encoder_output read_encoder_output();

        ~Reader();
    };
}

#endif
