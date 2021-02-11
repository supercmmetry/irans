#ifndef INTERLACED_ANS_WRITER_H
#define INTERLACED_ANS_WRITER_H

#include <string>
#include <opencl/interlaced_rans64.h>

namespace interlaced_ans {
    class Writer {
    private:
        FILE *_file;

    public:
        Writer(const std::string &filename);

        void write(uint64_t x);

        void write(const rainman::ptr<uint64_t> &ftable);

        void write(const encoder_output& output);

        ~Writer();
    };
}

#endif
