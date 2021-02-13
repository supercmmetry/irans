#include "multiblob.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <io/reader.h>
#include <io/writer.h>
#include <opencl/freq_dist.h>
#include <opencl/interlaced_rans64.h>
#include <errors/base.h>

using namespace interlaced_ans;

void MultiBlobCodec::compress_file(const std::string &src, const std::string &dst) {
    if (src == dst) {
        throw BaseErrors::InvalidOperationException("Source and destination cannot be the same");
    }

    if (std::filesystem::exists(dst)) {
        throw BaseErrors::InvalidOperationException("Destination is not empty");
    }

    if (!std::filesystem::exists(src) || (std::filesystem::exists(src) && std::filesystem::is_directory(src))) {
        throw BaseErrors::InvalidOperationException("Source file not found");
    }

    uint64_t file_size = std::filesystem::file_size(src);
    uint64_t blob_count = (file_size / _blob_size) + (file_size % _blob_size != 0);

    uint64_t stride_size = _blob_size / _n_kernels;
    double total_time = 0.0;

    auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();

    Writer writer(dst);
    Reader reader(src);

    writer.write(blob_count);

    uint64_t counter = 0;
    while (file_size > 0) {
        if (_verbose) {
            std::cout << "[MULTIBLOB]\t\tCompressing blob (" << ++counter << ")" << std::endl;
        }

        uint64_t curr_blob_size;
        if (file_size < _blob_size) {
            curr_blob_size = file_size;
            file_size = 0;
        } else {
            curr_blob_size = _blob_size;
            file_size -= _blob_size;
        }

        auto tmp_data = reader.read_data(curr_blob_size);

        auto start_i = clock.now();
        auto freq_dist = FrequencyDistribution(_verbose);

        auto ftable = freq_dist.opencl_freq_dist(tmp_data, stride_size);
        auto codec = Rans64Codec(ftable, _verbose);
        codec.normalize();
        codec.create_ctable();

        auto output = codec.opencl_encode(tmp_data, stride_size);

        auto diff = ((double) (clock.now() - start_i).count()) / 1000000000.0;
        if (_verbose) {
            std::cout << "[MULTIBLOB]\t\tFinished compressing blob (" << counter << ") in " <<
                      diff << "s" << std::endl;

            total_time += diff;
        }

        writer.write(ftable);
        writer.write(output);
    }

    if (_verbose) {
        std::cout << "[MULTIBLOB]\t\tFinished compressing " << blob_count << " blob(s) in " <<
                  total_time << "s" << std::endl;

        std::cout << "[MULTIBLOB]\t\tOperation finished in " <<
                  ((double) (clock.now() - start).count()) / 1000000000.0 << "s" << std::endl;
    }
}

void MultiBlobCodec::decompress_file(const std::string &src, const std::string &dst) {
    if (src == dst) {
        throw BaseErrors::InvalidOperationException("Source and destination cannot be the same");
    }

    if (std::filesystem::exists(dst)) {
        throw BaseErrors::InvalidOperationException("Destination is not empty");
    }

    if (!std::filesystem::exists(src) || (std::filesystem::exists(src) && std::filesystem::is_directory(src))) {
        throw BaseErrors::InvalidOperationException("Source file not found");
    }

    auto clock = std::chrono::high_resolution_clock();
    auto start = clock.now();

    Writer writer(dst);
    Reader reader(src);

    uint64_t blob_count = reader.read_u64();
    uint64_t counter = 0;
    double total_time = 0.0;

    while (blob_count--) {
        if (_verbose) {
            std::cout << "[MULTIBLOB]\t\tDecompressing blob (" << ++counter << ")" << std::endl;
        }

        auto ftable = reader.read_ftable();
        auto output = reader.read_encoder_output();

        auto start_i = clock.now();

        auto codec = Rans64Codec(ftable, _verbose);
        codec.create_ctable();

        auto tmp_data = codec.opencl_decode(output);

        auto diff = ((double) (clock.now() - start_i).count()) / 1000000000.0;
        if (_verbose) {
            std::cout << "[MULTIBLOB]\t\tFinished decompressing blob (" << counter << ") in " <<
                      diff << "s" << std::endl;

            total_time += diff;
        }

        writer.write(tmp_data);
    }

    if (_verbose) {
        std::cout << "[MULTIBLOB]\t\tFinished decompressing " << counter << " blob(s) in " <<
                  total_time << "s" << std::endl;

        std::cout << "[MULTIBLOB]\t\tOperation finished in " <<
                  ((double) (clock.now() - start).count()) / 1000000000.0 << "s" << std::endl;
    }
}
