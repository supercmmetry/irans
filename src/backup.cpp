#include "backup.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <thread>
#include <atomic>
#include <multiblob.h>
#include <errors/base.h>

interlaced_ans::Backup::Backup(
        uint64_t n_threads,
        uint64_t max_kernels,
        uint64_t max_blob_size
) : _n_threads(n_threads), _max_kernels(max_kernels), _semaphore(n_threads), _max_blob_size(max_blob_size) {}

std::string interlaced_ans::Backup::get_path_suffix(const std::string &prefix, const std::string &path) {
    return path.substr(prefix.length(), path.length());
}

std::string interlaced_ans::Backup::remove_irans_ext(const std::string &path) {
    return path.substr(0, path.length() - 6);
}

void interlaced_ans::Backup::backup(const std::string &source_dir, const std::string &target_dir) {
    if (!std::filesystem::is_directory(source_dir)) {
        throw BaseErrors::InvalidOperationException("[BACKUP] Source directory not found");
    }

    if (std::filesystem::exists(target_dir)) {
        throw BaseErrors::InvalidOperationException("[BACKUP] Target directory not empty");
    }

    std::filesystem::create_directory(target_dir);

    std::vector<std::string> path_suffixes;

    // First create directories in target dir.
    std::cout << "[BACKUP] Creating target directories" << std::endl;

    for (auto &p : std::filesystem::recursive_directory_iterator(source_dir)) {
        std::string path_suffix = get_path_suffix(source_dir, p.path());
        if (std::filesystem::is_directory(p.path())) {
            std::string dir_path = target_dir + path_suffix;

            std::filesystem::create_directory(dir_path);
        } else {
            path_suffixes.push_back(path_suffix);
        }
    }

    std::atomic<uint64_t> total_size = 0;
    std::mutex mutex;

    for (auto &path_suffix : path_suffixes) {
        _semaphore.acquire();

        std::thread([this, source_dir, target_dir, &total_size, &mutex](const std::string &path_suffix) {

            std::string source_path = source_dir + path_suffix;
            std::string destination_path = target_dir + path_suffix + ".irans";

            // Optimize kernel count at 10KB per kernel.
            uint64_t file_size = std::filesystem::file_size(source_path);
            uint64_t blob_size = std::min((uint64_t) INTERLACED_ANS_DEFAULT_BLOB_SIZE, file_size);
            uint64_t estimated_kernel_count = file_size / 10240;
            uint64_t adaptive_kernel_count;

            {
                std::unique_lock<std::mutex> lk(mutex);
                _cv.wait(lk, [this, &total_size, blob_size]() {
                    return total_size + blob_size < _max_blob_size;
                });

                total_size += blob_size;
                adaptive_kernel_count = std::max(((_max_blob_size - total_size) / _max_blob_size) * _max_kernels,
                                                 _max_kernels / _n_threads);
            }

            uint64_t kernel_count = 1 + std::min(estimated_kernel_count, adaptive_kernel_count);

            auto codec = MultiBlobCodec(kernel_count, _max_blob_size);
            codec.compress_file(source_path, destination_path);

            total_size -= file_size;
            _cv.notify_all();

            std::cout << "[BACKUP] Completed backup for file: " << source_path << std::endl;

            _semaphore.release();
        }, path_suffix).detach();
    }

    // Wait for all threads to finish.
    _semaphore.wait_all();

    std::cout << "[BACKUP] Backup completed successfully" << std::endl;
}

void interlaced_ans::Backup::restore(const std::string &source_dir, const std::string &target_dir) {
    if (!std::filesystem::is_directory(source_dir)) {
        throw BaseErrors::InvalidOperationException("[BACKUP] Source directory not found");
    }

    if (std::filesystem::exists(target_dir)) {
        throw BaseErrors::InvalidOperationException("[BACKUP] Target directory not empty");
    }

    std::filesystem::create_directory(target_dir);

    std::vector<std::string> path_suffixes;

    // First create directories in target dir.
    std::cout << "[BACKUP] Restoring target directories" << std::endl;

    for (auto &p : std::filesystem::recursive_directory_iterator(source_dir)) {
        std::string path_suffix = get_path_suffix(source_dir, p.path());
        if (std::filesystem::is_directory(p.path())) {
            std::string dir_path = target_dir + path_suffix;

            std::filesystem::create_directory(dir_path);
        } else {
            path_suffixes.push_back(path_suffix);
        }
    }

    std::atomic<uint64_t> total_size = 0;
    std::mutex mutex;

    for (auto &path_suffix : path_suffixes) {
        _semaphore.acquire();

        std::thread([this, source_dir, target_dir, &total_size, &mutex](const std::string &path_suffix) {
            std::string source_path = source_dir + path_suffix;
            std::string destination_path = target_dir + remove_irans_ext(path_suffix);

            // Optimize kernel count at 10KB per kernel.
            uint64_t file_size = std::filesystem::file_size(source_path);
            uint64_t blob_size = std::min((uint64_t) INTERLACED_ANS_DEFAULT_BLOB_SIZE, file_size);
            uint64_t estimated_kernel_count = file_size / 10240;
            uint64_t adaptive_kernel_count;

            {
                std::unique_lock<std::mutex> lk(mutex);
                _cv.wait(lk, [this, &total_size, blob_size]() {
                    return total_size + blob_size < _max_blob_size;
                });

                total_size += blob_size;
                adaptive_kernel_count = std::max(((_max_blob_size - total_size) / _max_blob_size) * _max_kernels,
                                                 _max_kernels / _n_threads);
            }


            uint64_t kernel_count = 1 + std::min(estimated_kernel_count, adaptive_kernel_count);

            auto codec = MultiBlobCodec(kernel_count, _max_blob_size);
            codec.decompress_file(source_path, destination_path);

            total_size -= file_size;
            _cv.notify_all();

            std::cout << "[BACKUP] Completed restoration for file: " << source_path << std::endl;

            _semaphore.release();
        }, path_suffix).detach();
    }

    // Wait for all threads to finish.
    _semaphore.wait_all();

    std::cout << "[BACKUP] Restore completed successfully" << std::endl;
}


