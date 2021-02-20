#include "backup.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <multiblob.h>
#include <errors/base.h>

interlaced_ans::Backup::Backup(
        uint64_t max_kernels,
        uint64_t max_blob_size
) : _max_kernels(max_kernels), _max_blob_size(max_blob_size) {}

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

    for (auto &path_suffix : path_suffixes) {
        std::string source_path = source_dir + path_suffix;
        std::string destination_path = target_dir + path_suffix + ".irans";

        // Optimize kernel count at 1KiB per kernel.
        uint64_t file_size = std::filesystem::file_size(source_path);
        uint64_t estimated_kernel_count = file_size / 1024;

        uint64_t kernel_count = std::max(
                uint64_t(INTERLACED_ANS_DEFAULT_N_KERNELS),
                std::min(estimated_kernel_count, _max_kernels)
        );

        std::cout << "[BACKUP] Processing file: " << source_path << " with " << kernel_count << " kernel(s)"
                  << std::endl;

        auto codec = MultiBlobCodec(kernel_count, _max_blob_size);
        codec.compress_file(source_path, destination_path);

        std::cout << "[BACKUP] Completed backup for file: " << source_path << std::endl;
    }


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

    for (auto &path_suffix : path_suffixes) {
        std::string source_path = source_dir + path_suffix;
        std::string destination_path = target_dir + remove_irans_ext(path_suffix);

        // Optimize kernel count at 1KiB per kernel.
        uint64_t file_size = std::filesystem::file_size(source_path);
        uint64_t estimated_kernel_count = file_size / 1024;

        uint64_t kernel_count = std::max(
                uint64_t(INTERLACED_ANS_DEFAULT_N_KERNELS),
                std::min(estimated_kernel_count, _max_kernels)
        );

        std::cout << "[BACKUP] Processing file: " << source_path << " with " << kernel_count << " kernel(s)"
                  << std::endl;

        auto codec = MultiBlobCodec(kernel_count, _max_blob_size);
        codec.decompress_file(source_path, destination_path);

        std::cout << "[BACKUP] Completed restoration for file: " << source_path << std::endl;
    }

    std::cout << "[BACKUP] Restore completed successfully" << std::endl;
}


