#include "backup.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <multiblob.h>
#include <errors/base.h>
#include <openssl/sha.h>
#include <unordered_map>

#define HASH_CHUNK_SIZE 0x100000

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

    std::unordered_map<std::string, std::string> hashes;

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

        hashes[hash_string(path_suffix)] = hash_file(source_path);

        auto codec = MultiBlobCodec(kernel_count, _max_blob_size);
        codec.compress_file(source_path, destination_path);

        std::cout << "[BACKUP] Completed backup for file: " << source_path << std::endl;
    }

    auto hash_file_path = target_dir + "/hashes.dat";

    std::cout << "[BACKUP] Generating hashfile " << std::endl;
    FILE *fp = std::fopen(hash_file_path.c_str(), "wb");
    for (const auto &entry: hashes) {
        std::fwrite(entry.first.c_str(), 1, entry.first.length(), fp);
        std::fwrite(entry.second.c_str(), 1, entry.second.length(), fp);
    }

    std::fclose(fp);

    std::cout << "[BACKUP] Backup completed successfully" << std::endl;
}

void interlaced_ans::Backup::restore(const std::string &source_dir, const std::string &target_dir) {
    if (!std::filesystem::is_directory(source_dir)) {
        throw BaseErrors::InvalidOperationException("[BACKUP] Source directory not found");
    }

    auto hash_file_path = source_dir + "/hashes.dat";
    if (!std::filesystem::exists(hash_file_path) || std::filesystem::is_directory(hash_file_path)) {
        throw BaseErrors::InvalidOperationException("[BACKUP] Cannot find hashes.dat");
    }

    if (std::filesystem::exists(target_dir)) {
        throw BaseErrors::InvalidOperationException("[BACKUP] Target directory not empty");
    }

    std::cout << "[BACKUP] Loading hashes.dat" << std::endl;
    uint64_t total_hash_count = std::filesystem::file_size(hash_file_path) / (4 * SHA512_DIGEST_LENGTH);
    FILE *fp = fopen(hash_file_path.c_str(), "rb");

    std::unordered_map<std::string, std::string> hashes;
    for (uint64_t i = 0; i < total_hash_count; i++) {
        char hashkey[SHA512_DIGEST_LENGTH * 2];
        char hashval[SHA512_DIGEST_LENGTH * 2];
        std::fread(hashkey, 1, 2 * SHA512_DIGEST_LENGTH, fp);
        std::fread(hashval, 1, 2 * SHA512_DIGEST_LENGTH, fp);
        hashes[std::string(hashkey, 2 * SHA512_DIGEST_LENGTH)] = std::string(hashval, 2 * SHA512_DIGEST_LENGTH);
    }

    fclose(fp);

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

    std::vector<std::string> failed_files;

    for (auto &path_suffix : path_suffixes) {
        if (path_suffix == "/hashes.dat") {
            continue;
        }

        std::string source_path = source_dir + path_suffix;
        std::string original_suffix = remove_irans_ext(path_suffix);
        std::string destination_path = target_dir + original_suffix;

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

        std::cout << "[BACKUP] Validating file: " << destination_path << std::endl;
        auto hash = hash_file(destination_path);
        auto hash_suf = hash_string(original_suffix);

        if (!hashes.contains(hash_suf)) {
            std::cerr << "[BACKUP] Hash not found for file: " << original_suffix << std::endl;
            failed_files.push_back(destination_path);
            continue;
        }

        if (hash != hashes[hash_suf]) {
            std::cerr << "[BACKUP] Validation failed for: " << destination_path << std::endl;
            std::cerr << "[BACKUP] Original file hash: " << hashes[hash_suf] << std::endl;
            std::cerr << "[BACKUP] Restored file hash: " << hash << std::endl;
            failed_files.push_back(destination_path);

            continue;
        }

        std::cout << "[BACKUP] Completed restoration for file: " << source_path << std::endl;
    }

    if (failed_files.empty()) {
        std::cout << "[BACKUP] Restore completed" << std::endl;
        return;
    }

    std::cerr << "[BACKUP] Could not complete restoration for the following files: " << std::endl << std::endl;
    for (const auto &item: failed_files) {
        std::cout << "\t[-] " << item << std::endl;
    }
}

std::string interlaced_ans::Backup::hash_file(const std::string &file_path) {
    unsigned char hash[SHA512_DIGEST_LENGTH];
    SHA512_CTX sha512;
    SHA512_Init(&sha512);
    FILE *fp = std::fopen(file_path.c_str(), "rb");

    auto buffer = new uint8_t[HASH_CHUNK_SIZE];
    uint64_t buffer_size = 0;
    do {
        buffer_size = std::fread(buffer, 1, HASH_CHUNK_SIZE, fp);
        if (buffer_size == 0) {
            break;
        }

        SHA512_Update(&sha512, buffer, buffer_size);
    } while (buffer_size == HASH_CHUNK_SIZE);

    SHA512_Final(hash, &sha512);

    std::stringstream ss;
    for (unsigned char i : hash) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int) i;
    }

    std::fclose(fp);
    delete[] buffer;

    return ss.str();
}

std::string interlaced_ans::Backup::hash_string(const std::string &str) {
    unsigned char hash[SHA512_DIGEST_LENGTH];
    SHA512_CTX sha512;
    SHA512_Init(&sha512);
    SHA512_Update(&sha512, str.c_str(), str.length());
    SHA512_Final(hash, &sha512);
    std::stringstream ss;
    for (unsigned char i : hash) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int) i;
    }
    return ss.str();
}


