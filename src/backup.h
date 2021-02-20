#ifndef INTERLACED_ANS_BACKUP_H
#define INTERLACED_ANS_BACKUP_H

#include <cstdint>
#include <string>
#include <condition_variable>
#include <utils/semaphore.h>
#include <multiblob.h>

namespace interlaced_ans {
    class Backup {
    private:
        uint64_t _max_kernels;
        uint64_t _max_blob_size;

        static std::string get_path_suffix(const std::string &prefix, const std::string &path);

        static std::string remove_irans_ext(const std::string &path);

    public:
        Backup(uint64_t max_kernels, uint64_t max_blob_size = INTERLACED_ANS_DEFAULT_BLOB_SIZE);

        void backup(const std::string &source_dir, const std::string &target_dir);

        void restore(const std::string &source_dir, const std::string &target_dir);
    };
}

#endif
