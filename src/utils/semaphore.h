#ifndef INTERLACED_ANS_UTILS_SEMAPHORE_H
#define INTERLACED_ANS_UTILS_SEMAPHORE_H

#include <mutex>
#include <condition_variable>

class Semaphore {
private:
    std::mutex _mutex;
    std::condition_variable _cv;
    uint64_t _count;
    uint64_t _max_count;

public:
    Semaphore(uint64_t count = 0) : _count(count), _max_count(count) {}

    void acquire();

    void release();

    void wait_all();
};

#endif
