#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <utility>      // std::move

template <typename T>
class FrameQueue {
private:
    std::queue<T>             queue_;
    mutable std::mutex        mtx_;
    std::condition_variable   cond_;       // 消费者等待："队列非空"
    std::condition_variable   cond_full_;  // 生产者等待："队列未满"
    size_t                    max_size_;
    bool                      stopped_ = false;
    size_t                    dropped_ = 0;

public:
    explicit FrameQueue(size_t size = 3) : max_size_(size) {}

    // ---- 生产者：推入（const 引用）----
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (queue_.size() >= max_size_) {
            queue_.pop();
            ++dropped_;
        }
        queue_.push(item);
        lock.unlock();
        cond_.notify_one();
    }

    // ---- 生产者：推入（移动语义）----
    void push(T&& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (queue_.size() >= max_size_) {
            queue_.pop();
            ++dropped_;
        }
        queue_.push(std::move(item));
        lock.unlock();
        cond_.notify_one();
    }

    // ---- 消费者：阻塞取出 ----
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait(lock, [this]() { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cond_full_.notify_one();
        return true;
    }

    // ---- 消费者：非阻塞取出 ----
    bool tryPop(T& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cond_full_.notify_one();
        return true;
    }

    // ---- 停止队列 ----
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stopped_ = true;
        }
        cond_.notify_all();
        cond_full_.notify_all();
    }

    // ---- 查询 ----
    bool isStopped() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return stopped_;
    }

    size_t droppedCount() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return dropped_;
    }
};
