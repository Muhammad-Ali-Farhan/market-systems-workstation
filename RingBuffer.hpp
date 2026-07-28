
#pragma once

#include <algorithm>
#include <memory>
#include <atomic>
#include <cstddef>
#include <type_traits>

#include "OrderBook.hpp"

template <std::size_t Capacity, typename Value = OrderBookState>
class SPSCRingBuffer {
public:
    SPSCRingBuffer()
        : buffer_(std::make_unique<Value[]>(Capacity)) {}

    static_assert(Capacity > 0, "Capacity must be greater than zero.");
    static_assert(
        (Capacity & (Capacity - 1)) == 0,
        "Capacity must be an exact power of two.");
    static_assert(
        std::is_trivially_copyable_v<Value>,
        "SPSC queue values must be trivially copyable.");

    bool push(const Value& value) noexcept {
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
        const std::size_t current_head = head_.load(std::memory_order_acquire);
        if (current_tail - current_head >= Capacity) {
            return false;
        }
        buffer_[current_tail & BufferMask] = value;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    std::size_t consume_batch(
        Value* destination,
        std::size_t maximum_count) noexcept {
        if (destination == nullptr || maximum_count == 0) {
            return 0;
        }

        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        const std::size_t current_tail = tail_.load(std::memory_order_acquire);
        const std::size_t available = current_tail - current_head;
        const std::size_t count = std::min(available, maximum_count);

        for (std::size_t index = 0; index < count; ++index) {
            destination[index] = buffer_[(current_head + index) & BufferMask];
        }
        if (count != 0) {
            head_.store(current_head + count, std::memory_order_release);
        }
        return count;
    }

    // Approximate outside the owning producer/consumer threads. Exact when
    // called by either owner because only one endpoint mutates each index.
    std::size_t size_approx() const noexcept {
        const std::size_t current_head = head_.load(std::memory_order_acquire);
        const std::size_t current_tail = tail_.load(std::memory_order_acquire);
        return current_tail - current_head;
    }

    static constexpr std::size_t capacity() noexcept {
        return Capacity;
    }

    // Only call this when neither producer nor consumer is using the queue.
    void reset() noexcept {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t BufferMask = Capacity - 1;
    std::unique_ptr<Value[]> buffer_;

    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

