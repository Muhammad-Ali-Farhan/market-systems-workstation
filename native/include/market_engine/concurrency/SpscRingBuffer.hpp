// Single-producer/single-consumer ring buffer used on the hot publication path.
// The implementation relies on strict endpoint ownership plus acquire/release
// ordering; it is intentionally not a general MPMC queue.


#pragma once

#include <algorithm>
#include <memory>
#include <atomic>
#include <cstddef>
#include <type_traits>

#include "market_engine/core/TopOfBookState.hpp"

// Bounded lock-free queue with one producer endpoint and one consumer endpoint.
// Capacity is fixed at compile time so publication never allocates on the hot path.
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

    // Publish one value; returns false rather than blocking when the queue is full.
    bool push(const Value& value) noexcept {
        // SPSC ownership invariant: only the producer writes tail_, so its own
        // current tail needs no synchronization. The acquire of head_ observes
        // slots the consumer has released back to the producer before reuse.
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
        const std::size_t current_head = head_.load(std::memory_order_acquire);
        if (current_tail - current_head >= Capacity) {
            return false;
        }
        // Write the payload before publishing the new tail. A consumer that sees
        // this release through its acquire load cannot observe a half-written slot.
        buffer_[current_tail & BufferMask] = value;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // Copy up to maximum_count published values in FIFO order into caller-owned storage.
    std::size_t consume_batch(
        Value* destination,
        std::size_t maximum_count) noexcept {
        if (destination == nullptr || maximum_count == 0) {
            return 0;
        }

        // Mirror of push(): the consumer owns head_, then acquires tail_ before
        // reading records that the producer has published.
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        const std::size_t current_tail = tail_.load(std::memory_order_acquire);
        const std::size_t available = current_tail - current_head;
        const std::size_t count = std::min(available, maximum_count);

        for (std::size_t index = 0; index < count; ++index) {
            destination[index] = buffer_[(current_head + index) & BufferMask];
        }
        if (count != 0) {
            // Release makes completed reads visible before the producer reuses
            // these slots after acquiring head_.
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
    // Resetting retains the allocated storage and only rewinds the endpoint indices.
    void reset() noexcept {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

private:
    // Power-of-two capacity turns modulo into a mask on the hot indexing path.
    static constexpr std::size_t BufferMask = Capacity - 1;
    std::unique_ptr<Value[]> buffer_;

    // Separate cache-line alignment prevents producer tail traffic and consumer
    // head traffic from repeatedly invalidating the same cache line (false sharing).
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

