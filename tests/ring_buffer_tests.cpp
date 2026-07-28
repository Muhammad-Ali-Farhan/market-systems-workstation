
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "RingBuffer.hpp"

namespace {

[[noreturn]] void fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void basic_capacity_test() {
    SPSCRingBuffer<8> queue;
    OrderBookState state{1, 100.0, 101.0, 1, 2};
    for (std::size_t index = 0; index < 8; ++index) {
        state.timestamp_ns = index + 1;
        if (!queue.push(state)) {
            fail("queue rejected an element before reaching usable capacity");
        }
    }
    if (queue.push(state)) {
        fail("queue accepted an element while full");
    }

    OrderBookState output[8]{};
    const auto count = queue.consume_batch(output, 8);
    if (count != 8) {
        fail("consume_batch returned the wrong count");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (output[index].timestamp_ns != index + 1) {
            fail("queue order was not FIFO");
        }
    }
    if (queue.consume_batch(output, 8) != 0) {
        fail("empty queue returned data");
    }
}

void wraparound_test() {
    SPSCRingBuffer<16> queue;
    std::uint64_t expected = 1;
    for (std::uint64_t cycle = 0; cycle < 10'000; ++cycle) {
        for (std::uint64_t index = 0; index < 11; ++index) {
            OrderBookState state{cycle * 11 + index + 1, 100.0, 101.0, 1, 1};
            if (!queue.push(state)) {
                fail("wraparound push failed");
            }
        }
        OrderBookState batch[11]{};
        if (queue.consume_batch(batch, 11) != 11) {
            fail("wraparound batch count was wrong");
        }
        for (const auto& state : batch) {
            if (state.timestamp_ns != expected++) {
                fail("wraparound changed record order");
            }
        }
    }
}

void threaded_stress_test() {
    constexpr std::uint64_t Total = 2'000'000;
    SPSCRingBuffer<65536> queue;
    std::atomic<bool> producer_done{false};
    std::atomic<bool> failed{false};

    std::thread producer([&] {
        for (std::uint64_t value = 1; value <= Total; ++value) {
            const OrderBookState state{value, 100.0, 101.0, 2, 3};
            while (!queue.push(state)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::uint64_t expected = 1;
    std::vector<OrderBookState> batch(4096);
    while (!producer_done.load(std::memory_order_acquire) || expected <= Total) {
        const std::size_t count = queue.consume_batch(batch.data(), batch.size());
        if (count == 0) {
            std::this_thread::yield();
            continue;
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (batch[index].timestamp_ns != expected) {
                failed.store(true, std::memory_order_relaxed);
                break;
            }
            ++expected;
        }
        if (failed.load(std::memory_order_relaxed)) {
            break;
        }
    }
    producer.join();

    if (failed.load(std::memory_order_relaxed) || expected != Total + 1) {
        fail("threaded SPSC stress test lost, duplicated, or reordered data");
    }
}

}  // namespace

int main() {
    basic_capacity_test();
    wraparound_test();
    threaded_stress_test();
    std::cout << "ring_buffer_tests: PASS\n";
    return 0;
}

