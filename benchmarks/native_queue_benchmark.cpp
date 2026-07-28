
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "RingBuffer.hpp"

int main(int argc, char** argv) {
    std::uint64_t total = 20'000'000;
    if (argc == 2) {
        total = std::strtoull(argv[1], nullptr, 10);
    }
    if (total == 0) {
        std::cerr << "Record count must be positive.\n";
        return 1;
    }

    SPSCRingBuffer<65536> queue;
    std::vector<OrderBookState> batch(4096);
    const auto started = std::chrono::steady_clock::now();

    std::thread producer([&] {
        for (std::uint64_t index = 1; index <= total; ++index) {
            const OrderBookState state{index, 100.0, 100.01, 1, 1};
            while (!queue.push(state)) {
                std::this_thread::yield();
            }
        }
    });

    std::uint64_t consumed = 0;
    std::uint64_t expected = 1;
    while (consumed < total) {
        const auto count = queue.consume_batch(batch.data(), batch.size());
        if (count == 0) {
            std::this_thread::yield();
            continue;
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (batch[index].timestamp_ns != expected++) {
                std::cerr << "Order violation at record " << consumed + index << '\n';
                producer.join();
                return 1;
            }
        }
        consumed += count;
    }
    producer.join();

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << std::fixed << std::setprecision(2)
              << "records=" << total
              << " seconds=" << elapsed
              << " records_per_second=" << total / elapsed << '\n';
    return 0;
}

