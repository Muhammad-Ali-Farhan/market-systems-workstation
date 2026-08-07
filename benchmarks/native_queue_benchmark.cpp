#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "RingBuffer.hpp"

namespace {

constexpr std::size_t PatternSize = 4096;
static_assert((PatternSize & (PatternSize - 1)) == 0);

struct PayloadChecksum {
    std::array<std::uint64_t, 4> lanes{};

    bool operator==(const PayloadChecksum&) const noexcept = default;
};

void accumulate(PayloadChecksum& checksum, const OrderBookState& state) noexcept {
    const auto words = std::bit_cast<std::array<std::uint64_t, 4>>(state);
    checksum.lanes[0] ^= words[0];
    checksum.lanes[1] ^= words[1];
    checksum.lanes[2] ^= words[2];
    checksum.lanes[3] ^= words[3];
}

std::uint64_t xor_one_to_n(std::uint64_t value) noexcept {
    switch (value & 3ULL) {
        case 0: return value;
        case 1: return 1;
        case 2: return value + 1;
        default: return 0;
    }
}

std::vector<OrderBookState> make_pattern() {
    std::mt19937_64 generator{2027};
    std::uniform_real_distribution<double> price_offset{0.0, 5.0};
    std::uniform_int_distribution<std::uint32_t> volume{1, 5'000'000};

    std::vector<OrderBookState> pattern;
    pattern.reserve(PatternSize);
    for (std::size_t index = 0; index < PatternSize; ++index) {
        const double best_bid = 100.0 + price_offset(generator);
        const double best_ask = best_bid + 0.01;
        pattern.push_back(OrderBookState{
            0,
            best_bid,
            best_ask,
            volume(generator),
            volume(generator),
        });
    }
    return pattern;
}

PayloadChecksum expected_checksum(
    const std::vector<OrderBookState>& pattern,
    std::uint64_t total) {
    PayloadChecksum pattern_checksum;
    std::vector<PayloadChecksum> prefix(PatternSize + 1);
    for (std::size_t index = 0; index < PatternSize; ++index) {
        auto state = pattern[index];
        state.timestamp_ns = 0;
        accumulate(pattern_checksum, state);
        prefix[index + 1] = prefix[index];
        accumulate(prefix[index + 1], state);
    }

    PayloadChecksum expected;
    expected.lanes[0] = xor_one_to_n(total);
    const std::uint64_t full_cycles = total / PatternSize;
    const std::size_t remainder = static_cast<std::size_t>(total % PatternSize);
    for (std::size_t lane = 1; lane < expected.lanes.size(); ++lane) {
        expected.lanes[lane] =
            ((full_cycles & 1ULL) != 0 ? pattern_checksum.lanes[lane] : 0ULL) ^
            prefix[remainder].lanes[lane];
    }
    return expected;
}

std::uint64_t fold_checksum(const PayloadChecksum& checksum) noexcept {
    return checksum.lanes[0] ^
           std::rotl(checksum.lanes[1], 13) ^
           std::rotl(checksum.lanes[2], 29) ^
           std::rotl(checksum.lanes[3], 47);
}

}  // namespace

int main(int argc, char** argv) {
    std::uint64_t total = 20'000'000;
    if (argc == 2) {
        total = std::strtoull(argv[1], nullptr, 10);
    }
    if (total == 0) {
        std::cerr << "Record count must be positive.\n";
        return 1;
    }

    const auto pattern = make_pattern();
    const auto expected_payload_checksum = expected_checksum(pattern, total);

    SPSCRingBuffer<65536> queue;
    std::vector<OrderBookState> batch(4096);
    std::atomic<bool> producer_ready{false};
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    PayloadChecksum consumer_checksum;

    std::thread producer([&] {
        producer_ready.store(true, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        for (std::uint64_t index = 1; index <= total; ++index) {
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            OrderBookState state = pattern[(index - 1) & (PatternSize - 1)];
            state.timestamp_ns = index;
            while (!queue.push(state)) {
                if (stop.load(std::memory_order_acquire)) {
                    return;
                }
                std::this_thread::yield();
            }
        }
    });

    while (!producer_ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto started = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);

    std::uint64_t consumed = 0;
    std::uint64_t expected_timestamp = 1;
    while (consumed < total) {
        const auto count = queue.consume_batch(batch.data(), batch.size());
        if (count == 0) {
            std::this_thread::yield();
            continue;
        }
        for (std::size_t index = 0; index < count; ++index) {
            const auto& state = batch[index];
            if (state.timestamp_ns != expected_timestamp++) {
                stop.store(true, std::memory_order_release);
                producer.join();
                std::cerr << "Order violation at record " << consumed + index << '\n';
                return 2;
            }
            accumulate(consumer_checksum, state);
        }
        consumed += count;
    }

    const auto stopped = std::chrono::steady_clock::now();
    producer.join();

    if (!(consumer_checksum == expected_payload_checksum)) {
        std::cerr << "Payload checksum mismatch: full 32-byte records were not preserved.\n";
        return 3;
    }

    const double elapsed = std::chrono::duration<double>(stopped - started).count();
    const double records_per_second = static_cast<double>(total) / elapsed;
    const double gib_per_second =
        records_per_second * static_cast<double>(sizeof(OrderBookState)) /
        static_cast<double>(1ULL << 30U);

    std::cout << std::fixed << std::setprecision(2)
              << "records=" << total
              << " bytes_per_record=" << sizeof(OrderBookState)
              << " seconds=" << elapsed
              << " records_per_second=" << records_per_second
              << " payload_gib_per_second=" << gib_per_second
              << " payload_checksum=" << fold_checksum(consumer_checksum)
              << '\n';
    return 0;
}