#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#include "L2Book.hpp"

using quant::l2::DepthUpdate;
using quant::l2::FlatOrderBook;
using quant::l2::Level;
using quant::l2::MapOrderBook;
using quant::l2::Snapshot;

namespace {

template <typename Book>
double snapshot_microseconds(Book& book, const Snapshot& snapshot, int trials) {
    std::vector<double> samples;
    samples.reserve(trials);
    for (int trial = 0; trial < trials; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        book.install_snapshot(snapshot);
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <typename Book>
std::pair<double, double> update_benchmark(
    Book& book,
    const Snapshot& snapshot,
    const std::vector<DepthUpdate>& updates) {
    book.install_snapshot(snapshot);
    constexpr std::size_t Batch = 1'000;
    std::vector<double> batch_nanoseconds;
    batch_nanoseconds.reserve(updates.size() / Batch + 1);
    const auto all_start = std::chrono::steady_clock::now();
    for (std::size_t start = 0; start < updates.size(); start += Batch) {
        const auto batch_start = std::chrono::steady_clock::now();
        const auto stop = std::min(start + Batch, updates.size());
        for (std::size_t index = start; index < stop; ++index) {
            book.apply(updates[index]);
        }
        const auto batch_stop = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double, std::nano>(
            batch_stop - batch_start).count();
        batch_nanoseconds.push_back(elapsed / static_cast<double>(stop - start));
    }
    const auto all_stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(all_stop - all_start).count();
    std::sort(batch_nanoseconds.begin(), batch_nanoseconds.end());
    const auto p99_index = static_cast<std::size_t>(
        std::floor(0.99 * static_cast<double>(batch_nanoseconds.size() - 1)));
    return {updates.size() / seconds, batch_nanoseconds[p99_index]};
}

Snapshot make_snapshot() {
    Snapshot snapshot;
    snapshot.receipt_timestamp_ns = 1;
    snapshot.last_update_id = 1;
    snapshot.bids.reserve(5'000);
    snapshot.asks.reserve(5'000);
    constexpr std::int64_t Mid = 6'000'000'000'000LL;
    constexpr std::int64_t Tick = 1'000'000LL;
    for (std::int64_t index = 0; index < 5'000; ++index) {
        snapshot.bids.push_back(Level{Mid - Tick * (index + 1), 100'000'000ULL + static_cast<std::uint64_t>(index)});
        snapshot.asks.push_back(Level{Mid + Tick * (index + 1), 100'000'000ULL + static_cast<std::uint64_t>(index)});
    }
    return snapshot;
}

std::vector<DepthUpdate> make_updates(std::size_t count) {
    std::mt19937_64 generator{2027};
    std::uniform_int_distribution<std::int64_t> level_distribution(1, 5'500);
    std::uniform_int_distribution<std::uint64_t> quantity_distribution(1, 2'000'000'000ULL);
    std::bernoulli_distribution side_distribution{0.5};
    constexpr std::int64_t Mid = 6'000'000'000'000LL;
    constexpr std::int64_t Tick = 1'000'000LL;
    std::vector<DepthUpdate> updates;
    updates.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto id = static_cast<std::uint64_t>(index + 2);
        const auto level = level_distribution(generator);
        const bool bid = side_distribution(generator);
        DepthUpdate update;
        update.receipt_timestamp_ns = id;
        update.event_time_ms = id;
        update.first_update_id = id;
        update.final_update_id = id;
        const Level value{
            bid ? Mid - Tick * level : Mid + Tick * level,
            quantity_distribution(generator),
        };
        (bid ? update.bids : update.asks).push_back(value);
        updates.push_back(std::move(update));
    }
    return updates;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t update_count = 1'000'000;
    if (argc == 2) {
        update_count = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (update_count == 0) {
        std::cerr << "Update count must be positive.\n";
        return 1;
    }
    const auto snapshot = make_snapshot();
    const auto updates = make_updates(update_count);
    MapOrderBook map_book;
    FlatOrderBook flat_book;
    const double map_snapshot_us = snapshot_microseconds(map_book, snapshot, 7);
    const double flat_snapshot_us = snapshot_microseconds(flat_book, snapshot, 7);
    const auto [map_rate, map_p99_ns] = update_benchmark(map_book, snapshot, updates);
    const auto [flat_rate, flat_p99_ns] = update_benchmark(flat_book, snapshot, updates);
    if (map_book.state_hash() != flat_book.state_hash()) {
        std::cerr << "Reference and flat books diverged.\n";
        return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t extraction_hash = 0;
    for (int index = 0; index < 100'000; ++index) {
        for (const auto& level : flat_book.bids(20)) {
            extraction_hash = extraction_hash * 1099511628211ULL +
                static_cast<std::uint64_t>(level.price) + level.quantity;
        }
        for (const auto& level : flat_book.asks(20)) {
            extraction_hash = extraction_hash * 1099511628211ULL +
                static_cast<std::uint64_t>(level.price) + level.quantity;
        }
    }
    const double top20_ns = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - started).count() / 100'000.0;

    std::cout << std::fixed << std::setprecision(2)
              << "{\n"
              << "  \"updates\": " << update_count << ",\n"
              << "  \"map_snapshot_median_us\": " << map_snapshot_us << ",\n"
              << "  \"flat_snapshot_median_us\": " << flat_snapshot_us << ",\n"
              << "  \"map_updates_per_second\": " << map_rate << ",\n"
              << "  \"flat_updates_per_second\": " << flat_rate << ",\n"
              << "  \"map_batch_p99_ns_per_update\": " << map_p99_ns << ",\n"
              << "  \"flat_batch_p99_ns_per_update\": " << flat_p99_ns << ",\n"
              << "  \"flat_top20_extract_ns\": " << top20_ns << ",\n"
              << "  \"final_state_hash\": " << flat_book.state_hash() << ",\n"
              << "  \"anti_optimization_hash\": " << extraction_hash << "\n"
              << "}\n";
    return 0;
}
