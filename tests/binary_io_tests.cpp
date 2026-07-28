
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "BinaryFormat.hpp"
#include "BinaryRecorder.hpp"
#include "BinaryReplay.hpp"

namespace {

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

std::filesystem::path unique_path(const std::string& name) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("quant_engine_" + name + "_" + std::to_string(stamp) + ".qbin");
}

void remove_recording(const std::filesystem::path& path) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(path.string() + ".qids", ignored);
    std::filesystem::remove(path.string() + ".meta.json", ignored);
}

std::vector<OrderBookState> make_records(std::size_t count) {
    std::vector<OrderBookState> records;
    records.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        records.push_back(OrderBookState{
            1'000'000'000ULL + static_cast<std::uint64_t>(index) * 1'000'000ULL,
            100.0 + static_cast<double>(index) * 0.001,
            100.01 + static_cast<double>(index) * 0.001,
            static_cast<std::uint32_t>(1'000 + index),
            static_cast<std::uint32_t>(2'000 + index),
        });
    }
    return records;
}

void round_trip_test() {
    const auto path = unique_path("round_trip");
    remove_recording(path);
    const auto source = make_records(4'000);

    BinaryRecorder recorder;
    recorder.start(path.string());
    recorder.mark_session_boundary("connection_start");
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (!recorder.record(source[index], 10'000 + index)) {
            fail("recorder rejected a valid record");
        }
    }
    recorder.stop();

    if (recorder.accepted_records() != source.size() ||
        recorder.recorded_records() != source.size() ||
        recorder.dropped_records() != 0 || recorder.write_errors() != 0) {
        fail("recorder counters do not describe the written payload");
    }
    if (!std::filesystem::exists(path.string() + ".meta.json")) {
        fail("recorder did not finalize its metadata sidecar");
    }
    if (!std::filesystem::exists(path.string() + ".qids")) {
        fail("recorder did not preserve exchange update IDs");
    }
    {
        std::ifstream market_input(path, std::ios::binary);
        quant::BinaryFileHeader market_header{};
        market_input.read(
            reinterpret_cast<char*>(&market_header), sizeof(market_header));
        std::ifstream input(path.string() + ".qids", std::ios::binary);
        quant::UpdateIdFileHeader header{};
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!quant::valid_update_id_header(
                header, market_header.created_unix_ns)) {
            fail("update-ID header is invalid");
        }
        std::vector<std::uint64_t> ids(source.size());
        input.read(
            reinterpret_cast<char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(std::uint64_t)));
        if (!input || ids.front() != 10'000 ||
            ids.back() != 10'000 + source.size() - 1) {
            fail("exchange update IDs were not written in record order");
        }
    }

    SPSCRingBuffer<65536> queue;
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> backpressure{0};
    std::atomic<std::uint64_t> errors{0};
    BinaryReplay replay(queue, replayed, backpressure, errors);
    std::atomic<bool> running{true};
    replay.run(running, path.string(), 0.0);

    if (errors.load() != 0 || replayed.load() != source.size()) {
        fail("valid recording did not replay completely");
    }
    std::vector<OrderBookState> destination(source.size());
    const auto consumed = queue.consume_batch(destination.data(), destination.size());
    if (consumed != source.size()) {
        fail("replay queue returned the wrong record count");
    }
    for (std::size_t index = 0; index < source.size(); ++index) {
        const auto& expected = source[index];
        const auto& actual = destination[index];
        if (actual.timestamp_ns != expected.timestamp_ns ||
            actual.best_bid != expected.best_bid ||
            actual.best_ask != expected.best_ask ||
            actual.bid_volume != expected.bid_volume ||
            actual.ask_volume != expected.ask_volume) {
            fail("replay did not preserve every source field exactly");
        }
    }
    remove_recording(path);
}

void overwrite_protection_test() {
    const auto path = unique_path("overwrite");
    remove_recording(path);
    {
        std::ofstream output(path, std::ios::binary);
        output << "do not overwrite";
    }
    BinaryRecorder recorder;
    bool threw = false;
    try {
        recorder.start(path.string());
    } catch (const std::exception&) {
        threw = true;
    }
    if (!threw) {
        fail("recorder overwrote an existing file");
    }
    remove_recording(path);
}


void interrupted_current_format_test() {
    const auto path = unique_path("interrupted");
    remove_recording(path);
    const auto source = make_records(8);
    BinaryRecorder recorder;
    recorder.start(path.string());
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (!recorder.record(source[index], 20'000 + index)) {
            fail("recorder rejected setup data");
        }
    }
    recorder.stop();
    std::error_code ignored;
    std::filesystem::remove(path.string() + ".meta.json", ignored);

    SPSCRingBuffer<65536> queue;
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> backpressure{0};
    std::atomic<std::uint64_t> errors{0};
    BinaryReplay replay(queue, replayed, backpressure, errors);
    std::atomic<bool> running{true};
    replay.run(running, path.string(), 0.0);
    if (errors.load() != 1 || replayed.load() != 0 ||
        replay.last_error().find("completion sidecar") == std::string::npos) {
        fail("interrupted current-format recording was not rejected");
    }
    remove_recording(path);
}

void zero_update_id_test() {
    const auto path = unique_path("zero_update_id");
    remove_recording(path);
    const auto source = make_records(8);
    BinaryRecorder recorder;
    recorder.start(path.string());
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (!recorder.record(source[index], 30'000 + index)) {
            fail("recorder rejected setup data");
        }
    }
    recorder.stop();

    {
        std::fstream ids(
            path.string() + ".qids",
            std::ios::binary | std::ios::in | std::ios::out);
        ids.seekp(static_cast<std::streamoff>(sizeof(quant::UpdateIdFileHeader)));
        const std::uint64_t zero = 0;
        ids.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    }

    SPSCRingBuffer<65536> queue;
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> backpressure{0};
    std::atomic<std::uint64_t> errors{0};
    BinaryReplay replay(queue, replayed, backpressure, errors);
    std::atomic<bool> running{true};
    replay.run(running, path.string(), 0.0);
    if (errors.load() != 1 || replayed.load() != 0 ||
        replay.last_error().find("zero exchange update ID") == std::string::npos) {
        fail("zero exchange update ID was not rejected");
    }
    remove_recording(path);
}

void metadata_finalization_failure_test() {
    const auto path = unique_path("metadata_failure");
    remove_recording(path);
    const auto source = make_records(16);
    BinaryRecorder recorder;
    recorder.start(path.string());
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (!recorder.record(source[index], 40'000 + index)) {
            fail("recorder rejected metadata-failure setup data");
        }
    }

    const std::filesystem::path sidecar{path.string() + ".meta.json"};
    std::error_code directory_error;
    std::filesystem::create_directory(sidecar, directory_error);
    if (directory_error) {
        fail("could not create metadata finalization obstacle");
    }
    recorder.stop();
    if (recorder.write_errors() != 1) {
        fail("metadata finalization failure was not exposed as a write error");
    }
    remove_recording(path);
}



void unsupported_market_header_flags_test() {
    const auto path = unique_path("market_flags");
    remove_recording(path);
    {
        std::ofstream output(path, std::ios::binary);
        auto header = quant::make_binary_header();
        header.flags = 1;
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    }

    SPSCRingBuffer<65536> queue;
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> backpressure{0};
    std::atomic<std::uint64_t> errors{0};
    BinaryReplay replay(queue, replayed, backpressure, errors);
    std::atomic<bool> running{true};
    replay.run(running, path.string(), 0.0);
    if (errors.load() != 1 || replayed.load() != 0 ||
        replay.last_error().find("header is invalid") == std::string::npos) {
        fail("unsupported market-header flags were not rejected");
    }
    remove_recording(path);
}

void unsupported_update_header_flags_test() {
    const auto path = unique_path("update_flags");
    remove_recording(path);
    const auto source = make_records(8);
    BinaryRecorder recorder;
    recorder.start(path.string());
    for (std::size_t index = 0; index < source.size(); ++index) {
        if (!recorder.record(source[index], 50'000 + index)) {
            fail("recorder rejected update-flag setup data");
        }
    }
    recorder.stop();

    {
        std::fstream ids(
            path.string() + ".qids",
            std::ios::binary | std::ios::in | std::ios::out);
        quant::UpdateIdFileHeader header{};
        ids.read(reinterpret_cast<char*>(&header), sizeof(header));
        header.flags = 1;
        ids.seekp(0);
        ids.write(reinterpret_cast<const char*>(&header), sizeof(header));
    }

    SPSCRingBuffer<65536> queue;
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> backpressure{0};
    std::atomic<std::uint64_t> errors{0};
    BinaryReplay replay(queue, replayed, backpressure, errors);
    std::atomic<bool> running{true};
    replay.run(running, path.string(), 0.0);
    if (errors.load() != 1 || replayed.load() != 0 ||
        replay.last_error().find("Update-ID file header is invalid") ==
            std::string::npos) {
        fail("unsupported update-ID header flags were not rejected");
    }
    remove_recording(path);
}

void corrupt_record_test() {
    const auto path = unique_path("corrupt");
    remove_recording(path);
    {
        std::ofstream output(path, std::ios::binary);
        const auto header = quant::make_binary_header();
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        const OrderBookState invalid{1, 101.0, 100.0, 1, 1};
        output.write(reinterpret_cast<const char*>(&invalid), sizeof(invalid));
    }

    SPSCRingBuffer<65536> queue;
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> backpressure{0};
    std::atomic<std::uint64_t> errors{0};
    BinaryReplay replay(queue, replayed, backpressure, errors);
    std::atomic<bool> running{true};
    replay.run(running, path.string(), 0.0);
    if (errors.load() != 1 || replayed.load() != 0 || replay.last_error().empty()) {
        fail("corrupt market record was not rejected clearly");
    }
    remove_recording(path);
}

}  // namespace

int main() {
    round_trip_test();
    overwrite_protection_test();
    interrupted_current_format_test();
    zero_update_id_test();
    metadata_finalization_failure_test();
    unsupported_market_header_flags_test();
    unsupported_update_header_flags_test();
    corrupt_record_test();
    std::cout << "binary_io_tests: PASS\n";
    return 0;
}

