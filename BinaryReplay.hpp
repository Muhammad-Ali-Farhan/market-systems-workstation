
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <mutex>
#include <string>
#include <thread>

#include "BinaryFormat.hpp"
#include "RingBuffer.hpp"

class BinaryReplay {
public:
    BinaryReplay(
        SPSCRingBuffer<65536>& ring_buffer,
        std::atomic<std::uint64_t>& replayed_records,
        std::atomic<std::uint64_t>& backpressure_events,
        std::atomic<std::uint64_t>& replay_errors)
        : ring_buffer_(ring_buffer),
          replayed_records_(replayed_records),
          backpressure_events_(backpressure_events),
          replay_errors_(replay_errors) {}

    void run(
        std::atomic<bool>& running,
        const std::string& file_path,
        double speed) noexcept {
        try {
            run_session(running, file_path, speed);
        } catch (const std::exception& exception) {
            replay_errors_.fetch_add(1, std::memory_order_relaxed);
            set_last_error(exception.what());
            std::cerr << "[Binary Replay] Error: " << exception.what() << '\n';
        } catch (...) {
            replay_errors_.fetch_add(1, std::memory_order_relaxed);
            set_last_error("Unknown replay error.");
            std::cerr << "[Binary Replay] Unknown error.\n";
        }
        running.store(false, std::memory_order_release);
    }

    std::string last_error() const {
        std::lock_guard lock(error_mutex_);
        return last_error_;
    }

private:
    SPSCRingBuffer<65536>& ring_buffer_;
    std::atomic<std::uint64_t>& replayed_records_;
    std::atomic<std::uint64_t>& backpressure_events_;
    std::atomic<std::uint64_t>& replay_errors_;
    mutable std::mutex error_mutex_;
    std::string last_error_;

    void set_last_error(const std::string& message) noexcept {
        try {
            std::lock_guard lock(error_mutex_);
            last_error_ = message;
        } catch (...) {
        }
    }

    static bool read_record(std::ifstream& input, OrderBookState& record) {
        input.read(reinterpret_cast<char*>(&record), sizeof(record));
        const std::streamsize bytes_read = input.gcount();

        if (bytes_read == 0 && input.eof()) {
            return false;
        }
        if (bytes_read != static_cast<std::streamsize>(sizeof(record))) {
            throw std::runtime_error("Recording contains a truncated record.");
        }
        if (!valid_order_book_state(record)) {
            throw std::runtime_error("Recording contains an invalid market record.");
        }
        return true;
    }

    static bool read_update_id(
        std::ifstream& input,
        std::uint64_t& update_id) {
        input.read(reinterpret_cast<char*>(&update_id), sizeof(update_id));
        if (input.gcount() != static_cast<std::streamsize>(sizeof(update_id))) {
            throw std::runtime_error(
                "Update-ID file is truncated or misaligned.");
        }
        return true;
    }

    static bool open_update_id_file(
        const std::filesystem::path& recording_path,
        const quant::BinaryFileHeader& market_header,
        std::uint64_t record_count,
        std::ifstream& input) {
        const std::filesystem::path update_path{
            recording_path.string() + ".qids"};
        std::error_code existence_error;
        const bool exists = std::filesystem::exists(update_path, existence_error);
        if (existence_error) {
            throw std::runtime_error(
                "Could not inspect update-ID file: " +
                existence_error.message());
        }
        if (!exists) {
            return false;
        }

        const std::filesystem::path sidecar_path{
            recording_path.string() + ".meta.json"};
        std::error_code sidecar_error;
        const bool sidecar_exists = std::filesystem::exists(
            sidecar_path, sidecar_error);
        if (sidecar_error) {
            throw std::runtime_error(
                "Could not inspect recording completion sidecar: " +
                sidecar_error.message());
        }
        if (!sidecar_exists) {
            throw std::runtime_error(
                "Current-format recording has no completion sidecar and may be interrupted.");
        }

        std::error_code size_error;
        const auto file_size = std::filesystem::file_size(update_path, size_error);
        if (size_error || file_size < sizeof(quant::UpdateIdFileHeader)) {
            throw std::runtime_error("Update-ID file is incomplete.");
        }
        const auto payload_size = file_size - sizeof(quant::UpdateIdFileHeader);
        if (payload_size % sizeof(std::uint64_t) != 0 ||
            payload_size / sizeof(std::uint64_t) != record_count) {
            throw std::runtime_error(
                "Update-ID count does not match the market recording.");
        }

        input.open(update_path, std::ios::binary | std::ios::in);
        if (!input.is_open()) {
            throw std::runtime_error("Could not open update-ID file.");
        }
        quant::UpdateIdFileHeader update_header{};
        input.read(
            reinterpret_cast<char*>(&update_header),
            sizeof(update_header));
        if (input.gcount() !=
                static_cast<std::streamsize>(sizeof(update_header)) ||
            !quant::valid_update_id_header(
                update_header, market_header.created_unix_ns)) {
            throw std::runtime_error("Update-ID file header is invalid.");
        }
        return true;
    }

    static bool wait_until(
        const std::atomic<bool>& running,
        std::chrono::steady_clock::time_point target) noexcept {
        using namespace std::chrono_literals;

        while (running.load(std::memory_order_acquire)) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= target) {
                return true;
            }

            const auto remaining = target - now;
            if (remaining > 10ms) {
                std::this_thread::sleep_for(std::min(remaining - 5ms, std::chrono::steady_clock::duration{10ms}));
            } else if (remaining > 1ms) {
                std::this_thread::sleep_for(remaining / 2);
            } else {
                std::this_thread::yield();
            }
        }
        return false;
    }

    static std::chrono::nanoseconds scaled_replay_delta(
        std::uint64_t source_delta_ns,
        double speed) {
        const long double scaled =
            static_cast<long double>(source_delta_ns) /
            static_cast<long double>(speed);
        const long double maximum = static_cast<long double>(
            std::numeric_limits<std::int64_t>::max());

        if (!std::isfinite(scaled) || scaled < 0.0L || scaled > maximum) {
            throw std::runtime_error(
                "Replay duration is outside the supported nanosecond range.");
        }
        return std::chrono::nanoseconds{static_cast<std::int64_t>(scaled)};
    }

    void run_session(
        std::atomic<bool>& running,
        const std::string& file_path,
        double speed) {
        if (file_path.empty()) {
            throw std::invalid_argument("Replay file path cannot be empty.");
        }
        if (!std::isfinite(speed) || speed < 0.0) {
            throw std::invalid_argument(
                "Replay speed must be finite and non-negative.");
        }

        const std::filesystem::path path{file_path};
        std::error_code size_error;
        const auto file_size = std::filesystem::file_size(path, size_error);
        if (size_error) {
            throw std::runtime_error(
                "Could not stat replay file: " + size_error.message());
        }
        if (file_size < sizeof(quant::BinaryFileHeader)) {
            throw std::runtime_error("Replay file is smaller than its header.");
        }
        const auto payload_size = file_size - sizeof(quant::BinaryFileHeader);
        if (payload_size % sizeof(OrderBookState) != 0) {
            throw std::runtime_error("Replay file ends with a partial record.");
        }

        std::ifstream input{path, std::ios::binary | std::ios::in};
        if (!input.is_open()) {
            throw std::runtime_error("Could not open replay file: " + file_path);
        }

        quant::BinaryFileHeader header{};
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) {
            throw std::runtime_error(
                "Replay file does not contain a complete header.");
        }
        if (!quant::valid_binary_header(header)) {
            throw std::runtime_error(
                "Replay file header is invalid or unsupported.");
        }

        const std::uint64_t record_count = static_cast<std::uint64_t>(
            payload_size / sizeof(OrderBookState));
        std::ifstream update_id_input;
        const bool has_update_ids = open_update_id_file(
            path, header, record_count, update_id_input);

        OrderBookState source_record{};
        if (!read_record(input, source_record)) {
            return;
        }
        std::uint64_t source_update_id = 0;
        if (has_update_ids) {
            read_update_id(update_id_input, source_update_id);
        }

        const std::uint64_t first_source_timestamp = source_record.timestamp_ns;
        std::uint64_t previous_source_timestamp = first_source_timestamp;
        std::optional<std::uint64_t> previous_update_id;
        const auto replay_start = std::chrono::steady_clock::now();

        while (running.load(std::memory_order_acquire)) {
            if (source_record.timestamp_ns < previous_source_timestamp) {
                throw std::runtime_error(
                    "Recording timestamps are not monotonic.");
            }
            if (has_update_ids) {
                if (source_update_id == 0) {
                    throw std::runtime_error(
                        "Current-format recording contains a zero exchange update ID.");
                }
                if (previous_update_id.has_value() &&
                    source_update_id <= previous_update_id.value()) {
                    throw std::runtime_error(
                        "Exchange update IDs are not strictly increasing.");
                }
                previous_update_id = source_update_id;
            }

            if (speed > 0.0) {
                const auto replay_delta = scaled_replay_delta(
                    source_record.timestamp_ns - first_source_timestamp,
                    speed);
                if (!wait_until(running, replay_start + replay_delta)) {
                    break;
                }
            }

            bool pushed = false;
            bool counted_backpressure = false;
            while (running.load(std::memory_order_acquire)) {
                // Preserve every source field exactly. Replay timing is controlled by
                // the scheduler above and must not alter research timestamps.
                if (ring_buffer_.push(source_record)) {
                    pushed = true;
                    break;
                }
                if (!counted_backpressure) {
                    backpressure_events_.fetch_add(1, std::memory_order_relaxed);
                    counted_backpressure = true;
                }
                std::this_thread::yield();
            }

            if (!pushed) {
                break;
            }

            replayed_records_.fetch_add(1, std::memory_order_relaxed);
            previous_source_timestamp = source_record.timestamp_ns;
            if (!read_record(input, source_record)) {
                break;
            }
            if (has_update_ids) {
                read_update_id(update_id_input, source_update_id);
            }
        }
    }
};

