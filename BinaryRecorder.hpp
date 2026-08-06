
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "BinaryFormat.hpp"
#include "RingBuffer.hpp"

class BinaryRecorder {
public:
    BinaryRecorder() = default;
    BinaryRecorder(const BinaryRecorder&) = delete;
    BinaryRecorder& operator=(const BinaryRecorder&) = delete;

    ~BinaryRecorder() {
        stop();
    }

    void start(const std::string& file_path) {
        if (file_path.empty()) {
            throw std::invalid_argument("Recording file path cannot be empty.");
        }
        if (active_.load(std::memory_order_acquire) || worker_thread_.joinable()) {
            throw std::runtime_error("Binary recorder is already active.");
        }

        reset_counters();
        queue_.reset();
        {
            std::lock_guard lock(metadata_mutex_);
            markers_.clear();
            file_path_ = std::filesystem::path{file_path};
            update_id_path_ = std::filesystem::path{file_path + ".qids"};
            created_unix_ns_ = 0;
            clean_shutdown_ = false;
            feed_dropped_ticks_ = 0;
            malformed_messages_ = 0;
            reconnect_count_ = 0;
            first_update_id_.store(0, std::memory_order_relaxed);
            last_update_id_.store(0, std::memory_order_relaxed);
            metadata_pending_ = false;
        }

        if (file_path_.has_parent_path()) {
            std::error_code directory_error;
            std::filesystem::create_directories(
                file_path_.parent_path(), directory_error);
            if (directory_error) {
                throw std::runtime_error(
                    "Could not create recording directory: " +
                    directory_error.message());
            }
        }

        reject_existing_path(file_path_);
        reject_existing_path(update_id_path_);
        reject_existing_path(std::filesystem::path{file_path_.string() + ".meta.json"});

        output_.open(
            file_path_,
            std::ios::binary | std::ios::out | std::ios::trunc);
        if (!output_.is_open()) {
            throw std::runtime_error(
                "Could not open recording file: " + file_path);
        }

        update_id_output_.open(
            update_id_path_,
            std::ios::binary | std::ios::out | std::ios::trunc);
        if (!update_id_output_.is_open()) {
            output_.close();
            remove_failed_creation(file_path_);
            throw std::runtime_error(
                "Could not open update-ID file: " + update_id_path_.string());
        }

        const quant::BinaryFileHeader header = quant::make_binary_header();
        const quant::UpdateIdFileHeader update_header =
            quant::make_update_id_header(header.created_unix_ns);
        {
            std::lock_guard lock(metadata_mutex_);
            created_unix_ns_ = header.created_unix_ns;
        }

        output_.write(reinterpret_cast<const char*>(&header), sizeof(header));
        update_id_output_.write(
            reinterpret_cast<const char*>(&update_header),
            sizeof(update_header));
        if (!output_ || !update_id_output_) {
            output_.close();
            update_id_output_.close();
            remove_failed_creation(file_path_);
            remove_failed_creation(update_id_path_);
            throw std::runtime_error("Could not write recording headers.");
        }
        {
            std::lock_guard lock(metadata_mutex_);
            metadata_pending_ = true;
        }

        active_.store(true, std::memory_order_release);
        try {
            worker_thread_ = std::thread(&BinaryRecorder::writer_loop, this);
        } catch (...) {
            active_.store(false, std::memory_order_release);
            {
                std::lock_guard lock(metadata_mutex_);
                metadata_pending_ = false;
            }
            output_.close();
            update_id_output_.close();
            remove_failed_creation(file_path_);
            remove_failed_creation(update_id_path_);
            throw;
        }
    }

    bool record(
        const OrderBookState& state,
        std::uint64_t exchange_update_id = 0) noexcept {
        if (!active_.load(std::memory_order_acquire) ||
            !valid_order_book_state(state)) {
            return false;
        }

        const RecorderEntry entry{state, exchange_update_id};
        if (!queue_.push(entry)) {
            dropped_records_.fetch_add(1, std::memory_order_relaxed);
            add_marker(
                "recording_queue_drop",
                accepted_records_.load(std::memory_order_relaxed));
            return false;
        }

        accepted_records_.fetch_add(1, std::memory_order_relaxed);
        if (exchange_update_id != 0) {
            std::uint64_t expected = 0;
            first_update_id_.compare_exchange_strong(
                expected,
                exchange_update_id,
                std::memory_order_relaxed,
                std::memory_order_relaxed);
            last_update_id_.store(exchange_update_id, std::memory_order_relaxed);
        }
        return true;
    }

    void mark_session_boundary(const std::string& reason) noexcept {
        add_marker(reason, accepted_records_.load(std::memory_order_relaxed));
    }

    void set_feed_summary(
        std::uint64_t dropped_ticks,
        std::uint64_t malformed_messages,
        std::uint64_t reconnect_count) noexcept {
        std::lock_guard lock(metadata_mutex_);
        feed_dropped_ticks_ = dropped_ticks;
        malformed_messages_ = malformed_messages;
        reconnect_count_ = reconnect_count;
    }

    void stop() noexcept {
        active_.store(false, std::memory_order_release);
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        bool close_ok = true;
        close_ok = flush_and_close(output_) && close_ok;
        close_ok = flush_and_close(update_id_output_) && close_ok;
        if (!close_ok) {
            note_write_error_once();
        }

        bool should_write_metadata = false;
        {
            std::lock_guard lock(metadata_mutex_);
            if (metadata_pending_) {
                clean_shutdown_ = true;
                metadata_pending_ = false;
                should_write_metadata = true;
            }
        }
        if (should_write_metadata && !write_metadata_sidecar_noexcept()) {
            note_write_error_once();
        }
    }

    bool is_active() const noexcept {
        return active_.load(std::memory_order_acquire);
    }
    std::uint64_t recorded_records() const noexcept {
        return recorded_records_.load(std::memory_order_relaxed);
    }
    std::uint64_t accepted_records() const noexcept {
        return accepted_records_.load(std::memory_order_relaxed);
    }
    std::uint64_t dropped_records() const noexcept {
        return dropped_records_.load(std::memory_order_relaxed);
    }
    std::uint64_t write_errors() const noexcept {
        return write_errors_.load(std::memory_order_relaxed);
    }

    void reset_counters() noexcept {
        recorded_records_.store(0, std::memory_order_relaxed);
        accepted_records_.store(0, std::memory_order_relaxed);
        dropped_records_.store(0, std::memory_order_relaxed);
        write_errors_.store(0, std::memory_order_relaxed);
        write_error_reported_.clear(std::memory_order_relaxed);
    }

private:
    struct RecorderEntry {
        OrderBookState state;
        std::uint64_t exchange_update_id;
    };
    static_assert(std::is_trivially_copyable_v<RecorderEntry>);

    struct Marker {
        std::uint64_t record_index;
        std::string kind;
    };

    static constexpr std::size_t WriteBatchSize = 4096;

    SPSCRingBuffer<65536, RecorderEntry> queue_;
    std::ofstream output_;
    std::ofstream update_id_output_;
    std::thread worker_thread_;
    std::atomic<bool> active_{false};
    std::atomic<std::uint64_t> recorded_records_{0};
    std::atomic<std::uint64_t> accepted_records_{0};
    std::atomic<std::uint64_t> dropped_records_{0};
    std::atomic<std::uint64_t> write_errors_{0};
    std::atomic_flag write_error_reported_ = ATOMIC_FLAG_INIT;

    mutable std::mutex metadata_mutex_;
    std::filesystem::path file_path_;
    std::filesystem::path update_id_path_;
    std::uint64_t created_unix_ns_{0};
    bool clean_shutdown_{false};
    std::uint64_t feed_dropped_ticks_{0};
    std::uint64_t malformed_messages_{0};
    std::uint64_t reconnect_count_{0};
    std::atomic<std::uint64_t> first_update_id_{0};
    std::atomic<std::uint64_t> last_update_id_{0};
    std::vector<Marker> markers_;
    bool metadata_pending_{false};

    static void reject_existing_path(const std::filesystem::path& path) {
        std::error_code error;
        const bool exists = std::filesystem::exists(path, error);
        if (error) {
            throw std::runtime_error(
                "Could not inspect recording destination: " + error.message());
        }
        if (exists) {
            throw std::runtime_error(
                "Refusing to overwrite an existing recording artifact: " +
                path.string());
        }
    }

    static void remove_failed_creation(const std::filesystem::path& path) noexcept {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }

    static bool flush_and_close(std::ofstream& stream) noexcept {
        if (!stream.is_open()) {
            return true;
        }
        stream.flush();
        const bool success = static_cast<bool>(stream);
        stream.close();
        return success;
    }

    static std::string json_escape(const std::string& value) {
        std::string output;
        output.reserve(value.size());
        for (const char raw_character : value) {
            const auto character = static_cast<unsigned char>(raw_character);
            switch (character) {
                case '\\': output += "\\\\"; break;
                case '"': output += "\\\""; break;
                case '\b': output += "\\b"; break;
                case '\f': output += "\\f"; break;
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                default:
                    if (character < 0x20) {
                        std::ostringstream escaped;
                        escaped << "\\u" << std::hex << std::setw(4)
                                << std::setfill('0')
                                << static_cast<int>(character);
                        output += escaped.str();
                    } else {
                        output.push_back(static_cast<char>(character));
                    }
            }
        }
        return output;
    }

    void add_marker(
        const std::string& kind,
        std::uint64_t record_index) noexcept {
        try {
            std::lock_guard lock(metadata_mutex_);
            if (!markers_.empty() &&
                markers_.back().record_index == record_index &&
                markers_.back().kind == kind) {
                return;
            }
            markers_.push_back(Marker{record_index, kind});
        } catch (...) {
            // Supplementary metadata must never jeopardize ingestion.
        }
    }

    void note_write_error_once() noexcept {
        if (!write_error_reported_.test_and_set(std::memory_order_relaxed)) {
            write_errors_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void writer_loop() noexcept {
        std::array<RecorderEntry, WriteBatchSize> entries{};
        std::array<OrderBookState, WriteBatchSize> states{};
        std::array<std::uint64_t, WriteBatchSize> update_ids{};

        while (true) {
            const std::size_t count = queue_.consume_batch(
                entries.data(), entries.size());
            if (count != 0) {
                for (std::size_t index = 0; index < count; ++index) {
                    states[index] = entries[index].state;
                    update_ids[index] = entries[index].exchange_update_id;
                }
                output_.write(
                    reinterpret_cast<const char*>(states.data()),
                    static_cast<std::streamsize>(
                        count * sizeof(OrderBookState)));
                update_id_output_.write(
                    reinterpret_cast<const char*>(update_ids.data()),
                    static_cast<std::streamsize>(
                        count * sizeof(std::uint64_t)));
                if (!output_ || !update_id_output_) {
                    note_write_error_once();
                    active_.store(false, std::memory_order_release);
                    break;
                }
                recorded_records_.fetch_add(
                    static_cast<std::uint64_t>(count),
                    std::memory_order_relaxed);
                continue;
            }

            if (!active_.load(std::memory_order_acquire)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }

    bool write_metadata_sidecar_noexcept() noexcept {
        try {
            std::filesystem::path path;
            std::filesystem::path update_path;
            std::uint64_t created = 0;
            bool clean = false;
            std::uint64_t feed_drops = 0;
            std::uint64_t malformed = 0;
            std::uint64_t reconnects = 0;
            std::uint64_t first_update = 0;
            std::uint64_t last_update = 0;
            std::vector<Marker> markers;
            {
                std::lock_guard lock(metadata_mutex_);
                path = file_path_;
                update_path = update_id_path_;
                created = created_unix_ns_;
                clean = clean_shutdown_;
                feed_drops = feed_dropped_ticks_;
                malformed = malformed_messages_;
                reconnects = reconnect_count_;
                markers = markers_;
            }
            first_update = first_update_id_.load(std::memory_order_relaxed);
            last_update = last_update_id_.load(std::memory_order_relaxed);
            if (path.empty()) {
                return false;
            }

            const auto sidecar = std::filesystem::path{path.string() + ".meta.json"};
            const auto temporary = std::filesystem::path{sidecar.string() + ".tmp"};
            std::ofstream metadata{temporary, std::ios::out | std::ios::trunc};
            if (!metadata.is_open()) {
                return false;
            }

            const auto recorded = recorded_records();
            const auto accepted = accepted_records();
            const auto recorder_drops = dropped_records();
            const auto errors = write_errors();
            const bool data_complete = clean && recorder_drops == 0 &&
                                       errors == 0 && recorded == accepted;

            metadata << "{\n"
                     << "  \"schema_version\": 1,\n"
                     << "  \"binary_version\": " << quant::BinaryVersion << ",\n"
                     << "  \"record_size\": " << sizeof(OrderBookState) << ",\n"
                     << "  \"volume_scale\": " << quant::BinaryVolumeScale << ",\n"
                     << "  \"source\": \"binance_spot\",\n"
                     << "  \"symbol\": \"BTCUSDT\",\n"
                     << "  \"stream\": \"bookTicker\",\n"
                     << "  \"recording_file\": \""
                     << json_escape(path.filename().string()) << "\",\n"
                     << "  \"update_id_file\": \""
                     << json_escape(update_path.filename().string()) << "\",\n"
                     << "  \"update_id_version\": "
                     << quant::UpdateIdVersion << ",\n"
                     << "  \"created_unix_ns\": " << created << ",\n"
                     << "  \"clean_shutdown\": "
                     << (clean ? "true" : "false") << ",\n"
                     << "  \"data_complete\": "
                     << (data_complete ? "true" : "false") << ",\n"
                     << "  \"accepted_records\": " << accepted << ",\n"
                     << "  \"recorded_records\": " << recorded << ",\n"
                     << "  \"recorded_update_ids\": " << recorded << ",\n"
                     << "  \"first_update_id\": " << first_update << ",\n"
                     << "  \"last_update_id\": " << last_update << ",\n"
                     << "  \"recording_dropped\": " << recorder_drops << ",\n"
                     << "  \"recording_write_errors\": " << errors << ",\n"
                     << "  \"consumer_queue_dropped\": " << feed_drops << ",\n"
                     << "  \"malformed_messages\": " << malformed << ",\n"
                     << "  \"reconnect_count\": " << reconnects << ",\n"
                     << "  \"boundaries\": [\n";

            for (std::size_t index = 0; index < markers.size(); ++index) {
                metadata << "    {\"record_index\": "
                         << markers[index].record_index
                         << ", \"kind\": \""
                         << json_escape(markers[index].kind) << "\"}";
                if (index + 1 != markers.size()) {
                    metadata << ',';
                }
                metadata << '\n';
            }
            metadata << "  ]\n}\n";
            metadata.flush();
            if (!metadata) {
                metadata.close();
                remove_failed_creation(temporary);
                return false;
            }
            metadata.close();

            std::error_code rename_error;
            std::filesystem::rename(temporary, sidecar, rename_error);
            if (rename_error) {
                remove_failed_creation(temporary);
                return false;
            }
            return true;
        } catch (...) {
            // Sidecar absence is observable and stop() must remain noexcept.
            return false;
        }
    }
};

