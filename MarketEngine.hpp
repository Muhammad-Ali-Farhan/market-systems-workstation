
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <shared_mutex>
#include <string>
#include <thread>

#include "BinaryRecorder.hpp"
#include "BinaryReplay.hpp"
#include "BinanceFeed.hpp"
#include "RingBuffer.hpp"

class IngestionEngine {
public:
    enum class State : std::uint8_t {
        stopped,
        starting,
        live,
        replaying,
        stopping,
        completed,
        failed,
    };

    IngestionEngine() = default;
    IngestionEngine(const IngestionEngine&) = delete;
    IngestionEngine& operator=(const IngestionEngine&) = delete;

    ~IngestionEngine() {
        stop();
    }

    void start() {
        start_live("");
    }

    void start_live(const std::string& recording_path) {
        std::lock_guard lock(lifecycle_mutex_);
        prepare_start_locked();
        state_.store(State::starting, std::memory_order_release);

        try {
            BinaryRecorder* recorder_pointer = nullptr;
            if (!recording_path.empty()) {
                recorder_.start(recording_path);
                recorder_pointer = &recorder_;
            }

            feed_ = std::make_unique<BinanceFeed>(
                ring_buffer_,
                dropped_ticks_,
                malformed_messages_,
                reconnect_count_,
                recorder_pointer);

            state_.store(State::live, std::memory_order_release);
            worker_thread_ = std::thread(&IngestionEngine::run_live, this);
        } catch (...) {
            running_.store(false, std::memory_order_release);
            feed_.reset();
            recorder_.stop();
            state_.store(State::failed, std::memory_order_release);
            throw;
        }
    }

    void start_replay(const std::string& file_path, double speed) {
        if (file_path.empty()) {
            throw std::invalid_argument("Replay file path cannot be empty.");
        }
        if (!std::isfinite(speed) || speed < 0.0) {
            throw std::invalid_argument(
                "Replay speed must be finite and non-negative.");
        }

        std::lock_guard lock(lifecycle_mutex_);
        prepare_start_locked();
        state_.store(State::starting, std::memory_order_release);
        replay_file_path_ = file_path;
        replay_speed_ = speed;

        try {
            replay_ = std::make_unique<BinaryReplay>(
                ring_buffer_,
                replayed_records_,
                replay_backpressure_events_,
                replay_errors_);
            state_.store(State::replaying, std::memory_order_release);
            worker_thread_ = std::thread(&IngestionEngine::run_replay, this);
        } catch (...) {
            running_.store(false, std::memory_order_release);
            replay_.reset();
            state_.store(State::failed, std::memory_order_release);
            throw;
        }
    }

    void stop() noexcept {
        std::lock_guard lock(lifecycle_mutex_);
        const State previous = state_.load(std::memory_order_acquire);
        if (previous != State::stopped) {
            state_.store(State::stopping, std::memory_order_release);
        }
        running_.store(false, std::memory_order_release);

        if (feed_ != nullptr) {
            feed_->request_stop();
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        recorder_.set_feed_summary(
            dropped_ticks_.load(std::memory_order_relaxed),
            malformed_messages_.load(std::memory_order_relaxed),
            reconnect_count_.load(std::memory_order_relaxed));
        recorder_.stop();

        if (feed_ != nullptr) {
            const std::string feed_error = feed_->last_error();
            if (!feed_error.empty()) {
                set_last_error(feed_error);
            }
        }
        feed_.reset();
        replay_.reset();
        state_.store(State::stopped, std::memory_order_release);
    }

    std::size_t consume_batch(
        OrderBookState* destination,
        std::size_t maximum_count) {
        std::shared_lock lifecycle_read_lock(consumer_lifecycle_mutex_);
        if (maximum_count == 0) {
            return 0;
        }
        if (destination == nullptr) {
            throw std::invalid_argument("Destination pointer cannot be null.");
        }
        if (consumer_active_.test_and_set(std::memory_order_acquire)) {
            throw std::runtime_error(
                "consume_batch supports exactly one concurrent consumer.");
        }

        struct ConsumerGuard {
            std::atomic_flag& flag;
            ~ConsumerGuard() { flag.clear(std::memory_order_release); }
        } guard{consumer_active_};

        return ring_buffer_.consume_batch(destination, maximum_count);
    }

    bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    std::string state_name() const {
        switch (state_.load(std::memory_order_acquire)) {
            case State::stopped: return "stopped";
            case State::starting: return "starting";
            case State::live: return "live";
            case State::replaying: return "replaying";
            case State::stopping: return "stopping";
            case State::completed: return "completed";
            case State::failed: return "failed";
        }
        return "unknown";
    }

    std::string last_error() const {
        std::lock_guard lifecycle_lock(lifecycle_mutex_);
        {
            std::lock_guard error_lock(error_mutex_);
            if (!last_error_.empty()) {
                return last_error_;
            }
        }
        return feed_ != nullptr ? feed_->last_error() : std::string{};
    }

    std::uint64_t now_ns() const noexcept {
        const auto now = std::chrono::steady_clock::now();
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count());
    }

    std::uint64_t dropped_ticks() const noexcept {
        return dropped_ticks_.load(std::memory_order_relaxed);
    }
    std::uint64_t malformed_messages() const noexcept {
        return malformed_messages_.load(std::memory_order_relaxed);
    }
    std::uint64_t reconnect_count() const noexcept {
        return reconnect_count_.load(std::memory_order_relaxed);
    }
    std::uint64_t recorded_ticks() const noexcept {
        return recorder_.recorded_records();
    }
    std::uint64_t recording_accepted() const noexcept {
        return recorder_.accepted_records();
    }
    std::uint64_t recording_dropped() const noexcept {
        return recorder_.dropped_records();
    }
    std::uint64_t recording_write_errors() const noexcept {
        return recorder_.write_errors();
    }
    std::uint64_t replayed_ticks() const noexcept {
        return replayed_records_.load(std::memory_order_relaxed);
    }
    std::uint64_t replay_backpressure_events() const noexcept {
        return replay_backpressure_events_.load(std::memory_order_relaxed);
    }
    std::uint64_t replay_errors() const noexcept {
        return replay_errors_.load(std::memory_order_relaxed);
    }

private:
    SPSCRingBuffer<65536> ring_buffer_;
    BinaryRecorder recorder_;
    std::unique_ptr<BinanceFeed> feed_;
    std::unique_ptr<BinaryReplay> replay_;
    std::thread worker_thread_;
    std::string replay_file_path_;
    double replay_speed_{1.0};

    mutable std::mutex lifecycle_mutex_;
    mutable std::shared_mutex consumer_lifecycle_mutex_;
    mutable std::mutex error_mutex_;
    std::string last_error_;
    std::atomic<State> state_{State::stopped};
    std::atomic<bool> running_{false};
    std::atomic_flag consumer_active_ = ATOMIC_FLAG_INIT;

    std::atomic<std::uint64_t> dropped_ticks_{0};
    std::atomic<std::uint64_t> malformed_messages_{0};
    std::atomic<std::uint64_t> reconnect_count_{0};
    std::atomic<std::uint64_t> replayed_records_{0};
    std::atomic<std::uint64_t> replay_backpressure_events_{0};
    std::atomic<std::uint64_t> replay_errors_{0};

    void set_last_error(const std::string& message) noexcept {
        try {
            std::lock_guard lock(error_mutex_);
            last_error_ = message;
        } catch (...) {
        }
    }

    void prepare_start_locked() {
        std::unique_lock consumer_reset_lock(consumer_lifecycle_mutex_);
        if (running_.load(std::memory_order_acquire)) {
            throw std::runtime_error("Ingestion engine is already running.");
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        feed_.reset();
        replay_.reset();
        recorder_.stop();

        ring_buffer_.reset();
        dropped_ticks_.store(0, std::memory_order_relaxed);
        malformed_messages_.store(0, std::memory_order_relaxed);
        reconnect_count_.store(0, std::memory_order_relaxed);
        replayed_records_.store(0, std::memory_order_relaxed);
        replay_backpressure_events_.store(0, std::memory_order_relaxed);
        replay_errors_.store(0, std::memory_order_relaxed);
        recorder_.reset_counters();
        consumer_active_.clear(std::memory_order_relaxed);
        {
            std::lock_guard error_lock(error_mutex_);
            last_error_.clear();
        }
        running_.store(true, std::memory_order_release);
    }

    void run_live() noexcept {
        try {
            if (feed_ != nullptr) {
                feed_->run(running_);
            }
            if (state_.load(std::memory_order_acquire) != State::stopping) {
                state_.store(State::completed, std::memory_order_release);
            }
        } catch (const std::exception& exception) {
            set_last_error(exception.what());
            state_.store(State::failed, std::memory_order_release);
        } catch (...) {
            set_last_error("Unknown live-feed failure.");
            state_.store(State::failed, std::memory_order_release);
        }
        running_.store(false, std::memory_order_release);
    }

    void run_replay() noexcept {
        try {
            if (replay_ != nullptr) {
                replay_->run(running_, replay_file_path_, replay_speed_);
            }
            if (replay_errors_.load(std::memory_order_relaxed) != 0) {
                const std::string message = replay_ != nullptr
                    ? replay_->last_error()
                    : std::string{};
                set_last_error(message.empty()
                    ? "Binary replay failed; inspect stderr for details."
                    : message);
                state_.store(State::failed, std::memory_order_release);
            } else if (state_.load(std::memory_order_acquire) != State::stopping) {
                state_.store(State::completed, std::memory_order_release);
            }
        } catch (const std::exception& exception) {
            replay_errors_.fetch_add(1, std::memory_order_relaxed);
            set_last_error(exception.what());
            state_.store(State::failed, std::memory_order_release);
        } catch (...) {
            replay_errors_.fetch_add(1, std::memory_order_relaxed);
            set_last_error("Unknown replay failure.");
            state_.store(State::failed, std::memory_order_release);
        }
        running_.store(false, std::memory_order_release);
    }
};

