#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string_view>

#include "L2Book.hpp"

namespace quant::l2 {

enum class SyncState : std::uint8_t {
    awaiting_snapshot,
    live,
    gap,
};

enum class ApplyResult : std::uint8_t {
    applied,
    ignored_stale,
    buffered,
    gap_detected,
};

enum class SnapshotResult : std::uint8_t {
    synchronized,
    awaiting_bridge,
    snapshot_too_old,
    gap_detected,
};

struct SnapshotInstallResult {
    SnapshotResult result{SnapshotResult::awaiting_bridge};
    std::size_t stale_events{0};
    std::size_t applied_events{0};
    std::size_t first_applied_buffer_index{0};
};

class Synchronizer {
public:
    explicit Synchronizer(std::size_t maximum_buffered_events = 200'000)
        : maximum_buffered_events_(maximum_buffered_events) {
        if (maximum_buffered_events_ == 0) {
            throw std::invalid_argument("Maximum L2 buffer size must be positive.");
        }
        book_.reserve(5'000);
    }

    void reset() noexcept {
        state_ = SyncState::awaiting_snapshot;
        book_.clear();
        buffered_.clear();
        ++reset_count_;
    }

    ApplyResult ingest(const DepthUpdate& update) {
        validate_update(update);
        if (state_ != SyncState::live) {
            buffer(update);
            return ApplyResult::buffered;
        }
        return apply_live(update);
    }

    SnapshotInstallResult install_snapshot(const Snapshot& snapshot) {
        if (snapshot.last_update_id == 0) {
            throw std::invalid_argument("Snapshot update ID must be positive.");
        }

        SnapshotInstallResult result{};
        while (!buffered_.empty() &&
               buffered_.front().final_update_id <= snapshot.last_update_id) {
            buffered_.pop_front();
            ++result.stale_events;
        }

        if (buffered_.empty()) {
            book_.install_snapshot(snapshot);
            state_ = SyncState::awaiting_snapshot;
            result.result = SnapshotResult::awaiting_bridge;
            return result;
        }

        const auto expected = snapshot.last_update_id + 1;
        const auto& first = buffered_.front();
        if (first.first_update_id > expected) {
            result.result = SnapshotResult::snapshot_too_old;
            return result;
        }
        if (first.final_update_id < expected) {
            throw std::logic_error("Stale L2 events were not fully removed.");
        }

        book_.install_snapshot(snapshot);
        result.first_applied_buffer_index = result.stale_events;
        state_ = SyncState::live;

        while (!buffered_.empty()) {
            auto event = std::move(buffered_.front());
            buffered_.pop_front();
            const auto applied = apply_live(event);
            if (applied == ApplyResult::gap_detected) {
                result.result = SnapshotResult::gap_detected;
                return result;
            }
            if (applied == ApplyResult::applied) {
                ++result.applied_events;
            }
        }
        result.result = SnapshotResult::synchronized;
        return result;
    }

    [[nodiscard]] SyncState state() const noexcept { return state_; }
    [[nodiscard]] const FlatOrderBook& book() const noexcept { return book_; }
    [[nodiscard]] FlatOrderBook& book() noexcept { return book_; }
    [[nodiscard]] std::size_t buffered_events() const noexcept { return buffered_.size(); }
    [[nodiscard]] std::uint64_t reset_count() const noexcept { return reset_count_; }

private:
    SyncState state_{SyncState::awaiting_snapshot};
    FlatOrderBook book_;
    std::deque<DepthUpdate> buffered_;
    std::size_t maximum_buffered_events_;
    std::uint64_t reset_count_{0};

    void buffer(const DepthUpdate& update) {
        if (buffered_.size() >= maximum_buffered_events_) {
            state_ = SyncState::gap;
            throw std::runtime_error("L2 synchronization buffer capacity exceeded.");
        }
        if (!buffered_.empty() &&
            update.final_update_id < buffered_.back().final_update_id) {
            state_ = SyncState::gap;
            throw std::runtime_error("L2 depth events arrived out of order.");
        }
        buffered_.push_back(update);
    }

    ApplyResult apply_live(const DepthUpdate& update) {
        const auto local = book_.last_update_id();
        if (update.final_update_id <= local) {
            return ApplyResult::ignored_stale;
        }
        if (update.first_update_id > local + 1) {
            state_ = SyncState::gap;
            buffer(update);
            return ApplyResult::gap_detected;
        }
        book_.apply(update);
        state_ = SyncState::live;
        return ApplyResult::applied;
    }
};

inline std::string_view to_string(SyncState value) noexcept {
    switch (value) {
        case SyncState::awaiting_snapshot: return "awaiting_snapshot";
        case SyncState::live: return "live";
        case SyncState::gap: return "gap";
    }
    return "unknown";
}

inline std::string_view to_string(ApplyResult value) noexcept {
    switch (value) {
        case ApplyResult::applied: return "applied";
        case ApplyResult::ignored_stale: return "ignored_stale";
        case ApplyResult::buffered: return "buffered";
        case ApplyResult::gap_detected: return "gap_detected";
    }
    return "unknown";
}

inline std::string_view to_string(SnapshotResult value) noexcept {
    switch (value) {
        case SnapshotResult::synchronized: return "synchronized";
        case SnapshotResult::awaiting_bridge: return "awaiting_bridge";
        case SnapshotResult::snapshot_too_old: return "snapshot_too_old";
        case SnapshotResult::gap_detected: return "gap_detected";
    }
    return "unknown";
}

}  // namespace quant::l2
