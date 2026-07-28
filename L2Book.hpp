#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "L2Types.hpp"

namespace quant::l2 {

namespace detail {

inline void fnv_mix(std::uint64_t& hash, const void* data, std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= 1'099'511'628'211ULL;
    }
}

template <typename T>
inline void fnv_value(std::uint64_t& hash, const T& value) noexcept {
    fnv_mix(hash, &value, sizeof(value));
}

inline std::uint64_t hash_levels(
    std::uint64_t last_update_id,
    std::span<const Level> bids,
    std::span<const Level> asks) noexcept {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    fnv_value(hash, last_update_id);
    const auto bid_count = static_cast<std::uint64_t>(bids.size());
    const auto ask_count = static_cast<std::uint64_t>(asks.size());
    fnv_value(hash, bid_count);
    for (const auto& level : bids) {
        fnv_value(hash, level.price);
        fnv_value(hash, level.quantity);
    }
    fnv_value(hash, ask_count);
    for (const auto& level : asks) {
        fnv_value(hash, level.price);
        fnv_value(hash, level.quantity);
    }
    return hash;
}

}  // namespace detail

class MapOrderBook {
public:
    void clear() {
        bids_.clear();
        asks_.clear();
        last_update_id_ = 0;
    }

    void install_snapshot(const Snapshot& snapshot) {
        clear();
        if (snapshot.last_update_id == 0) {
            throw std::invalid_argument("Snapshot update ID must be positive.");
        }
        for (const auto& level : snapshot.bids) {
            update_bid(level);
        }
        for (const auto& level : snapshot.asks) {
            update_ask(level);
        }
        last_update_id_ = snapshot.last_update_id;
        validate_or_throw();
    }

    void apply(const DepthUpdate& update) {
        validate_update(update);
        for (const auto& level : update.bids) {
            update_bid(level);
        }
        for (const auto& level : update.asks) {
            update_ask(level);
        }
        last_update_id_ = update.final_update_id;
        validate_or_throw();
    }

    [[nodiscard]] std::uint64_t last_update_id() const noexcept {
        return last_update_id_;
    }

    [[nodiscard]] std::vector<Level> bids(std::size_t limit = 0) const {
        return copy_levels(bids_, limit);
    }

    [[nodiscard]] std::vector<Level> asks(std::size_t limit = 0) const {
        return copy_levels(asks_, limit);
    }

    [[nodiscard]] std::uint64_t state_hash() const {
        const auto bid_levels = bids();
        const auto ask_levels = asks();
        return detail::hash_levels(last_update_id_, bid_levels, ask_levels);
    }

    void validate_or_throw() const {
        if (bids_.empty() || asks_.empty()) {
            throw std::runtime_error("L2 book must remain two-sided.");
        }
        if (bids_.begin()->first >= asks_.begin()->first) {
            throw std::runtime_error("L2 order book is crossed or locked.");
        }
        for (const auto& [price, quantity] : bids_) {
            if (price <= 0 || quantity == 0) {
                throw std::runtime_error("Invalid active bid level.");
            }
        }
        for (const auto& [price, quantity] : asks_) {
            if (price <= 0 || quantity == 0) {
                throw std::runtime_error("Invalid active ask level.");
            }
        }
    }

private:
    std::map<std::int64_t, std::uint64_t, std::greater<>> bids_;
    std::map<std::int64_t, std::uint64_t, std::less<>> asks_;
    std::uint64_t last_update_id_{0};

    void update_bid(const Level& level) {
        validate_level(level);
        if (level.quantity == 0) {
            bids_.erase(level.price);
        } else {
            bids_[level.price] = level.quantity;
        }
    }

    void update_ask(const Level& level) {
        validate_level(level);
        if (level.quantity == 0) {
            asks_.erase(level.price);
        } else {
            asks_[level.price] = level.quantity;
        }
    }

    template <typename Map>
    static std::vector<Level> copy_levels(const Map& source, std::size_t limit) {
        const std::size_t count = limit == 0 ? source.size() : std::min(limit, source.size());
        std::vector<Level> result;
        result.reserve(count);
        for (const auto& [price, quantity] : source) {
            if (result.size() == count) {
                break;
            }
            result.push_back(Level{price, quantity});
        }
        return result;
    }
};

class FlatOrderBook {
public:
    void clear() noexcept {
        bids_.clear();
        asks_.clear();
        last_update_id_ = 0;
    }

    void reserve(std::size_t levels_per_side) {
        bids_.reserve(levels_per_side);
        asks_.reserve(levels_per_side);
    }

    void install_snapshot(const Snapshot& snapshot) {
        clear();
        if (snapshot.last_update_id == 0) {
            throw std::invalid_argument("Snapshot update ID must be positive.");
        }
        bids_ = normalize(snapshot.bids, true);
        asks_ = normalize(snapshot.asks, false);
        last_update_id_ = snapshot.last_update_id;
        validate_or_throw();
    }

    void apply(const DepthUpdate& update) {
        validate_update(update);
        for (const auto& level : update.bids) {
            update_level(bids_, level, true);
        }
        for (const auto& level : update.asks) {
            update_level(asks_, level, false);
        }
        last_update_id_ = update.final_update_id;
        validate_or_throw();
    }

    [[nodiscard]] std::uint64_t last_update_id() const noexcept {
        return last_update_id_;
    }

    [[nodiscard]] const std::vector<Level>& all_bids() const noexcept { return bids_; }
    [[nodiscard]] const std::vector<Level>& all_asks() const noexcept { return asks_; }

    [[nodiscard]] std::vector<Level> bids(std::size_t limit = 0) const {
        return prefix(bids_, limit);
    }

    [[nodiscard]] std::vector<Level> asks(std::size_t limit = 0) const {
        return prefix(asks_, limit);
    }

    [[nodiscard]] const Level& best_bid() const {
        if (bids_.empty()) {
            throw std::runtime_error("Bid book is empty.");
        }
        return bids_.front();
    }

    [[nodiscard]] const Level& best_ask() const {
        if (asks_.empty()) {
            throw std::runtime_error("Ask book is empty.");
        }
        return asks_.front();
    }

    [[nodiscard]] std::uint64_t quantity_at(bool bid_side, std::int64_t price) const noexcept {
        const auto& levels = bid_side ? bids_ : asks_;
        const auto iterator = find_level(levels, price, bid_side);
        return iterator != levels.end() && iterator->price == price
            ? iterator->quantity
            : 0;
    }

    [[nodiscard]] std::uint64_t state_hash() const noexcept {
        return detail::hash_levels(last_update_id_, bids_, asks_);
    }

    void validate_or_throw() const {
        if (bids_.empty() || asks_.empty()) {
            throw std::runtime_error("L2 book must remain two-sided.");
        }
        if (bids_.front().price >= asks_.front().price) {
            throw std::runtime_error("L2 order book is crossed or locked.");
        }
        validate_side(bids_, true);
        validate_side(asks_, false);
    }

private:
    std::vector<Level> bids_;
    std::vector<Level> asks_;
    std::uint64_t last_update_id_{0};

    static std::vector<Level>::iterator find_level(
        std::vector<Level>& levels,
        std::int64_t price,
        bool descending) noexcept {
        return std::lower_bound(
            levels.begin(), levels.end(), price,
            [descending](const Level& level, std::int64_t candidate) {
                return descending ? level.price > candidate : level.price < candidate;
            });
    }

    static std::vector<Level>::const_iterator find_level(
        const std::vector<Level>& levels,
        std::int64_t price,
        bool descending) noexcept {
        return std::lower_bound(
            levels.begin(), levels.end(), price,
            [descending](const Level& level, std::int64_t candidate) {
                return descending ? level.price > candidate : level.price < candidate;
            });
    }

    static void update_level(
        std::vector<Level>& levels,
        const Level& level,
        bool descending) {
        validate_level(level);
        auto iterator = find_level(levels, level.price, descending);
        const bool exists = iterator != levels.end() && iterator->price == level.price;
        if (level.quantity == 0) {
            if (exists) {
                levels.erase(iterator);
            }
            return;
        }
        if (exists) {
            iterator->quantity = level.quantity;
        } else {
            levels.insert(iterator, level);
        }
    }

    static std::vector<Level> normalize(
        const std::vector<Level>& input,
        bool descending) {
        std::vector<Level> result;
        result.reserve(input.size());
        for (const auto& level : input) {
            update_level(result, level, descending);
        }
        return result;
    }

    static std::vector<Level> prefix(
        const std::vector<Level>& levels,
        std::size_t limit) {
        const auto count = limit == 0 ? levels.size() : std::min(limit, levels.size());
        return std::vector<Level>(levels.begin(), levels.begin() + static_cast<std::ptrdiff_t>(count));
    }

    static void validate_side(const std::vector<Level>& levels, bool descending) {
        for (std::size_t index = 0; index < levels.size(); ++index) {
            const auto& level = levels[index];
            if (level.price <= 0 || level.quantity == 0) {
                throw std::runtime_error("Invalid active L2 level.");
            }
            if (index > 0) {
                const bool ordered = descending
                    ? levels[index - 1].price > level.price
                    : levels[index - 1].price < level.price;
                if (!ordered) {
                    throw std::runtime_error("L2 levels are not strictly ordered.");
                }
            }
        }
    }
};

}  // namespace quant::l2
