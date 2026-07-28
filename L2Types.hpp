#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace quant::l2 {

inline constexpr std::int64_t PriceScale = 100'000'000LL;
inline constexpr std::uint64_t QuantityScale = 100'000'000ULL;
inline constexpr int ScaleDecimals = 8;

struct Level {
    std::int64_t price{0};
    std::uint64_t quantity{0};

    friend bool operator==(const Level&, const Level&) = default;
};

struct Snapshot {
    std::uint64_t receipt_timestamp_ns{0};
    std::uint64_t last_update_id{0};
    std::vector<Level> bids;
    std::vector<Level> asks;
};

struct DepthUpdate {
    std::uint64_t receipt_timestamp_ns{0};
    std::uint64_t event_time_ms{0};
    std::uint64_t first_update_id{0};
    std::uint64_t final_update_id{0};
    std::vector<Level> bids;
    std::vector<Level> asks;
};

struct Trade {
    std::uint64_t receipt_timestamp_ns{0};
    std::uint64_t event_time_ms{0};
    std::uint64_t aggregate_trade_id{0};
    std::int64_t price{0};
    std::uint64_t quantity{0};
    bool buyer_is_maker{false};
};

inline bool parse_fixed_decimal(
    std::string_view text,
    std::uint64_t scale,
    std::uint64_t& output) noexcept {
    if (text.empty() || scale == 0) {
        return false;
    }
    if (text.front() == '+') {
        text.remove_prefix(1);
    }
    if (text.empty() || text.front() == '-') {
        return false;
    }

    const auto dot = text.find('.');
    const auto integer_part = text.substr(0, dot);
    const auto fractional_part = dot == std::string_view::npos
        ? std::string_view{}
        : text.substr(dot + 1);
    if (integer_part.empty() && fractional_part.empty()) {
        return false;
    }

    std::uint64_t integer = 0;
    if (!integer_part.empty()) {
        const auto parsed = std::from_chars(
            integer_part.data(), integer_part.data() + integer_part.size(), integer);
        if (parsed.ec != std::errc{} ||
            parsed.ptr != integer_part.data() + integer_part.size()) {
            return false;
        }
    }

    int decimals = 0;
    std::uint64_t scale_copy = scale;
    while (scale_copy > 1 && scale_copy % 10 == 0) {
        ++decimals;
        scale_copy /= 10;
    }
    if (scale_copy != 1) {
        return false;
    }

    std::uint64_t fraction = 0;
    int consumed = 0;
    for (char character : fractional_part) {
        if (character < '0' || character > '9') {
            return false;
        }
        if (consumed < decimals) {
            fraction = fraction * 10 + static_cast<std::uint64_t>(character - '0');
            ++consumed;
        } else if (character != '0') {
            return false;
        }
    }
    while (consumed < decimals) {
        fraction *= 10;
        ++consumed;
    }

    if (integer > (std::numeric_limits<std::uint64_t>::max() - fraction) / scale) {
        return false;
    }
    output = integer * scale + fraction;
    return true;
}

inline bool parse_price(std::string_view text, std::int64_t& output) noexcept {
    std::uint64_t raw = 0;
    if (!parse_fixed_decimal(text, static_cast<std::uint64_t>(PriceScale), raw) ||
        raw > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return false;
    }
    output = static_cast<std::int64_t>(raw);
    return output > 0;
}

inline bool parse_quantity(std::string_view text, std::uint64_t& output) noexcept {
    return parse_fixed_decimal(text, QuantityScale, output);
}

inline double price_to_double(std::int64_t value) noexcept {
    return static_cast<double>(value) / static_cast<double>(PriceScale);
}

inline double quantity_to_double(std::uint64_t value) noexcept {
    return static_cast<double>(value) / static_cast<double>(QuantityScale);
}

inline void validate_level(const Level& level, bool allow_zero_quantity = true) {
    if (level.price <= 0) {
        throw std::invalid_argument("L2 price must be positive.");
    }
    if (!allow_zero_quantity && level.quantity == 0) {
        throw std::invalid_argument("Active L2 quantity must be positive.");
    }
}

inline void validate_update(const DepthUpdate& update) {
    if (update.first_update_id == 0 ||
        update.final_update_id == 0 ||
        update.first_update_id > update.final_update_id) {
        throw std::invalid_argument("Invalid L2 update-ID range.");
    }
    for (const auto& level : update.bids) {
        validate_level(level);
    }
    for (const auto& level : update.asks) {
        validate_level(level);
    }
}

}  // namespace quant::l2
