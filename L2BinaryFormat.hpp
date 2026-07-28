#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>

#include "L2Types.hpp"

namespace quant::l2::binary {

inline constexpr std::array<char, 8> Magic{'Q', 'L', '2', 'E', 'V', 'T', '1', '\0'};
inline constexpr std::uint32_t Version = 1;
inline constexpr std::size_t MaximumLevelsPerSide = 20'000;

enum class EventType : std::uint32_t {
    snapshot = 1,
    depth = 2,
    trade = 3,
    boundary = 4,
};

struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t event_header_size;
    std::uint32_t level_size;
    std::uint64_t price_scale;
    std::uint64_t quantity_scale;
    std::uint64_t created_unix_ns;
    char symbol[16];
    std::uint64_t reserved[8];
};

struct EventHeader {
    std::uint32_t type;
    std::uint32_t flags;
    std::uint64_t receipt_timestamp_ns;
    std::uint64_t exchange_time_ms;
    std::uint64_t first_id;
    std::uint64_t final_id;
    std::uint32_t bid_count;
    std::uint32_t ask_count;
    std::int64_t trade_price;
    std::uint64_t trade_quantity;
    std::uint32_t payload_crc32;
    std::uint32_t record_size;
    std::uint64_t reserved;
};

struct LevelRecord {
    std::int64_t price;
    std::uint64_t quantity;
};

static_assert(std::endian::native == std::endian::little,
              "L2 binary format currently requires little-endian hosts.");
static_assert(sizeof(FileHeader) == 128);
static_assert(sizeof(EventHeader) == 80);
static_assert(sizeof(LevelRecord) == 16);

inline std::uint32_t crc32(std::span<const std::byte> bytes) noexcept {
    std::uint32_t value = 0xFFFF'FFFFU;
    for (const auto byte : bytes) {
        value ^= static_cast<std::uint32_t>(std::to_integer<unsigned char>(byte));
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = -(value & 1U);
            value = (value >> 1U) ^ (0xEDB8'8320U & mask);
        }
    }
    return ~value;
}

inline bool validate_header(const FileHeader& header) noexcept {
    if (std::memcmp(header.magic, Magic.data(), Magic.size()) != 0 ||
        header.version != Version ||
        header.header_size != sizeof(FileHeader) ||
        header.event_header_size != sizeof(EventHeader) ||
        header.level_size != sizeof(LevelRecord) ||
        header.price_scale != static_cast<std::uint64_t>(PriceScale) ||
        header.quantity_scale != QuantityScale ||
        header.created_unix_ns == 0 ||
        header.symbol[0] == '\0') {
        return false;
    }
    for (const auto value : header.reserved) {
        if (value != 0) {
            return false;
        }
    }
    return true;
}

inline bool validate_buffer(std::span<const std::byte> bytes) noexcept {
    if (bytes.size() < sizeof(FileHeader)) {
        return false;
    }
    FileHeader file_header{};
    std::memcpy(&file_header, bytes.data(), sizeof(file_header));
    if (!validate_header(file_header)) {
        return false;
    }

    std::size_t offset = sizeof(FileHeader);
    while (offset < bytes.size()) {
        if (bytes.size() - offset < sizeof(EventHeader)) {
            return false;
        }
        EventHeader header{};
        std::memcpy(&header, bytes.data() + offset, sizeof(header));
        if (header.reserved != 0 ||
            header.receipt_timestamp_ns == 0 ||
            header.bid_count > MaximumLevelsPerSide ||
            header.ask_count > MaximumLevelsPerSide) {
            return false;
        }
        const std::size_t level_count =
            static_cast<std::size_t>(header.bid_count) + header.ask_count;
        if (level_count > 2 * MaximumLevelsPerSide ||
            level_count > (std::numeric_limits<std::size_t>::max() - sizeof(EventHeader)) /
                              sizeof(LevelRecord)) {
            return false;
        }
        const std::size_t expected_size =
            sizeof(EventHeader) + level_count * sizeof(LevelRecord);
        if (header.record_size != expected_size ||
            expected_size > bytes.size() - offset) {
            return false;
        }
        const auto payload = bytes.subspan(
            offset + sizeof(EventHeader),
            expected_size - sizeof(EventHeader));
        if (crc32(payload) != header.payload_crc32) {
            return false;
        }

        const auto type = static_cast<EventType>(header.type);
        if (type == EventType::snapshot) {
            if (header.first_id == 0 || header.first_id != header.final_id ||
                header.exchange_time_ms != 0 || header.trade_price != 0 ||
                header.trade_quantity != 0 || header.flags != 0 || level_count == 0) {
                return false;
            }
        } else if (type == EventType::depth) {
            if (header.first_id == 0 || header.first_id > header.final_id ||
                header.trade_price != 0 || header.trade_quantity != 0 ||
                header.flags != 0) {
                return false;
            }
        } else if (type == EventType::trade) {
            if (level_count != 0 || header.first_id == 0 || header.final_id != 0 ||
                header.trade_price <= 0 || header.trade_quantity == 0 ||
                (header.flags & ~1U) != 0) {
                return false;
            }
        } else if (type == EventType::boundary) {
            if (level_count != 0 || header.first_id != 0 || header.final_id != 0 ||
                header.exchange_time_ms != 0 || header.trade_price != 0 ||
                header.trade_quantity != 0 || header.flags == 0 || header.flags > 6) {
                return false;
            }
        } else {
            return false;
        }

        for (std::size_t index = 0; index < level_count; ++index) {
            LevelRecord level{};
            std::memcpy(
                &level,
                payload.data() + index * sizeof(LevelRecord),
                sizeof(level));
            if (level.price <= 0) {
                return false;
            }
            if (type == EventType::snapshot && level.quantity == 0) {
                return false;
            }
        }
        offset += expected_size;
    }
    return offset == bytes.size();
}

}  // namespace quant::l2::binary
