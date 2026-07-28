
#pragma once

#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "OrderBook.hpp"

namespace quant {

inline constexpr std::array<char, 8> BinaryMagic{
    'Q', 'E', 'N', 'G', 'I', 'N', 'E', '1'
};
inline constexpr std::uint32_t BinaryVersion = 1;
inline constexpr std::uint64_t BinaryVolumeScale = 1'000'000ULL;

// Version 1 remains byte-for-byte compatible with existing recordings.
// Additional provenance and completeness information is written to the
// optional <recording>.meta.json sidecar instead of changing this header.
struct BinaryFileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t record_size;
    std::uint32_t flags;
    std::uint64_t volume_scale;
    std::uint64_t created_unix_ns;
    std::uint64_t reserved[3];
};

static_assert(std::endian::native == std::endian::little,
              "The binary format currently requires a little-endian system.");
static_assert(sizeof(BinaryFileHeader) == 64,
              "BinaryFileHeader must remain exactly 64 bytes.");
static_assert(std::is_trivially_copyable_v<BinaryFileHeader>,
              "BinaryFileHeader must be trivially copyable.");

inline std::uint64_t unix_time_ns() noexcept {
    const auto now = std::chrono::system_clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
}

inline BinaryFileHeader make_binary_header() noexcept {
    BinaryFileHeader header{};
    std::memcpy(header.magic, BinaryMagic.data(), BinaryMagic.size());
    header.version = BinaryVersion;
    header.header_size = static_cast<std::uint32_t>(sizeof(BinaryFileHeader));
    header.record_size = static_cast<std::uint32_t>(sizeof(OrderBookState));
    header.flags = 0;
    header.volume_scale = BinaryVolumeScale;
    header.created_unix_ns = unix_time_ns();
    return header;
}

inline bool valid_binary_header(const BinaryFileHeader& header) noexcept {
    return std::memcmp(header.magic, BinaryMagic.data(), BinaryMagic.size()) == 0 &&
           header.version == BinaryVersion &&
           header.header_size == sizeof(BinaryFileHeader) &&
           header.record_size == sizeof(OrderBookState) &&
           header.flags == 0 &&
           header.volume_scale == BinaryVolumeScale;
}

inline constexpr std::array<char, 8> UpdateIdMagic{
    'Q', 'U', 'P', 'D', 'I', 'D', '1', '\0'
};

inline constexpr std::uint32_t UpdateIdVersion = 1;

struct UpdateIdFileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t record_size;
    std::uint32_t flags;
    std::uint64_t created_unix_ns;
};

static_assert(sizeof(UpdateIdFileHeader) == 32);
static_assert(std::is_trivially_copyable_v<UpdateIdFileHeader>);

inline UpdateIdFileHeader make_update_id_header(
    std::uint64_t created_unix_ns) noexcept {
    UpdateIdFileHeader header{};
    std::memcpy(header.magic, UpdateIdMagic.data(), UpdateIdMagic.size());
    header.version = UpdateIdVersion;
    header.header_size = static_cast<std::uint32_t>(sizeof(UpdateIdFileHeader));
    header.record_size = static_cast<std::uint32_t>(sizeof(std::uint64_t));
    header.flags = 0;
    header.created_unix_ns = created_unix_ns;
    return header;
}

inline bool valid_update_id_header(
    const UpdateIdFileHeader& header,
    std::uint64_t expected_created_unix_ns) noexcept {
    return std::memcmp(
               header.magic, UpdateIdMagic.data(), UpdateIdMagic.size()) == 0 &&
           header.version == UpdateIdVersion &&
           header.header_size == sizeof(UpdateIdFileHeader) &&
           header.record_size == sizeof(std::uint64_t) &&
           header.flags == 0 &&
           header.created_unix_ns == expected_created_unix_ns;
}

}  // namespace quant

