#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

#include "L2BinaryFormat.hpp"

using quant::l2::binary::EventHeader;
using quant::l2::binary::EventType;
using quant::l2::binary::FileHeader;
using quant::l2::binary::LevelRecord;

namespace {

std::vector<std::byte> valid_file() {
    FileHeader header{};
    std::memcpy(header.magic, quant::l2::binary::Magic.data(), 8);
    header.version = quant::l2::binary::Version;
    header.header_size = sizeof(FileHeader);
    header.event_header_size = sizeof(EventHeader);
    header.level_size = sizeof(LevelRecord);
    header.price_scale = quant::l2::PriceScale;
    header.quantity_scale = quant::l2::QuantityScale;
    header.created_unix_ns = 1;
    std::memcpy(header.symbol, "BTCUSDT", 7);

    const std::array<LevelRecord, 2> levels{{
        {10'000, 500},
        {10'100, 600},
    }};
    const auto payload = std::as_bytes(std::span{levels});
    EventHeader event{};
    event.type = static_cast<std::uint32_t>(EventType::snapshot);
    event.receipt_timestamp_ns = 1;
    event.first_id = 100;
    event.final_id = 100;
    event.bid_count = 1;
    event.ask_count = 1;
    event.payload_crc32 = quant::l2::binary::crc32(payload);
    event.record_size = sizeof(EventHeader) + payload.size();

    std::vector<std::byte> bytes(sizeof(header) + sizeof(event) + payload.size());
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + sizeof(header), &event, sizeof(event));
    std::memcpy(bytes.data() + sizeof(header) + sizeof(event), payload.data(), payload.size());
    return bytes;
}

}  // namespace

int main() {
    auto bytes = valid_file();
    assert(quant::l2::binary::validate_buffer(bytes));

    auto corrupted = bytes;
    corrupted.back() ^= std::byte{0x01};
    assert(!quant::l2::binary::validate_buffer(corrupted));

    auto truncated = bytes;
    truncated.pop_back();
    assert(!quant::l2::binary::validate_buffer(truncated));

    auto reserved = bytes;
    FileHeader header{};
    std::memcpy(&header, reserved.data(), sizeof(header));
    header.reserved[0] = 1;
    std::memcpy(reserved.data(), &header, sizeof(header));
    assert(!quant::l2::binary::validate_buffer(reserved));

    std::cout << "l2_binary_format_tests: PASS\n";
    return 0;
}
