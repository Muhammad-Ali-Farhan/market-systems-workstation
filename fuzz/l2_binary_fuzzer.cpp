// libFuzzer entry point for the defensive Level-2 binary validator. Arbitrary byte
// strings must be rejected safely or accepted only when all format invariants hold.

#include <cstddef>
#include <cstdint>
#include <span>

#include "market_engine/recording/L2BinaryFormat.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const auto* bytes = reinterpret_cast<const std::byte*>(data);
    (void)quant::l2::binary::validate_buffer(std::span<const std::byte>{bytes, size});
    return 0;
}
