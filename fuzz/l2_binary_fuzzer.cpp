#include <cstddef>
#include <cstdint>
#include <span>

#include "L2BinaryFormat.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const auto* bytes = reinterpret_cast<const std::byte*>(data);
    (void)quant::l2::binary::validate_buffer(std::span<const std::byte>{bytes, size});
    return 0;
}
