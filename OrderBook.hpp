
#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

// Fixed 32-byte record shared between C++ and NumPy.
//
// timestamp_ns is the local steady-clock receipt timestamp captured after the
// complete WebSocket message has been read. Replay preserves this value exactly.
// Volumes use fixed-point scaling: stored value 1,000,000 = quantity 1.0.
struct OrderBookState {
    std::uint64_t timestamp_ns;
    double best_bid;
    double best_ask;
    std::uint32_t bid_volume;
    std::uint32_t ask_volume;
};

static_assert(sizeof(OrderBookState) == 32,
              "OrderBookState must remain exactly 32 bytes.");
static_assert(std::is_standard_layout_v<OrderBookState>,
              "OrderBookState must use standard layout.");
static_assert(std::is_trivially_copyable_v<OrderBookState>,
              "OrderBookState must be trivially copyable.");

inline bool valid_order_book_state(const OrderBookState& state) noexcept {
    return state.timestamp_ns > 0 &&
           std::isfinite(state.best_bid) &&
           std::isfinite(state.best_ask) &&
           state.best_bid > 0.0 &&
           state.best_ask > 0.0 &&
           state.best_bid <= state.best_ask;
}

