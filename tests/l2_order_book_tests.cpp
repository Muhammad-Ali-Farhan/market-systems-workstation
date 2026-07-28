#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "L2Synchronizer.hpp"

using quant::l2::ApplyResult;
using quant::l2::DepthUpdate;
using quant::l2::FlatOrderBook;
using quant::l2::Level;
using quant::l2::MapOrderBook;
using quant::l2::Snapshot;
using quant::l2::SnapshotResult;
using quant::l2::Synchronizer;

namespace {

Snapshot base_snapshot() {
    return Snapshot{
        1'000,
        100,
        {{10'000, 500}, {9'900, 700}, {9'800, 900}},
        {{10'100, 600}, {10'200, 800}, {10'300, 1'000}},
    };
}

void test_decimal_parser() {
    std::int64_t price = 0;
    std::uint64_t quantity = 0;
    assert(quant::l2::parse_price("64130.50000000", price));
    assert(price == 6'413'050'000'000LL);
    assert(quant::l2::parse_quantity("1.25000000", quantity));
    assert(quantity == 125'000'000ULL);
    assert(!quant::l2::parse_price("-1.0", price));
    assert(!quant::l2::parse_price("1.000000001", price));
    assert(quant::l2::parse_price("1.000000000", price));
}

void test_snapshot_and_delta() {
    FlatOrderBook book;
    book.install_snapshot(base_snapshot());
    assert(book.best_bid().price == 10'000);
    assert(book.best_ask().price == 10'100);

    book.apply(DepthUpdate{
        2'000,
        123,
        101,
        102,
        {{10'000, 0}, {10'050, 300}},
        {{10'100, 550}},
    });
    assert(book.best_bid().price == 10'050);
    assert(book.best_ask().quantity == 550);
    assert(book.last_update_id() == 102);
    assert(book.state_hash() == 2'980'195'951'085'581'956ULL);
    book.validate_or_throw();
}

void test_official_snapshot_bridge() {
    Synchronizer synchronizer;
    assert(synchronizer.ingest(DepthUpdate{2, 2, 99, 101, {{10'000, 450}}, {}}) ==
           ApplyResult::buffered);
    assert(synchronizer.ingest(DepthUpdate{3, 3, 102, 103, {}, {{10'100, 500}}}) ==
           ApplyResult::buffered);

    const auto result = synchronizer.install_snapshot(base_snapshot());
    assert(result.result == SnapshotResult::synchronized);
    assert(result.applied_events == 2);
    assert(synchronizer.book().last_update_id() == 103);
    assert(synchronizer.book().quantity_at(true, 10'000) == 450);
    assert(synchronizer.book().quantity_at(false, 10'100) == 500);
}

void test_stale_snapshot_and_gap() {
    Synchronizer synchronizer;
    synchronizer.ingest(DepthUpdate{2, 2, 150, 151, {{10'000, 450}}, {}});
    const auto stale = synchronizer.install_snapshot(base_snapshot());
    assert(stale.result == SnapshotResult::snapshot_too_old);

    synchronizer.reset();
    synchronizer.ingest(DepthUpdate{2, 2, 101, 101, {{10'000, 450}}, {}});
    assert(synchronizer.install_snapshot(base_snapshot()).result ==
           SnapshotResult::synchronized);
    assert(synchronizer.ingest(DepthUpdate{3, 3, 103, 103, {}, {{10'100, 500}}}) ==
           ApplyResult::gap_detected);
}

void test_randomized_reference_equivalence() {
    std::mt19937_64 generator{0xC0FFEEULL};
    std::uniform_int_distribution<int> side_distribution(0, 1);
    std::uniform_int_distribution<int> offset_distribution(1, 80);
    std::uniform_int_distribution<int> quantity_distribution(0, 10'000);

    Snapshot snapshot;
    snapshot.last_update_id = 1;
    for (int index = 0; index < 50; ++index) {
        snapshot.bids.push_back(Level{10'000 - index * 10, static_cast<std::uint64_t>(100 + index)});
        snapshot.asks.push_back(Level{10'100 + index * 10, static_cast<std::uint64_t>(100 + index)});
    }

    MapOrderBook reference;
    FlatOrderBook candidate;
    reference.install_snapshot(snapshot);
    candidate.install_snapshot(snapshot);
    assert(reference.state_hash() == candidate.state_hash());

    for (std::uint64_t update_id = 2; update_id < 25'000; ++update_id) {
        DepthUpdate update;
        update.first_update_id = update_id;
        update.final_update_id = update_id;
        const bool bid = side_distribution(generator) == 0;
        const auto offset = offset_distribution(generator);
        const auto quantity = static_cast<std::uint64_t>(quantity_distribution(generator));
        const auto price = bid ? 10'000 - offset * 10 : 10'100 + offset * 10;
        (bid ? update.bids : update.asks).push_back(Level{price, quantity});
        reference.apply(update);
        candidate.apply(update);
        if (update_id % 257 == 0) {
            assert(reference.state_hash() == candidate.state_hash());
            candidate.validate_or_throw();
        }
    }
    assert(reference.state_hash() == candidate.state_hash());
}

void test_crossed_book_rejected() {
    auto snapshot = base_snapshot();
    snapshot.asks.front().price = 10'000;
    FlatOrderBook book;
    bool rejected = false;
    try {
        book.install_snapshot(snapshot);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

}  // namespace

int main() {
    test_decimal_parser();
    test_snapshot_and_delta();
    test_official_snapshot_bridge();
    test_stale_snapshot_and_gap();
    test_randomized_reference_equivalence();
    test_crossed_book_rejected();
    std::cout << "l2_order_book_tests: PASS\n";
    return 0;
}
