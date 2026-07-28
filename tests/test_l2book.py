from __future__ import annotations

import random

import pytest

from l2book import (
    ApplyResult,
    DepthSynchronizer,
    DepthUpdate,
    L2OrderBook,
    Level,
    Snapshot,
    SnapshotResult,
    parse_price,
    parse_quantity,
)


def snapshot(update_id: int = 100) -> Snapshot:
    return Snapshot(
        1_000,
        update_id,
        (Level(10_000, 500), Level(9_900, 700)),
        (Level(10_100, 600), Level(10_200, 800)),
    )


def test_fixed_point_parsing_is_exact() -> None:
    assert parse_price("64130.50000000") == 6_413_050_000_000
    assert parse_quantity("1.25000000") == 125_000_000
    with pytest.raises(ValueError):
        parse_price("1.000000001")
    assert parse_price("1.000000000") == 100_000_000


def test_snapshot_bridge_and_gap_detection() -> None:
    synchronizer = DepthSynchronizer()
    assert (
        synchronizer.ingest(
            DepthUpdate(2, 2, 99, 101, (Level(10_000, 450),), ())
        )
        is ApplyResult.BUFFERED
    )
    synchronizer.ingest(DepthUpdate(3, 3, 102, 103, (), (Level(10_100, 550),)))
    installed = synchronizer.install_snapshot(snapshot())
    assert installed.result is SnapshotResult.SYNCHRONIZED
    assert [event.final_update_id for event in installed.applied_events] == [101, 103]
    assert synchronizer.book.last_update_id == 103
    assert synchronizer.book.best_bid.quantity == 450

    result = synchronizer.ingest(
        DepthUpdate(4, 4, 105, 105, (), (Level(10_100, 500),))
    )
    assert result is ApplyResult.GAP_DETECTED


def test_stale_snapshot_is_retried() -> None:
    synchronizer = DepthSynchronizer()
    synchronizer.ingest(DepthUpdate(2, 2, 150, 151, (Level(10_000, 450),), ()))
    assert (
        synchronizer.install_snapshot(snapshot()).result
        is SnapshotResult.SNAPSHOT_TOO_OLD
    )


def test_random_updates_preserve_invariants() -> None:
    generator = random.Random(17)
    book = L2OrderBook()
    book.install_snapshot(
        Snapshot(
            1,
            1,
            tuple(Level(10_000 - index * 10, 100 + index) for index in range(50)),
            tuple(Level(10_100 + index * 10, 100 + index) for index in range(50)),
        )
    )
    for update_id in range(2, 10_000):
        bid = generator.choice((True, False))
        offset = generator.randint(1, 80)
        price = 10_000 - offset * 10 if bid else 10_100 + offset * 10
        quantity = generator.randint(0, 10_000)
        update = DepthUpdate(
            update_id,
            update_id,
            update_id,
            update_id,
            (Level(price, quantity),) if bid else (),
            () if bid else (Level(price, quantity),),
        )
        book.apply(update)
        book.validate()
    assert book.state_hash() > 0


def test_crossed_book_is_rejected() -> None:
    book = L2OrderBook()
    with pytest.raises(RuntimeError, match="crossed"):
        book.install_snapshot(
            Snapshot(1, 1, (Level(10_000, 1),), (Level(10_000, 1),))
        )


def test_fixed_point_text_parser_rejects_noncanonical_syntax() -> None:
    with pytest.raises(ValueError):
        parse_price("1e-8")
    with pytest.raises(ValueError):
        parse_price(" 1.0")
    with pytest.raises(TypeError):
        parse_quantity(True)


def test_l2_identifiers_must_fit_uint64() -> None:
    with pytest.raises(ValueError):
        Snapshot(1, 1 << 64, (Level(10_000, 1),), (Level(10_100, 1),))
    with pytest.raises(ValueError):
        DepthUpdate(1, 0, 1, 1 << 64)
