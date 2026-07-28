# L2 Binary Format v1

All integers are little-endian. The format currently rejects big-endian hosts in the native implementation. Prices and quantities are exact fixed point with scale `100,000,000`.

## File header — 128 bytes

| Field | Type | Meaning |
|---|---:|---|
| magic | 8 bytes | `QL2EVT1\0` |
| version | u32 | `1` |
| header_size | u32 | `128` |
| event_header_size | u32 | `80` |
| level_size | u32 | `16` |
| price_scale | u64 | `100000000` |
| quantity_scale | u64 | `100000000` |
| created_unix_ns | u64 | artifact creation time |
| symbol | 16 bytes | upper-case ASCII, NUL padded |
| reserved | 8 × u64 | must be zero |

## Event header — 80 bytes

Events are snapshot, depth, aggregate trade, or boundary. The header stores type, flags, receipt timestamp, exchange event time, update/trade identifiers, bid/ask level counts, trade price/quantity, payload CRC32, total record size, and a reserved field that must be zero.

Snapshot and depth payloads contain bids followed by asks. Each level is:

```text
price:    signed 64-bit fixed point
quantity: unsigned 64-bit fixed point
```

A zero quantity is valid in a depth event and means deletion. Snapshot levels must be positive.

## Checkpoint companion

`<recording>.l2chk` contains a 32-byte header followed by 24-byte records:

```text
event_index: u64
update_id:   u64
state_hash:  u64
```

Checkpoint event indices must increase strictly. A complete recording ends with a checkpoint matching the final event count, update ID, and state hash in metadata.

## Metadata sidecar

The atomically published JSON sidecar binds the symbol, source, stream identity, scales, created timestamp, filenames, event counts, counters, completion status, final state, and SHA-256 hashes.

`data_complete=true` requires:

- Clean shutdown.
- At least one snapshot and one depth event.
- At least one checkpoint.
- Zero queue drops.
- Zero malformed messages.
- Positive final update ID and state hash.
- Final checkpoint equal to sidecar final state.

Sequence gaps may occur and be recovered; their boundaries divide contiguous research segments. A missing sidecar beside a current checkpoint companion is treated as interruption, never as a legacy complete recording.

## Compatibility rule

Readers reject unknown versions, nonzero reserved fields, unknown event types, invalid level counts, impossible record sizes, CRC failures, partial records, and unsupported sidecar identities. Format changes require a new version; they must never silently reinterpret v1 bytes.
