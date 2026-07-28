
# Security Policy

## Reporting

Please report security-sensitive issues privately to the repository owner rather than opening a public issue with exploit details.

Include the affected commit, operating system, reproduction steps, expected behavior, observed behavior, and whether recordings or credentials were exposed.

## Data and credential policy

- The project does not require Binance API keys for public market-data ingestion.
- Do not commit recordings that contain information you do not intend to publish.
- Do not disable TLS peer or hostname verification.
- Treat `.qbin`, `.qids`, model artifacts, reports, logs, and crash traces as generated data.
- Do not load untrusted native `.pyd`/DLL files.

## Supported branch

Security fixes are applied to the current default branch. Historical snapshots and generated desktop releases may not receive backports.


## L2 network and artifact safety

- WebSocket TLS certificates and hostnames are verified.
- REST snapshots use HTTPS with the certifi trust store and a bounded timeout.
- Symbols are normalized and validated before being placed in endpoints or filenames.
- L2 readers bound level counts and record sizes before allocation.
- Event payload CRCs, reserved fields, versions, scales, symbol identity, sidecar counts, checkpoint identity, and optional SHA-256 hashes are validated.
- Existing recordings, checkpoints, metadata, research reports, and packaged evidence are never overwritten implicitly.
- Fuzz targets must operate only on temporary/generated inputs and must not access live credentials or user data.
