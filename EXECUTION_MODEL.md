# Execution Simulator Model

## Goal

The simulator tests whether held-out signals remain plausible after market interaction assumptions. It is not a matching-engine replica.

## Orders

- Market and limit orders.
- Buy and sell sides.
- Partial fills.
- Decision + transmission latency before arrival.
- Cancel latency.
- Optional limit-order expiration.
- Position limit and marked-equity loss kill switch.

## Market orders

A market order sweeps current visible levels in price priority. Consumed quantity is shadowed until the next snapshot/depth update so two simulated orders cannot reuse the same displayed liquidity.

## Passive orders

At arrival, queue ahead includes a configured fraction of displayed quantity and earlier own orders at the same side/price. Aggregate trades at the order price consume absolute queue position in FIFO order.

Three explicit sensitivity modes exist:

- `trade_only`: only observed aggregate trades can fill passive orders.
- `pro_rata_depth`: qualifying depth depletion contributes a configured fraction.
- `optimistic_depth`: all qualifying depletion can fill; interpret as an optimistic upper bound.

Depth decreases are ambiguous because aggregated feeds do not identify cancellations versus executions. Results must be presented across models, never as exact fills.

## Provenance gate

Execution-sensitivity analysis requires the originating L2 research report. Before replay, the tool verifies the prediction artifact hash and checks each held-out session against the exact recording hash, checkpoint-sidecar hash, and symbol recorded by the experiment. Session mappings that are missing, extra, or content-mismatched are rejected.

## Accounting

- Buy fills reduce cash and increase inventory.
- Sell fills increase cash and reduce inventory.
- Maker/taker fees are assessed in quote currency; maker rebates may be negative fees.
- Ending equity marks remaining inventory at the latest midpoint.
- Signed future midpoint markouts measure adverse or favorable selection from each fill.

## Boundaries

A connection or sequence boundary cancels active orders and clears the book. Continuing orders across an unknown market-data interval would be misleading.

## Known omissions

- Exact exchange queue priority.
- Hidden/iceberg liquidity.
- Per-order market-by-order messages.
- Self-trade prevention and account-specific exchange filters.
- Rate limits, acknowledgements, rejects, and matching-engine timestamps.
- Market impact beyond displayed-depth consumption.
- Cross-venue routing.

These are stated limitations, not silently approximated facts.
