# Data contracts

No market or connectome data is committed to this repository.

## BTC event stream

`flyquant baseline` reads a UTF-8 CSV with this exact header:

```text
exchange_ts_ns,receive_ts_ns,sequence,event_type,bid_px,bid_qty,ask_px,ask_qty,trade_px,trade_qty,aggressor
```

Rules:

- Rows are strictly ordered by `(exchange_ts_ns, sequence)`.
- `event_type` is `QUOTE` or `TRADE`.
- QUOTE rows populate `bid_px,bid_qty,ask_px,ask_qty`; trade fields may be empty.
- TRADE rows populate `trade_px,trade_qty,aggressor`; quote fields may be empty.
- `aggressor` is `B`, `S`, or `U` when unknown.
- `receive_ts_ns` is preserved as supplied. It is not compared numerically with the exchange clock because the two clocks may not share a synchronized epoch.
- Prices and quantities use exchange-native units converted to decimal values before ingestion. That conversion must be documented with the dataset.

## Normalized connectome

The core runtime intentionally does not depend on Python/R or Apache Arrow. A C++ importer will normalize official MaleCNS data into two CSVs.

`neurons.csv`:

```text
id,type,region,neurotransmitter
```

`edges.csv`:

```text
source_id,target_id,synapse_count,sign
```

`sign` is `1`, `-1`, or `0` for excitatory, inhibitory, or unknown. Unknown is not silently assigned a biological sign.

Official MaleCNS v1.0 bulk data includes curated neuron annotations, aggregate neurotransmitter predictions, a full segment-to-segment connectivity table, and Neo4j input CSV files. The importer is deliberately separate so the experiment can preserve the original source files and hashes while keeping the modeling runtime minimal.
