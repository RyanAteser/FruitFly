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

### Binance USD-M archive conversion

`tools/binance_to_flyquant.py` downloads daily `bookTicker` and `aggTrades`
archives and converts them without third-party Python packages. Raw archives and
generated datasets remain untracked.

The conversion contract is:

- use `transaction_time` for `bookTicker` and `transact_time` for `aggTrades`;
- multiply Binance millisecond timestamps by 1,000,000;
- apply a half-open `[start, end)` UTC window;
- order equal timestamps by QUOTE before TRADE, then native Binance ID, then source row;
- assign a one-based global `sequence` after the merge;
- leave `receive_ts_ns` empty because archive data has no local receive time;
- map `is_buyer_maker=true` to sell aggressor `S`, otherwise buy aggressor `B`;
- preserve the archive's decimal price and quantity strings.

Example:

```bash
python3 tools/binance_to_flyquant.py \
  --date 2024-01-15 \
  --start 2024-01-15T00:00:00Z \
  --end 2024-01-15T03:00:00Z \
  --archive-dir data/raw/binance \
  --output data/btc_events.csv
```

Pass the recorded archive and output hashes with `--book-sha256`,
`--trades-sha256`, and `--expected-output-sha256` to make a provenance mismatch
fail immediately.

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
