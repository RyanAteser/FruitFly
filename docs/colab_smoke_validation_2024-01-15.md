# Colab smoke validation — 2024-01-15 BTCUSDT window

This records the first end-to-end FlyQuant Colab smoke run that completed successfully and the checked-in procedure for reconstructing its market-data window.

## Status

The historical Colab run passed:

- clean clone of `RyanAteser/FruitFly`
- CMake Release build and CTest
- market-event ordering/schema validation
- chronological TRAIN / VALIDATION / TEST split construction with 5-minute embargoes
- baseline VALIDATION run
- synthetic connectome load and reservoir training
- `.fqmodel` export and production `predict-connectome` round-trip reload

TEST was defined but not evaluated. The graph/model were synthetic. The original
`btc_events.csv` was produced by an uncommitted Colab cell, so its byte hash is
retained below as historical provenance rather than presented as reproducible.

The checked-in `tools/binance_to_flyquant.py` now makes archive download and
conversion rerunnable. It reproduces the historical event counts from the exact
source archives. Its explicit conversion contract produces a new canonical CSV
hash, also recorded below. Downstream model hashes still describe the historical
run; they have not been relabeled as outputs of the new canonical CSV.

## Reproduce the market-data stage

From the repository root with Python 3.10 or newer:

```bash
python3 tools/binance_to_flyquant.py \
  --date 2024-01-15 \
  --start 2024-01-15T00:00:00Z \
  --end 2024-01-15T03:00:00Z \
  --archive-dir data/raw/binance \
  --output data/btc_events.csv \
  --book-sha256 50de14aafe593160e613bcdb46e0f172cbc9ef222ae64ec84b6e3e5ef711d37f \
  --trades-sha256 9a2ecc07cfcb79f83ead700249866c7cb8de80930e2287e4e17618a3d8a09029 \
  --expected-output-sha256 fa1433c0b51dee66e95f71ba64f3436f1bdb8a80de8e118727621ebe927ad013
```

Expected summary:

```text
events=2782788 quotes=2658946 trades=123842
fa1433c0b51dee66e95f71ba64f3436f1bdb8a80de8e118727621ebe927ad013  data/btc_events.csv
```

The converter uses Binance transaction timestamps, a half-open UTC window,
QUOTE-before-TRADE ordering for equal timestamps, native IDs and source rows as
subsequent tie-breakers, and a new one-based global sequence. The exact contract
is documented in `data/README.md` and locked by fixture tests.

Run all C++ and ingestion tests with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## Market-data snapshot

- venue: Binance
- market: BTCUSDT USD-M perpetual
- archive date: 2024-01-15
- selected UTC window: 00:00:00–03:00:00
- generated events: 2,782,788
- quote events: 2,658,946
- trade events: 123,842
- historical smoke-run `btc_events.csv` SHA-256: `4c68c0c184bc3b3492fbadc10e59fff9d7a5ba8b543ade0943a4709a52a3c3bf`
- canonical checked-in conversion SHA-256: `fa1433c0b51dee66e95f71ba64f3436f1bdb8a80de8e118727621ebe927ad013`

Archive hashes:

- `bookTicker`: `50de14aafe593160e613bcdb46e0f172cbc9ef222ae64ec84b6e3e5ef711d37f`
- `aggTrades`: `9a2ecc07cfcb79f83ead700249866c7cb8de80930e2287e4e17618a3d8a09029`

## Split boundaries

- TRAIN: 2024-01-15T00:01:00Z → 2024-01-15T01:58:00Z
- VALIDATION: 2024-01-15T02:03:00Z → 2024-01-15T02:28:00Z
- TEST: 2024-01-15T02:33:00Z → 2024-01-15T02:59:00Z

The automatic split is an engineering smoke-test split, not a frozen scientific evaluation schedule.

## Synthetic smoke graph

- neurons: 128
- edges: 1,024
- seed: 20260917
- neurons SHA-256: `000ec2a2df263cdbafef43348689af1eb54fd6e2dc086aed069f54ca32d9e01c`
- edges SHA-256: `d89205f3d0d10140660588e9cba338fc91e90971d421b7ccdfc7423ebd46076a`
- sensory SHA-256: `677335cd3e0cc7988b61beef1ee0c41b7ac6313703804b51fc9ec8cb568750e3`
- outputs SHA-256: `6d16973f0c7ebf09608627d55727a4f56b08e84c598350c323e8f3adf1a17a5a`

## Export

- model: `synthetic_smoke_reservoir.fqmodel`
- model size: 2,408 bytes
- model SHA-256: `67af9ea39d9b27e78c5f3b1d56704032bb8cca61901f9b5c498f3742757f262e`
- source commit used by the historical successful run: `770c133d23614d041b177cfd8e13f59b12ca9f84`

## Interpretation

This run validates the software and data plumbing only. The graph is synthetic
and the exported model must not be interpreted as evidence for or against the
Drosophila connectome hypothesis. A future full rerun should use the canonical
converter output and record new downstream hashes instead of copying the
historical model hash.

The next research step is to implement a frozen MaleCNS importer that produces
the existing normalized `neurons.csv` and `edges.csv` contracts. After that, the
same BTC → feature → train → export → round-trip path can be reused without
changing the experiment infrastructure.
