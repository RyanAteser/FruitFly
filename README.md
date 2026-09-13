# FruitFly / FlyQuant-1

FlyQuant-1 is an experimental C++ research system testing a narrow falsifiable question:

> Does authentic *Drosophila* connectome topology add reproducible out-of-sample information for short-horizon BTC direction prediction compared with conventional and randomized controls?

It does **not** assume a fruit fly can trade Bitcoin. The repository is intentionally built so the result “no detectable advantage” is easy to obtain and report honestly.

## Status

Phase 1 infrastructure only. No profitability or predictive-success claim is made.

Implemented:

- C++20/CMake build with no runtime third-party dependency
- ordered BTC event-stream parser
- exact 13-feature v0.1 feature pipeline
- exact 5-minute midpoint target
- chronological TRAIN/VALIDATION/TEST with 5-minute embargo
- leakage audits and CTest coverage
- Always-UP / Always-DOWN baselines
- deterministic C++ logistic regression
- connectome neuron/edge structures and normalized CSV loader
- explicit synapse-count-to-computational-weight transforms
- versionable sensory encoder and UP/DOWN decoder
- deterministic recurrent graph propagation proposal/implementation
- degree-preserving shuffled-connectome control
- matched-size random-graph control
- fixed prediction metrics
- SHA-256 provenance and append-only run directories
- TEST lock requiring an explicit CLI override

See `docs/protocol_v0.1.md` for the frozen initial definitions.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Baseline run

Copy `config/phase1.example.conf`, set a real dataset path and explicit Unix-nanosecond split boundaries, then run VALIDATION:

```bash
./build/flyquant baseline --config config/phase1.conf --split validation
```

Do not run TEST while developing the model. Once the protocol is frozen:

```bash
./build/flyquant baseline --config config/phase1.conf --split test --allow-test
```

The explicit flag is intentional friction against accidental peeking.

## Market-data schema

See `data/README.md`. The primary input is one ordered CSV containing QUOTE and TRADE events with exchange timestamp, local receive timestamp when available, and sequence number when available.

## MaleCNS source

The intended biological source is HHMI Janelia's MaleCNS v1.0 connectome, released in 2026 and described by Google Research. The official release exposes curated annotations, neurotransmitter predictions, full segment-to-segment connectivity, skeletons, and bulk database inputs. FlyQuant-1 will import a frozen source release using C++ and store the original source hash plus a normalized graph hash.

The repository does not vendor the connectome or BTC datasets.

## Research rule

If logistic regression wins, report it. If a shuffled graph wins, report it. If the real connectome does not survive controls, report that. TEST data that influences design is no longer TEST data.
