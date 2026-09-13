# FlyQuant-1 Phase 1 protocol v0.1

## Scientific question

**H0:** authentic fruit-fly connectome topology provides no meaningful advantage for BTC direction prediction versus matched controls.

**H1:** authentic topology provides a reproducible out-of-sample advantage.

This repository is built to make H0 easy to retain. A negative result is a valid result.

## Initial market task

- Underlying: BTC-USD spot or perpetual data from one liquid centralized exchange per dataset version.
- Observation anchors: exact 1-minute UTC grid.
- Features may use sub-minute events, but only events with `exchange_ts_ns <= anchor_ts_ns`.
- Primary target horizon: exactly 300 seconds.
- Target midpoint uses the most recent valid quote at or before `t + 300s`, subject to `max_quote_age_ns`.
- `y_t = 1` iff `mid(t+300s) > mid(t)`; otherwise `0` (ties are DOWN for the binary v0.1 target and are counted, not discarded).
- Future log return is stored for analysis but is not a second optimization target.

## Exact v0.1 features

All return features use midpoint `m(t)=(bid(t)+ask(t))/2` with a quote no older than `max_quote_age_ns`.

1. `log_return_1s = ln(m(t)/m(t-1s))`
2. `log_return_5s = ln(m(t)/m(t-5s))`
3. `log_return_15s = ln(m(t)/m(t-15s))`
4. `log_return_30s = ln(m(t)/m(t-30s))`
5. `log_return_60s = ln(m(t)/m(t-60s))`
6. `log_return_300s = ln(m(t)/m(t-300s))`
7. `momentum_acceleration = log_return_5s/5 - log_return_30s/30`
8. `realized_volatility_60s = sqrt(sum_{k=1..60} r_1s(k)^2)` using one-second midpoint samples ending at `t`
9. `relative_spread = (ask-bid)/mid`
10. `top_book_imbalance = (bid_qty-ask_qty)/(bid_qty+ask_qty)`, or zero if both sizes are zero
11. `microprice_displacement = (microprice-mid)/mid`, with `microprice=(ask*bid_qty + bid*ask_qty)/(bid_qty+ask_qty)`
12. `signed_trade_flow_60s = sum(signed_qty)/sum(qty)` over `(t-60s,t]`; unknown aggressor contributes zero signed quantity and remains in total quantity
13. `trade_arrival_intensity_60s = trade_count/60` over `(t-60s,t]`

No feature is added or removed after TEST is viewed without creating a new experiment ID and a new untouched test period.

## Splits and normalization

TRAIN, VALIDATION, and TEST timestamps are explicit configuration values. Each split must be separated from the next by at least the 5-minute prediction horizon. A sample belongs to a split only if both its anchor and its target timestamp fall inside that split.

Feature mean and standard deviation are fit on TRAIN only. The same frozen transform is applied to VALIDATION and TEST. Standardized values are clipped to `[-8,8]` only for numerical robustness; the clipping rule is fixed before evaluation.

TEST is locked by default in the CLI. Running TEST requires `--allow-test`. This is friction, not security; research discipline still matters.

## Leakage assertions

The executable and tests enforce:

- strict market-event ordering by `(exchange_ts_ns, sequence)`;
- all feature source quotes are at or before the prediction anchor;
- target source quote is at or before the target timestamp;
- target timestamp is strictly after anchor;
- chronological sample order;
- split containment and 5-minute embargo;
- normalization fitted from TRAIN only by API design.

The local receive clock is retained for later latency work but is not numerically compared to the exchange clock in v0.1 because clock domains may differ.

## Baselines

The first runnable experiment evaluates on exactly the same samples:

- Always-UP (`p_up=1`)
- Always-DOWN (`p_up=0`)
- deterministic C++ logistic regression with full-batch gradient descent, fixed learning rate, epochs, and L2 regularization from the frozen config

Small feed-forward and recurrent baselines are staged after this infrastructure passes tests and before any positive connectome claim.

## Connectome representation

`Neuron` stores ID, type, region, neurotransmitter metadata, and activation. `Edge` stores source, target, synapse count, biological sign when supported, and a derived computational weight.

Synapse count is **not** treated as a neural-network weight directly. v0.1 provides two explicit transforms:

- `UnsignedLog1pIncomingNormalized`: `log(1+count)` normalized by total incoming transformed magnitude at the target; all edges propagate positively. This is the initial topology-only assumption.
- `SignedLog1pIncomingNormalized`: same magnitude, with explicit excitatory/inhibitory sign; unknown-sign edges receive zero weight rather than an invented sign.

The first topology experiment must report which transform was used.

## Sensory encoder

The market encoder is a versioned CSV manifest:

```text
neuron_id,feature_index,gain,bias
```

It is resolved against real connectome neuron IDs. The manifest is frozen before evaluation. v0.1 does not hard-code a profitable-looking sensory assignment into C++.

The initial semantic design groups momentum/return channels, volatility, spread, order-book imbalance, microprice displacement, and trade-flow channels into predeclared sensory neuron populations. Exact MaleCNS neuron IDs will be selected from the official annotations and committed as a versioned manifest before the connectome experiment begins.

## Neural dynamics

For each market sample, neural state starts at zero to avoid hidden cross-sample state leakage. For eight deterministic propagation steps:

`candidate_i = tanh(input_gain * external_i + recurrent_gain * sum_j(w_ji * state_j))`

`state_i <- (1-leak)*state_i + leak*candidate_i`

Initial constants: `steps=8`, `leak=0.5`, `recurrent_gain=1.0`, `input_gain=1.0`. These are hypotheses, not biological measurements.

## UP/DOWN decoder

A second versioned CSV lists fixed output neurons:

```text
neuron_id,channel
```

where `channel` is `UP` or `DOWN`. Mean activation is computed per population. Probability is the two-way softmax of the two raw activities. Raw UP and DOWN activities must be logged when the connectome runner is enabled.

## Graph controls

- Degree-preserving shuffle: directed double-edge swaps preserve every node's in-degree and out-degree. Seed is fixed. Self loops and duplicate directed pairs are rejected.
- Random graph: same neuron count and edge count, no self loops or duplicate directed pairs; the synapse-count/sign attribute multiset is shuffled onto random pairs.
- Weight-shuffled, sensory-shuffled, output-shuffled, and reduced-connectome controls are required before claiming topology-specific evidence.

All stochastic controls run seeds `42,1337,2026,9001,314159` unless a future preregistered protocol supersedes them.

## Predetermined metrics

Primary metric for Phase 1: **log loss**. It rewards probabilistic correctness and penalizes overconfidence.

Secondary metrics, all reported: directional accuracy, balanced accuracy, precision, recall, ROC AUC, Brier score, 10-bin expected calibration error, and Pearson correlation between `p_up` and future BTC log return.

No single secondary metric can replace the primary metric after results are seen.

## Experiment record

Before an important run, record experiment ID, hypothesis, dataset hash, explicit split timestamps, features, model, seeds, changed/fixed factors, primary metric, pass condition, and fail condition.

Every run directory is newly created and refuses overwrite. It contains manifest, predictions, metrics, and errors. The manifest includes git commit, SHA-256 dataset/config hashes, split periods, feature/connectome/encoder/decoder versions, and seed.

## Initial pass/fail definition

Infrastructure pass: build succeeds, tests pass, baseline run completes on a valid dataset, provenance hashes are written, and TEST remains untouched.

The connectome hypothesis has **no pass condition yet** in v0.1 because the real connectome mapping and matched capacity controls have not been frozen. Setting that condition after seeing connectome results would be invalid.

## Staged implementation order

1. Market-data contract and ordering validator.
2. Feature construction and leakage tests.
3. Chronological split/embargo and TRAIN-only normalization.
4. Always-UP/DOWN and logistic baseline.
5. Immutable provenance/results writer.
6. MaleCNS C++ importer and source-file hashing.
7. Freeze sensory/output manifests from biological annotations.
8. Run real topology with fixed dynamics.
9. Run degree-preserving and random-graph controls across all seeds.
10. Add small feed-forward and recurrent C++ baselines with matched budgets.
11. Add block bootstrap, permutation tests, and paired comparisons.
12. Only after Phase-1 evidence is understood, design the execution/risk layer for trading.
