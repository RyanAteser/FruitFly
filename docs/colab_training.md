# Google Colab training path

The Colab notebook is only an execution host. All feature construction, splitting, normalization, neural propagation, optimization, metrics, serialization, and inference are implemented in C++20 in this repository.

Notebook orchestration uses shell/CMake commands because Colab is hosted around a Python kernel, but there are no Python research, preprocessing, training, evaluation, or conversion scripts.

## Required inputs

Upload these files into `/content/FruitFly/input/` using Colab's Files panel:

- `phase1.conf` — frozen experiment configuration with absolute Unix-nanosecond split boundaries.
- `btc_events.csv` — ordered BTC event stream matching `data/README.md`.
- `neurons.csv` — normalized connectome neurons.
- `edges.csv` — normalized connectome edges.
- `sensory.csv` — frozen market-to-neuron mapping, header `neuron_id,feature_index,gain,bias`.
- `outputs.csv` — frozen output mapping, header `neuron_id,channel`, with `UP` and `DOWN` rows.

The `dataset_path` in `phase1.conf` should be `/content/FruitFly/input/btc_events.csv`, and `results_dir` should be `/content/FruitFly/results`.

## Training rule

The first notebook run uses `--split validation`. The C++ trainer fits feature normalization and readout weights using TRAIN only. Validation is evaluated but never used by the optimizer. TEST remains locked.

The recurrent edge weights are not trained in v1. The real connectome acts as a fixed reservoir. This is deliberate: the first experiment asks whether authentic topology creates useful states before adding a large trainable parameter budget.

## Export

The notebook writes:

`/content/FruitFly/models/exports/connectome_reservoir_v1.fqmodel`

Download that file using Colab's Files panel. The same repository can reload it with `predict-connectome`; source hashes are verified before inference.
