# FlyQuant model artifacts

`*.fqmodel` files are portable C++ FlyQuant model artifacts. Version 1 stores:

- the TRAIN-only feature mean and standard deviation;
- recurrent dynamics parameters;
- frozen sensory neuron assignments;
- frozen UP/DOWN output neuron assignments;
- trained UP/DOWN readout weights and biases;
- SHA-256 hashes of the neuron, edge, sensory-map, and output-map source files;
- the readout training hyperparameters.

The artifact intentionally does **not** contain BTC observations or experimental results.

## Load contract

`predict-connectome` requires the same normalized `neurons.csv` and `edges.csv` used during training. Their SHA-256 hashes are checked against the artifact before inference. A mismatch fails closed.

```bash
./build/flyquant predict-connectome \
  --config config/phase1.conf \
  --neurons data/connectome/neurons.csv \
  --edges data/connectome/edges.csv \
  --model-in models/exports/connectome_reservoir_v1.fqmodel \
  --split validation
```

Do not commit a trained artifact as evidence of performance without its immutable run directory and the exact source-data hashes.
