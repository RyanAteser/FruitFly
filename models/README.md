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

## Colab export and GitHub publish workflow

`notebooks/FlyQuant_Colab_Train.ipynb` is the supported hosted-compute path for producing a repository-ready model export. It:

1. clones the canonical repository, builds the C++20 binary, and runs CTest;
2. verifies the frozen market/connectome input files and records their SHA-256 hashes;
3. trains with `train-connectome` on TRAIN while evaluating VALIDATION;
4. writes `models/exports/connectome_reservoir_v1.fqmodel`;
5. reloads the artifact through `predict-connectome` so the production loader and connectome-source hash checks must pass;
6. writes `models/exports/connectome_reservoir_v1.manifest.json` with the source commit, model hash, input hashes, runtime-config hash, file size, and validation status;
7. pushes only the model and manifest to a new `model-export/<model-sha-prefix>` GitHub branch.

GitHub authentication is supplied at runtime through a Colab secret named `GITHUB_TOKEN`. The notebook uses a temporary `GIT_ASKPASS` helper and does not store the token in the notebook, git remote, model, or manifest.

The notebook refuses a direct GitHub commit when the exported model is at least 95 MiB so large artifacts can be moved to Git LFS or another controlled artifact channel instead of approaching GitHub's normal per-file limit.

TEST remains locked: the notebook never passes `--allow-test`.
