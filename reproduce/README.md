# Reproduction of MOF-GRU

This folder contains a self-contained, cross-platform driver that reproduces the
training/evaluation pipeline of the parent project on Linux/CI.

## Why a new script?

The original `training.py`, `test.py`, and `utils.py` are hard to run as-is:

* they use hard-coded Windows paths (e.g. `dataset\\MOFseq_output.txt`,
  `my_models\\new\\...`) that fail on Linux/macOS;
* `training.py` sweeps a large hyper-parameter grid (`epochs × embedding × hidden`
  = 36 models per property) which is impractical for a quick reproduction;
* `test.py` references model files and column names that are not produced by the
  default training run.

`reproduce.py` reuses the **same model** (`models.GRUModel`) and the **same data
files** (`dataset/MOFseq_output.txt`, `dataset/mof_output.csv`,
`dataset/my_dict_output.json`) through a small, parameterised, path-safe driver.

## Data

The tokenised sequences and labels live in `dataset/` as zip archives. Unzip them
once before running:

```bash
cd dataset
unzip -o MOFseq_output.zip   # -> MOFseq_output.txt (113,160 MOF token sequences)
unzip -o mof_output.zip      # -> mof_output.csv     (113,160 rows of properties)
```

## Usage

```bash
# Quick smoke test
python reproduce/reproduce.py --properties gASA --limit 2000 --epochs 5

# Subset reproduction across several properties (what produced reproduce/results)
python reproduce/reproduce.py \
    --properties Density PLD LCD CH4ABL \
    --limit 4000 --epochs 12 --save-model --outdir reproduce/results
```

Each property writes a parity plot `parity_<prop>.png`, an optional model
`biGRU_<prop>.pt` (state dict), and an aggregate `metrics.json`.

## Results (committed in `results/`)

Subset of 4,000 MOFs (80/20 train/test split, seed 42), bidirectional GRU
(embedding=80, hidden=200, 1 layer), Adam (lr=1e-3), MSE loss, 12 epochs, CPU.

| Property | Test R² | Test MAE | Test SRCC | Train time |
|----------|--------:|---------:|----------:|-----------:|
| Density  | 0.767   | 0.091    | 0.864     | ~6.7 min   |
| CH4ABL   | 0.703   | 0.151    | 0.863     | ~6.9 min   |
| LCD      | 0.681   | 1.71     | 0.815     | ~6.6 min   |
| PLD      | 0.461   | 1.73     | 0.711     | ~6.7 min   |

The training MSE decreases monotonically and the parity plots cluster around the
y = x line, confirming the pipeline reproduces the expected structure–property
learning behaviour. Accuracy is expected to improve further with the full
113k-MOF dataset and more epochs/hyper-parameter tuning, at the cost of much
longer (multi-hour) CPU runtime — this subset run was chosen to fit a short
time budget while demonstrating reproducibility.
