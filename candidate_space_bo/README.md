# Candidate-space (featurization) Bayesian optimization

A worked, runnable demonstration of the **predefined-candidate-space /
featurization** approach to MOF optimization — the alternative to generative
latent-space optimization discussed in
[`docs/candidate-space-optimization.md`](../docs/candidate-space-optimization.md).

Instead of decoding a free latent vector, we treat the existing MOF library as a
**fixed discrete search space**, featurize each candidate, and let Bayesian
optimization choose which *existing* MOF to evaluate next. Every proposal is
therefore a real, valid MOF — no decoder needed.

## Run

```bash
# structural-descriptor featurizer (no training, reads dataset/mof_output.zip,
# dataset/mof_output.csv, or MOF-GRU.zip automatically). --n-candidates 0 uses
# the FULL candidate library (no subsampling).
python candidate_space_bo/optimize_candidates.py \
    --objective CH4ABL --featurizer descriptors \
    --n-candidates 0 --iters 100 --seeds 8

# learned MOF-GRU-embedding featurizer (needs torch + a trained checkpoint)
python candidate_space_bo/optimize_candidates.py \
    --objective CH4ABL --featurizer gru \
    --checkpoint my_models/new/biGRU_CH4ABL_model_ep_40_em_80_hd200.pth
```

Outputs `bo_trace.png` comparing GP+Expected-Improvement BO against random search
over the same candidate pool.

The committed `bo_trace.png` was generated on the **full dataset** — all
**113,160** real MOFs in `dataset/mof_output.csv` (`--n-candidates 0`), 100 BO
iterations averaged over 8 seeds (~6 min, CPU only). Over that pool BO reaches a
best `CH4ABL` of ≈2.47 vs ≈2.16 for random search after 100 labelled
evaluations (pool optimum ≈3.63), confirming the GP+EI surrogate consistently
out-samples random selection even when the library is large. Pass a smaller
`--n-candidates` (e.g. `6000`) for a quick subsampled demo.

![BO trace](bo_trace.png)

## Surrogate comparison: GP over GNN embeddings vs. the GNN's own predictions

[`gp_vs_gnn_surrogate.py`](gp_vs_gnn_surrogate.py) keeps the **same** frozen
MOF-GRU encoder (the 400-d pooled hidden state) and swaps only the *surrogate*
that drives the active-learning loop:

```bash
python candidate_space_bo/gp_vs_gnn_surrogate.py \
    --objective CH4ABL \
    --checkpoint my_models/new/biGRU_CH4ABL_model_ep_40_em_80_hd200.pth \
    --n-candidates 6000 --iters 50 --seeds 6
```

It compares a **GP over the GNN embeddings** (Expected Improvement), the **GNN's
own predictor as a deep-ensemble surrogate** (retrained on observed labels each
round, EI), a **leaky greedy reference** using the pretrained GNN's in-sample
predictions, and **random search**. Over the raw 400-d embedding the
neural-ensemble surrogate (≈2.73) clearly beats the GP (≈2.07) and nearly matches
the leaky pretrained-GNN ceiling (≈2.79); see
[`docs/candidate-space-optimization.md`](../docs/candidate-space-optimization.md)
for discussion (and why reducing dimensionality makes the GP competitive again).

![GP vs GNN surrogate](gp_vs_gnn_trace.png)

## Dependencies

- `descriptors` featurizer: `numpy`, `scikit-learn`, `matplotlib` (`scipy` comes
  with scikit-learn).
- `gru` featurizer / `gp_vs_gnn_surrogate.py`: additionally `torch` and
  `selfies`; reuses `models.py` / `utils.py` from the repository root.
