# Edison Scientific spot-check of the MOF-GRU reproduction

This folder contains the artifacts from an Edison Scientific **Analysis** task
requesting a critical feedback / spot-check of the reproduction effort in
[`reproduce/`](../).

* **Job type:** `JobNames.ANALYSIS` (`job-futurehouse-data-analysis-crow-high`)
* **Task id:** `8db472b9-4272-4794-82ef-e274e1ff6e21`
* **Uploaded bundle:** `models.py`, `training.py`, `test.py`, `utils.py`, the
  original and reproduce READMEs, `reproduce.py`, `metrics.json`, `run.log`, and
  the four `parity_<prop>.png` plots.

## Artifacts

* `analysis_answer.md` — the full written assessment.
* `analysis_notebook.ipynb` — the analysis notebook (the embedded figures are the
  same parity plots that live in [`../results/`](../results/), so they are not
  duplicated here as standalone PNGs).

## Summary of the feedback

* **Faithful to the original *executable* pipeline** (same `models.GRUModel`,
  tokenisation, batch-wise zero padding, 80/20 split, Adam `lr=1e-3`, MSE loss,
  batch size 100, seed 42) but **not a like-for-like reproduction of the original
  training regime** (12 epochs vs. the original 20/40/60 sweep; single
  hyper-parameter setting vs. the 36-model grid; `drop_last=False` on the test
  loader vs. the original `drop_last=True`).
* **No evidence of train/test leakage or a broken pipeline.** Metrics and parity
  plots are internally consistent and plausible for a 4,000-sample, 12-epoch
  subset.
* **PLD being worst (R²≈0.46) is believable**, not a bug: PLD is a bottleneck
  (narrowest-constriction) descriptor and is intrinsically harder than the more
  global Density/LCD/CH4ABL descriptors; its parity plot shows tail under-fit and
  slight overfitting beginning at epoch 12.
* **Highest-value follow-ups:** confirm index `0` is reserved for padding only,
  report the row counts removed by NaN/empty filtering, rerun the 20/40/60 epoch
  ladder (especially the 40-epoch model `test.py` expects), add a
  validation-based early-stopping checkpoint, optionally standardise/log-transform
  the targets (train-set statistics only), and report mean ± SD across several
  seeds.
