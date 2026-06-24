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
# structural-descriptor featurizer (no training, reads dataset/mof_output.csv
# or MOF-GRU.zip automatically)
python candidate_space_bo/optimize_candidates.py \
    --objective CH4ABL --featurizer descriptors \
    --n-candidates 6000 --iters 60 --seeds 5

# learned MOF-GRU-embedding featurizer (needs torch + a trained checkpoint)
python candidate_space_bo/optimize_candidates.py \
    --objective CH4ABL --featurizer gru \
    --checkpoint my_models/new/biGRU_CH4ABL_model_ep_40_em_80_hd200.pth
```

Outputs `bo_trace.png` comparing GP+Expected-Improvement BO against random search
over the same candidate pool.

![BO trace](bo_trace.png)

## Dependencies

- `descriptors` featurizer: `numpy`, `scikit-learn`, `matplotlib` (`scipy` comes
  with scikit-learn).
- `gru` featurizer: additionally `torch` and `selfies`; reuses `models.py` /
  `utils.py` from the repository root.
