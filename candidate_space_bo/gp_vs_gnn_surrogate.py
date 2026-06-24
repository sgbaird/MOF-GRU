"""GP vs. GNN as the *surrogate model* for candidate-space optimization.

This answers a specific question raised on issue #3:

    "How does a GP vs. the GNN fare as the surrogate model -- i.e., a GP over
     the GNN embedding space vs. the GNN with GNN predictions?"

Both strategies optimize over the **same predefined candidate space** (the fixed
MOF library) and share the **same frozen MOF-GRU encoder**: every candidate is
turned into the 400-d pooled bidirectional hidden state
(``GRUModel.get_hidden_layer_output`` / ``models.py``) exactly once. They differ
only in the *surrogate* used to decide which unlabeled candidate to evaluate
next:

* ``gp``  -- a **Gaussian process** is fit on the embeddings of the already
  labeled candidates and an Expected-Improvement acquisition selects the next
  one. The GNN is used only as a featurizer; the GP supplies the predictive mean
  *and* the calibrated uncertainty that BO needs.

* ``gnn`` -- the **GNN's own neural predictor is the surrogate**. We rebuild the
  MOF-GRU regression head (``Linear(D, hidden) -> ReLU -> Linear(hidden, 1)``,
  the exact ``fc_1``/``fc_2`` architecture) on top of the frozen embeddings and
  **train it from scratch on the observed labels each round**. Epistemic
  uncertainty comes from a small **deep ensemble** of these heads, so the same
  Expected-Improvement acquisition can be used. This is "the GNN making the
  predictions" inside the loop.

* ``gnn-pretrained`` -- a reference line that ranks candidates purely by the
  *pretrained* MOF-GRU's end-to-end predictions (greedy exploitation, no
  re-fitting). It is the literal "GNN with GNN predictions" but is an
  **in-sample / leaky upper bound**: the shipped checkpoint was trained on these
  very MOFs (r >= 0.98 in-sample), so it is not an honest active-learning
  surrogate -- it is included only to show the ceiling.

* ``random`` -- random search over the same pool (lower baseline).

A fair active-learning surrogate must *learn from the labels revealed so far*;
that is why the live ``gnn`` strategy retrains from scratch rather than reusing
the leaky pretrained head. All strategies share identical random initialisations
per seed so the comparison is apples-to-apples.

Example
-------
    python candidate_space_bo/gp_vs_gnn_surrogate.py \
        --objective CH4ABL \
        --checkpoint my_models/new/biGRU_CH4ABL_model_ep_40_em_80_hd200.pth \
        --n-candidates 4000 --iters 40 --seeds 5

Use ``--n-candidates 0`` to run over the full library (all candidates).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from optimize_candidates import (  # noqa: E402 - sibling module
    REPO_ROOT,
    expected_improvement,
    gru_features,
    load_pool,
)


# --------------------------------------------------------------------------- #
# Surrogates
# --------------------------------------------------------------------------- #
def _fit_gp(x_obs, y_obs):
    """Gaussian-process regressor over the (standardized) GNN embeddings."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.exceptions import ConvergenceWarning

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=np.ones(x_obs.shape[1]), length_scale_bounds=(1e-1, 1e3), nu=2.5)
        + WhiteKernel(1e-2, (1e-4, 1e1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x_obs, y_obs)
    return gp


def _gp_predict(gp, x_pool):
    mu, sigma = gp.predict(x_pool, return_std=True)
    return mu, sigma


def _gnn_head_ensemble(x_obs, y_obs, x_pool, hidden, n_models, epochs, seed):
    """Deep ensemble of MOF-GRU regression heads trained on the observed labels.

    Each head reproduces the network's own predictor (``Linear(D, hidden) ->
    ReLU -> Linear(hidden, 1)``) and is trained from scratch on the embeddings of
    the labeled candidates. The ensemble mean/std over the pool give the
    predictive mean and epistemic uncertainty used by the acquisition function.
    """
    import torch
    import torch.nn as nn

    y_mean, y_std = float(y_obs.mean()), float(y_obs.std() + 1e-8)
    xt = torch.tensor(x_obs, dtype=torch.float32)
    yt = torch.tensor((y_obs - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
    xp = torch.tensor(x_pool, dtype=torch.float32)

    preds = []
    for k in range(n_models):
        torch.manual_seed(seed * 100 + k)
        net = nn.Sequential(
            nn.Linear(x_obs.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(net(xt), yt)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(xp).squeeze(1).numpy() * y_std + y_mean)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


# --------------------------------------------------------------------------- #
# Active-learning harness (shared by every strategy)
# --------------------------------------------------------------------------- #
def run_strategy(strategy, features, y, preds_pretrained, iters, n_init, seed, args):
    """One best-so-far trace; each step reveals the label of one pool candidate."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = len(y)
    observed = list(rng.choice(n, size=n_init, replace=False))
    obs_set = set(observed)
    remaining = [i for i in range(n) if i not in obs_set]
    trace = [float(np.max(y[observed]))]

    scaler = StandardScaler().fit(features)
    x = scaler.transform(features)

    for _ in range(iters):
        if not remaining:
            break
        rem = np.array(remaining)

        if strategy == "random":
            pick_pos = int(rng.integers(len(rem)))
        elif strategy == "gnn-pretrained":
            # Static greedy ranking by the pretrained network's own predictions.
            pick_pos = int(np.argmax(preds_pretrained[rem]))
        else:
            best = float(np.max(y[observed]))
            if strategy == "gp":
                gp = _fit_gp(x[observed], y[observed])
                mu, sigma = _gp_predict(gp, x[rem])
            elif strategy == "gnn":
                mu, sigma = _gnn_head_ensemble(
                    x[observed], y[observed], x[rem],
                    hidden=args.gnn_hidden, n_models=args.gnn_ensemble,
                    epochs=args.gnn_epochs, seed=seed,
                )
            else:  # pragma: no cover - guarded by argparse choices
                raise ValueError(strategy)
            ei = expected_improvement(mu, sigma, best=best)
            pick_pos = int(np.argmax(ei))

        pick = int(rem[pick_pos])
        observed.append(pick)
        remaining.remove(pick)
        trace.append(float(np.max(y[observed])))
    return np.array(trace)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="dataset/mof_output.csv")
    p.add_argument("--objective", default="CH4ABL", help="Property column to maximize.")
    p.add_argument("--checkpoint", default="my_models/new/biGRU_CH4ABL_model_ep_40_em_80_hd200.pth",
                   help="Trained GRUModel .pth used as the frozen encoder / GNN predictor.")
    p.add_argument("--n-candidates", type=int, default=4000,
                   help="Candidate pool size; 0 uses the full library (no subsampling).")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--gnn-ensemble", type=int, default=5, help="Heads in the GNN deep ensemble.")
    p.add_argument("--gnn-epochs", type=int, default=200, help="Epochs per head per round.")
    p.add_argument("--gnn-hidden", type=int, default=200, help="Hidden width of the GNN predictor head.")
    p.add_argument("--strategies", default="gp,gnn,gnn-pretrained,random",
                   help="Comma-separated subset of gp,gnn,gnn-pretrained,random.")
    p.add_argument("--out", default="candidate_space_bo/gp_vs_gnn_trace.png")
    args = p.parse_args()

    csv_path = (REPO_ROOT / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    ckpt = (REPO_ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    rng = np.random.default_rng(0)
    names, _, _, y = load_pool(csv_path, args.objective, args.n_candidates, rng)

    print(f"Computing MOF-GRU embeddings for {len(names)} candidates (frozen encoder)...")
    order, features, preds_pretrained = gru_features(names, csv_path, ckpt, return_preds=True)
    # Re-align y to the (possibly reduced) GRU-featurized subset, by name.
    name_to_y = dict(zip(names, y))
    y = np.array([name_to_y[n] for n in order])
    print(f"Embeddings: {features.shape} | objective '{args.objective}': "
          f"min={y.min():.3g} max={y.max():.3g} mean={y.mean():.3g}")
    # Sanity check on the pretrained predictor (in-sample correlation).
    r = float(np.corrcoef(preds_pretrained, y)[0, 1])
    print(f"Pretrained GNN in-sample Pearson r = {r:.3f} (used only for the leaky reference line)")

    traces = {s: [] for s in strategies}
    for s in strategies:
        for seed in range(args.seeds):
            traces[s].append(run_strategy(s, features, y, preds_pretrained,
                                          args.iters, args.n_init, seed, args))
        print(f"  [{s}] done")

    L = min(min(len(t) for t in traces[s]) for s in strategies)
    best = float(y.max())
    evals = np.arange(L)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "gp": ("tab:red", "GP over GNN embeddings (EI)"),
        "gnn": ("tab:green", "GNN predictor surrogate, deep ensemble (EI)"),
        "gnn-pretrained": ("tab:purple", "Pretrained GNN predictions, greedy (leaky upper bound)"),
        "random": ("tab:blue", "Random search"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    summary = {}
    for s in strategies:
        mat = np.vstack([t[:L] for t in traces[s]])
        m, sd = mat.mean(axis=0), mat.std(axis=0)
        color, label = style.get(s, ("tab:gray", s))
        ls = "--" if s == "gnn-pretrained" else "-"
        ax.plot(evals, m, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(evals, m - sd, m + sd, color=color, alpha=0.15)
        summary[s] = float(m[-1])
    ax.axhline(best, ls=":", c="k", lw=1, label=f"pool optimum ({best:.3g})")
    ax.set_xlabel("Candidates evaluated (after random init)")
    ax.set_ylabel(f"Best {args.objective} found so far")
    ax.set_title("Surrogate comparison over a predefined MOF candidate space\n"
                 "GP over GNN embeddings vs. GNN-prediction surrogate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.savefig(out_path, dpi=150)

    print(f"\nAfter {L - 1} evaluations (mean over {args.seeds} seeds), best {args.objective}:")
    for s in strategies:
        print(f"  {s:16s} {summary[s]:.4g}")
    print(f"  pool optimum     {best:.4g}")
    print("Saved", out_path)


if __name__ == "__main__":
    main()
