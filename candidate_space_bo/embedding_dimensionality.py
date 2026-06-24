"""Improving the *surrogate's representation* for candidate-space BO over MOFs.

Follow-up to ``gp_vs_gnn_surrogate.py``. There a stationary GP fit on the *raw*
400-d MOF-GRU embedding barely beat random, while a deep-ensemble neural head
won. The question raised on issue #3 was how to make a GP competitive over the
learned embedding -- e.g. dimensionality reduction, adding hand-engineered
descriptors, or a smaller learned embedding -- and whether unsupervised PCA helps
(intuition: the 400 GRU hidden units may carry roughly equal importance, so PCA
might not compress much).

This script holds the **same frozen MOF-GRU encoder** fixed and compares GP-EI
active learning over several *representations* of each candidate, plus the
GNN deep-ensemble surrogate and a random baseline:

* ``gp-raw``        -- GP on the raw 400-d embedding (the previous weak result).
* ``gp-pca{k}``     -- GP on an unsupervised PCA-``k`` projection of the
  embedding. PCA is label-free, so it can be fit transductively on the whole
  candidate pool once.
* ``gp-pls{k}``     -- GP on a *supervised* PLS-``k`` projection. PLS uses the
  labels, so it is **re-fit each round on the observed (embedding, y) pairs**
  only (no leakage from unlabeled candidates' objective values).
* ``gp-desc``       -- GP on the 10 hand-engineered structural descriptors.
* ``gp-desc+emb``   -- GP on standardized descriptors concatenated with the
  PCA-reduced embedding (descriptors + learned features together).
* ``gnn``           -- deep-ensemble MOF-GRU head over the raw embedding (the
  prior winner), for reference.
* ``random``        -- random search baseline.

It also writes the PCA explained-variance spectrum of the embedding, which
directly tests the "is the embedding low-rank?" intuition.

Example
-------
    python candidate_space_bo/embedding_dimensionality.py \
        --objective CH4ABL \
        --checkpoint my_models/new/biGRU_CH4ABL_model_ep_40_em_80_hd200.pth \
        --n-candidates 4000 --iters 40 --seeds 5

``--n-candidates 0`` uses the full library.
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
from gp_vs_gnn_surrogate import _fit_gp, _gp_predict, _gnn_head_ensemble  # noqa: E402


# --------------------------------------------------------------------------- #
# Representations / projectors
# --------------------------------------------------------------------------- #
def _standardize(train, *others):
    """Standardize using train statistics; apply to every matrix passed in."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1e-8
    return tuple((m - mu) / sd for m in (train, *others))


def project(strategy, emb, desc, observed, k):
    """Return (x_obs, x_pool_rem_indexable) standardized features for a strategy.

    ``emb``/``desc`` are the full-pool raw matrices. The returned ``x`` is a
    full-pool feature matrix already standardized on the *observed* rows; the
    harness indexes it by ``observed`` / ``remaining``. Supervised projectors
    (PLS) only ever see the labels of ``observed`` rows.
    """
    if strategy.startswith("gp-pca") or strategy == "gp-desc+emb":
        from sklearn.decomposition import PCA

        # Unsupervised: fit on the whole pool (label-free, transductive).
        z = PCA(n_components=k, random_state=0).fit_transform(_standardize(emb)[0])
        if strategy == "gp-desc+emb":
            x = np.hstack([_standardize(desc)[0], _standardize(z)[0]])
        else:
            x = z
    elif strategy == "gp-desc":
        x = desc
    else:  # gp-raw / gnn / fallback
        x = emb
    # Standardize on observed statistics so the GP sees zero-mean inputs.
    return _standardize(x[observed], x)[1]


def project_pls(emb, observed, y, k):
    """Supervised PLS projection refit on the observed labels only."""
    from sklearn.cross_decomposition import PLSRegression

    k = min(k, emb.shape[1], max(1, len(observed) - 1))
    pls = PLSRegression(n_components=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pls.fit(_standardize(emb[observed])[0], y[observed])
        z = pls.transform(_standardize(emb[observed], emb)[1])
    return _standardize(z[observed], z)[1]


# --------------------------------------------------------------------------- #
# Active-learning harness
# --------------------------------------------------------------------------- #
def run_strategy(strategy, emb, desc, y, iters, n_init, seed, args, emb_small=None):
    rng = np.random.default_rng(seed)
    n = len(y)
    observed = list(rng.choice(n, size=n_init, replace=False))
    remaining = [i for i in range(n) if i not in set(observed)]
    trace = [float(np.max(y[observed]))]

    for _ in range(iters):
        if not remaining:
            break
        rem = np.array(remaining)
        best = float(np.max(y[observed]))

        if strategy == "random":
            pick_pos = int(rng.integers(len(rem)))
        elif strategy == "gnn":
            xs = project("gp-raw", emb, desc, observed, args.k)
            mu, sigma = _gnn_head_ensemble(
                xs[observed], y[observed], xs[rem],
                hidden=args.gnn_hidden, n_models=args.gnn_ensemble,
                epochs=args.gnn_epochs, seed=seed,
            )
            ei = expected_improvement(mu, sigma, best=best)
            pick_pos = int(np.argmax(ei))
        else:
            if strategy.startswith("gp-pls"):
                xs = project_pls(emb, observed, y, args.k)
            elif strategy == "gp-small":
                xs = _standardize(emb_small[observed], emb_small)[1]
            else:
                xs = project(strategy, emb, desc, observed, args.k)
            gp = _fit_gp(xs[observed], y[observed], max_iter=args.gp_max_iter)
            mu, sigma = _gp_predict(gp, xs[rem])
            ei = expected_improvement(mu, sigma, best=best)
            pick_pos = int(np.argmax(ei))

        pick = int(rem[pick_pos])
        observed.append(pick)
        remaining.remove(pick)
        trace.append(float(np.max(y[observed])))
    return np.array(trace)


def _pca_spectrum_plot(emb, out_path):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = StandardScaler().fit_transform(emb)
    ev = PCA().fit(x).explained_variance_ratio_
    cum = np.cumsum(ev)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(cum) + 1), cum, lw=2, color="tab:purple")
    ax.axhline(0.9, ls="--", c="gray", lw=1)
    for frac, c in ((0.5, "tab:blue"), (0.9, "tab:red")):
        k = int(np.searchsorted(cum, frac) + 1)
        ax.axvline(k, ls=":", c=c, lw=1)
        ax.text(k + 3, 0.15, f"{int(frac*100)}% var\n@ {k} PCs", color=c, fontsize=8)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA spectrum of the 400-d MOF-GRU embedding\n"
                 "(flat spectrum => no dominant low-rank structure)")
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return [int(np.searchsorted(cum, f) + 1) for f in (0.5, 0.9)], cum


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="dataset/mof_output.csv")
    p.add_argument("--objective", default="CH4ABL")
    p.add_argument("--checkpoint", default="my_models/new/biGRU_CH4ABL_model_ep_40_em_80_hd200.pth")
    p.add_argument("--n-candidates", type=int, default=4000)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--k", type=int, default=20, help="Target dimensionality for PCA/PLS projections.")
    p.add_argument("--gp-max-iter", type=int, default=200,
                   help="Cap on GP hyperparameter L-BFGS iterations (keeps 400-d ARD fits tractable).")
    p.add_argument("--gnn-ensemble", type=int, default=5)
    p.add_argument("--gnn-epochs", type=int, default=200)
    p.add_argument("--gnn-hidden", type=int, default=200)
    p.add_argument("--strategies",
                   default="gp-raw,gp-pca,gp-pls,gp-desc,gp-desc+emb,gnn,random")
    p.add_argument("--extra-checkpoint", default=None,
                   help="Optional 2nd GRUModel .pth (e.g. a small-bottleneck encoder from "
                        "train_small_embedding.py); adds a 'gp-small' strategy over its embedding.")
    p.add_argument("--out", default="candidate_space_bo/embedding_dimensionality_trace.png")
    p.add_argument("--spectrum-out", default="candidate_space_bo/pca_spectrum.png")
    args = p.parse_args()

    csv_path = (REPO_ROOT / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    ckpt = (REPO_ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    rng = np.random.default_rng(0)
    names, desc_pool, desc_names, y = load_pool(csv_path, args.objective, args.n_candidates, rng)
    name_to_desc = dict(zip(names, desc_pool))
    name_to_y = dict(zip(names, y))

    print(f"Computing MOF-GRU embeddings for {len(names)} candidates (frozen encoder)...")
    order, emb = gru_features(names, csv_path, ckpt)
    emb = np.asarray(emb, dtype=float)
    desc = np.vstack([name_to_desc[n] for n in order])
    y = np.array([name_to_y[n] for n in order])
    print(f"Embedding: {emb.shape} | descriptors: {desc.shape} | "
          f"'{args.objective}' min={y.min():.3g} max={y.max():.3g}")

    spectrum_out = (REPO_ROOT / args.spectrum_out) if not Path(args.spectrum_out).is_absolute() else Path(args.spectrum_out)
    (k50, k90), _cum = _pca_spectrum_plot(emb, spectrum_out)
    print(f"PCA: 50% variance needs {k50} PCs, 90% needs {k90} PCs (of {emb.shape[1]}). Saved {spectrum_out}")

    emb_small = None
    if args.extra_checkpoint:
        extra = (REPO_ROOT / args.extra_checkpoint) if not Path(args.extra_checkpoint).is_absolute() else Path(args.extra_checkpoint)
        order2, emb_small_all = gru_features(order, csv_path, extra)
        small_map = dict(zip(order2, np.asarray(emb_small_all, dtype=float)))
        emb_small = np.vstack([small_map[n] for n in order])
        print(f"Compact embedding from {extra.name}: {emb_small.shape}")
        if "gp-small" not in strategies:
            strategies.append("gp-small")

    traces = {s: [] for s in strategies}
    for s in strategies:
        for seed in range(args.seeds):
            traces[s].append(run_strategy(s, emb, desc, y, args.iters, args.n_init, seed, args, emb_small=emb_small))
        print(f"  [{s}] done")

    L = min(min(len(t) for t in traces[s]) for s in strategies)
    best = float(y.max())
    evals = np.arange(L)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "gp-raw": ("tab:red", f"GP raw {emb.shape[1]}-d embedding"),
        "gp-pca": ("tab:orange", f"GP PCA-{args.k} embedding"),
        "gp-pls": ("tab:green", f"GP PLS-{args.k} (supervised) embedding"),
        "gp-desc": ("tab:brown", "GP descriptors only"),
        "gp-desc+emb": ("tab:olive", f"GP descriptors + PCA-{args.k} embedding"),
        "gnn": ("tab:purple", "GNN deep-ensemble head (raw embedding)"),
        "gp-small": ("tab:cyan", "GP small retrained embedding"),
        "random": ("tab:blue", "Random search"),
    }
    fig, ax = plt.subplots(figsize=(8, 5.5))
    summary = {}
    for s in strategies:
        mat = np.vstack([t[:L] for t in traces[s]])
        m, sd = mat.mean(axis=0), mat.std(axis=0)
        color, label = style.get(s, ("tab:gray", s))
        ax.plot(evals, m, color=color, lw=2, label=label)
        ax.fill_between(evals, m - sd, m + sd, color=color, alpha=0.12)
        summary[s] = float(m[-1])
    ax.axhline(best, ls=":", c="k", lw=1, label=f"pool optimum ({best:.3g})")
    ax.set_xlabel("Candidates evaluated (after random init)")
    ax.set_ylabel(f"Best {args.objective} found so far")
    ax.set_title("Surrogate representation comparison over a MOF candidate space\n"
                 "dimensionality reduction & descriptor fusion vs. raw embedding")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.savefig(out_path, dpi=150)

    print(f"\nAfter {L - 1} evaluations (mean over {args.seeds} seeds), best {args.objective}:")
    for s in sorted(summary, key=summary.get, reverse=True):
        print(f"  {s:14s} {summary[s]:.4g}")
    print(f"  pool optimum   {best:.4g}")
    print("Saved", out_path)


if __name__ == "__main__":
    main()
