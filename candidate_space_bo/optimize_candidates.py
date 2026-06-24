"""Bayesian optimization over a *predefined candidate space* of MOFs.

This is the "featurization" alternative to generative latent-space optimization
(see ``docs/candidate-space-optimization.md``). Instead of optimizing a free
continuous vector that has to be *decoded* back into a (possibly invalid) MOF,
we:

1. Take a **fixed library of real MOFs** as the search space (here, the
   ``dataset/mof_output.csv`` candidates that ship with MOF-GRU; any enumerated
   pool of synthesizable MOFs works).
2. **Featurize** every candidate into a continuous vector. Two featurizers are
   supported:
     * ``descriptors`` -- hand-engineered structural descriptors (pore geometry,
       density, porosity, ...) already present in the dataset.
     * ``gru`` -- the learned MOF-GRU embedding, i.e. the pooled bidirectional
       hidden state from ``GRUModel.get_hidden_layer_output`` (``models.py``),
       computed from the SELFIES "sentence" of each MOF. This reuses the exact
       representation MOF-GRU is already trained on.
3. Run **Bayesian optimization over the discrete pool**: fit a Gaussian-process
   surrogate on the featurized, already-evaluated candidates and use an
   Expected-Improvement acquisition function to choose which *existing* candidate
   to evaluate (label) next.

Because every proposal is an entry in the library, every proposal is a real,
valid MOF -- there is no decoder, no invertibility requirement, and no
off-manifold/validity problem. This mirrors the Honegumi/Ax "featurization over
a candidate set" tutorial
(https://honegumi.readthedocs.io/en/latest/curriculum/tutorials/featurization/featurization.html).

The script is dependency-light by default (numpy + scikit-learn + matplotlib).
The ``gru`` featurizer additionally needs ``torch`` and a trained checkpoint and
reuses ``models.py``/``utils.py`` from this repository.

Example
-------
    python candidate_space_bo/optimize_candidates.py \
        --csv dataset/mof_output.csv --objective CH4ABL \
        --featurizer descriptors --n-candidates 0 --iters 100 --seeds 8

``--n-candidates 0`` uses the full candidate library (all 113,160 MOFs);
pass a positive value to subsample for a quicker run.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Structural descriptors used by the ``descriptors`` featurizer. These describe
# pore geometry / density and never include the optimization target (which would
# leak the answer). Columns missing from a given CSV are silently skipped.
DESCRIPTOR_COLUMNS = [
    "UC_volume",
    "Density",
    "ASA",
    "vASA",
    "gASA",
    "GCD",
    "Porosity",
    "PV",
    "PLD",
    "LCD",
]


def _open_csv(csv_path: Path):
    """Yield rows from ``csv_path``; falls back to bundled zip archives.

    The repository ships the large CSVs zipped, so if the plain CSV is absent we
    transparently read it out of the sibling ``<name>.zip`` (e.g.
    ``dataset/mof_output.zip``) or, failing that, the whole-repo ``MOF-GRU.zip``
    snapshot.
    """
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            yield from csv.DictReader(f)
        return

    # 1) Sibling archive shipped next to the CSV, e.g. dataset/mof_output.zip.
    sibling_zip = csv_path.with_suffix(".zip")
    if sibling_zip.exists():
        with zipfile.ZipFile(sibling_zip) as zf:
            with zf.open(csv_path.name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                yield from csv.DictReader(text)
        return

    # 2) Whole-repository snapshot bundled at the repo root.
    zip_path = REPO_ROOT / "MOF-GRU.zip"
    inner = f"MOF-GRU/dataset/{csv_path.name}"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(inner) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                yield from csv.DictReader(text)
        return

    raise FileNotFoundError(
        f"Could not find {csv_path}, {sibling_zip}, or {inner} inside {zip_path}."
    )


def load_pool(csv_path: Path, objective: str, n_candidates: int, rng: np.random.Generator):
    """Load the candidate library and the objective values.

    Returns ``(names, descriptor_matrix, descriptor_names, y)`` for the (optionally
    subsampled) pool of candidates that have a finite objective value.
    """
    names: list[str] = []
    rows: list[dict] = []
    for row in _open_csv(csv_path):
        val = row.get(objective, "")
        try:
            y = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(y):
            continue
        names.append(row.get("Name", str(len(names))))
        rows.append(row)

    if not rows:
        raise ValueError(f"No rows with a numeric '{objective}' value were found.")

    descriptor_names = [c for c in DESCRIPTOR_COLUMNS if c in rows[0] and c != objective]
    desc = np.full((len(rows), len(descriptor_names)), np.nan)
    for i, row in enumerate(rows):
        for j, col in enumerate(descriptor_names):
            try:
                desc[i, j] = float(row[col])
            except (TypeError, ValueError):
                pass
    y = np.array([float(r[objective]) for r in rows])

    # Keep only fully-featurized candidates so the GP sees clean inputs.
    keep = np.all(np.isfinite(desc), axis=1)
    names = [n for n, k in zip(names, keep) if k]
    desc, y = desc[keep], y[keep]

    if n_candidates and len(y) > n_candidates:
        idx = rng.choice(len(y), size=n_candidates, replace=False)
        names = [names[i] for i in idx]
        desc, y = desc[idx], y[idx]
    return names, desc, descriptor_names, y


def gru_features(names, csv_path: Path, checkpoint: Path):
    """Featurize candidates with the MOF-GRU pooled hidden state.

    Reuses ``models.py``/``utils.py``: tokenizes each MOF sentence with the same
    vocabulary and returns ``GRUModel.get_hidden_layer_output`` (a ``2*hidden``
    embedding). Requires ``torch`` and a trained checkpoint.
    """
    import json
    import sys

    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from utils import collate_fn  # noqa: E402  (repo module)

    dict_path = REPO_ROOT / "dataset" / "my_dict_output.json"
    with dict_path.open() as f:
        symbol2idx = json.load(f)["symbol2idx"]

    # Map MOF name -> tokenized sentence by re-reading the linker/topology columns.
    name_to_tokens: dict[str, list[int]] = {}
    for row in _open_csv(csv_path):
        import selfies as sf

        e1 = list(sf.split_selfies(row["linker_1"]))
        e2 = list(sf.split_selfies(row["linker_2"]))
        sentence = (
            ["<T>", row["topo"], row["cat"], "<N>", row["node"], "<E_1>"]
            + e1
            + ["<E_2>"]
            + e2
        )
        try:
            name_to_tokens[row["Name"]] = [symbol2idx[w] for w in sentence]
        except KeyError:
            continue

    model = torch.load(checkpoint, map_location="cpu")
    model.eval()

    feats = []
    batch = []
    order = []
    for name in names:
        toks = name_to_tokens.get(name)
        if toks is None:
            continue
        batch.append((torch.tensor(toks, dtype=torch.long), 0.0))
        order.append(name)
        if len(batch) == 128:
            x, _ = collate_fn(batch)
            with torch.no_grad():
                feats.append(model.get_hidden_layer_output(x).cpu().numpy())
            batch = []
    if batch:
        x, _ = collate_fn(batch)
        with torch.no_grad():
            feats.append(model.get_hidden_layer_output(x).cpu().numpy())
    return order, np.concatenate(feats, axis=0)


def expected_improvement(mu, sigma, best, xi=0.01):
    """EI for maximization (closed form, vectorized over the candidate pool)."""
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def run_bo(features, y, iters, n_init, seed):
    """Discrete-candidate BO loop: each step labels one *existing* candidate.

    Returns the best-objective-so-far trace over evaluations.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(features)
    x = scaler.transform(features)

    n = len(y)
    observed = list(rng.choice(n, size=n_init, replace=False))
    remaining = [i for i in range(n) if i not in set(observed)]
    trace = [float(np.max(y[observed]))]

    for _ in range(iters):
        if not remaining:
            break
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * Matern(length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-1, 1e3), nu=2.5)
            + WhiteKernel(1e-2, (1e-4, 1e1))
        )
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x[observed], y[observed])
        mu, sigma = gp.predict(x[remaining], return_std=True)
        ei = expected_improvement(mu, sigma, best=np.max(y[observed]))
        pick = remaining[int(np.argmax(ei))]
        observed.append(pick)
        remaining.remove(pick)
        trace.append(float(np.max(y[observed])))
    return np.array(trace)


def run_random(y, iters, n_init, seed):
    """Random-search baseline over the same discrete pool."""
    rng = np.random.default_rng(seed + 10_000)
    n = len(y)
    chosen = list(rng.choice(n, size=n_init, replace=False))
    remaining = [i for i in range(n) if i not in set(chosen)]
    rng.shuffle(remaining)
    trace = [float(np.max(y[chosen]))]
    for k in range(iters):
        if k >= len(remaining):
            break
        chosen.append(remaining[k])
        trace.append(float(np.max(y[chosen])))
    return np.array(trace)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="dataset/mof_output.csv")
    p.add_argument("--objective", default="CH4ABL", help="Property column to maximize.")
    p.add_argument("--featurizer", choices=["descriptors", "gru"], default="descriptors")
    p.add_argument("--checkpoint", default=None, help="Trained GRUModel .pth (gru featurizer).")
    p.add_argument("--n-candidates", type=int, default=4000,
                   help="Candidate pool size; 0 uses the full library (no subsampling).")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="candidate_space_bo/bo_trace.png")
    args = p.parse_args()

    csv_path = (REPO_ROOT / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    rng = np.random.default_rng(0)
    names, desc, desc_names, y = load_pool(csv_path, args.objective, args.n_candidates, rng)

    if args.featurizer == "gru":
        if not args.checkpoint:
            raise SystemExit("--featurizer gru requires --checkpoint <trained .pth>")
        names, features = gru_features(names, csv_path, Path(args.checkpoint))
        # Re-align y to the (possibly reduced) GRU-featurized subset.
        name_to_y = dict(zip(names, y)) if len(names) == len(y) else None
        if name_to_y is None:
            raise SystemExit("GRU featurization changed candidate ordering; rerun with descriptors.")
        feat_label = f"MOF-GRU embedding ({features.shape[1]}-d)"
    else:
        features = desc
        feat_label = f"structural descriptors ({', '.join(desc_names)})"

    print(f"Candidates: {len(y)} | features: {features.shape[1]} ({feat_label})")
    print(f"Objective '{args.objective}': min={y.min():.3g} max={y.max():.3g} mean={y.mean():.3g}")

    bo_traces, rand_traces = [], []
    for s in range(args.seeds):
        bo_traces.append(run_bo(features, y, args.iters, args.n_init, seed=s))
        rand_traces.append(run_random(y, args.iters, args.n_init, seed=s))
    L = min(min(len(t) for t in bo_traces), min(len(t) for t in rand_traces))
    bo = np.vstack([t[:L] for t in bo_traces])
    rd = np.vstack([t[:L] for t in rand_traces])

    best = y.max()
    evals = np.arange(L)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for mat, color, label in ((bo, "tab:red", "BO (GP + EI over candidate pool)"),
                              (rd, "tab:blue", "Random search over candidate pool")):
        m = mat.mean(axis=0)
        sd = mat.std(axis=0)
        ax.plot(evals, m, color=color, lw=2, label=label)
        ax.fill_between(evals, m - sd, m + sd, color=color, alpha=0.18)
    ax.axhline(best, ls="--", c="k", lw=1, label=f"pool optimum ({best:.3g})")
    ax.set_xlabel("Candidates evaluated (after random init)")
    ax.set_ylabel(f"Best {args.objective} found so far")
    ax.set_title(f"Discrete-candidate BO over predefined MOF library\nfeaturizer: {args.featurizer}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out_path = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    fig.savefig(out_path, dpi=150)
    final_bo = bo.mean(axis=0)[-1]
    final_rd = rd.mean(axis=0)[-1]
    print(f"After {L - 1} evaluations: BO best={final_bo:.4g}, random best={final_rd:.4g}, "
          f"pool optimum={best:.4g}")
    print("Saved", out_path)


if __name__ == "__main__":
    main()
