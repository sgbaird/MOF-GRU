"""Retrain a MOF-GRU encoder with a *small bottleneck embedding* dimension.

The shipped checkpoints use ``hidden_size=200`` so the pooled bidirectional
hidden state (``GRUModel.get_hidden_layer_output``) is 400-dimensional. A GP
surrogate struggles in 400-d (see ``embedding_dimensionality.py``). One of the
options raised on issue #3 was to *retrain the encoder with a smaller embedding*
so the surrogate lives in a compact space to begin with, instead of reducing a
400-d embedding after the fact.

This script trains the existing ``models.GRUModel`` with a small ``hidden_size``
(so the embedding is ``2*hidden_size``, e.g. 16 or 32-d) on one objective, then
saves a checkpoint that is a drop-in for ``gru_features`` (a pickled ``GRUModel``
instance, loadable with ``torch.load(..., weights_only=False)``). The compact
embedding can then be compared head-to-head with the 400-d embedding and its
PCA/PLS reductions in ``embedding_dimensionality.py`` via
``--extra-checkpoint``.

It is intentionally self-contained and cross-platform: it reads the zipped
``dataset/MOFseq_output.zip`` and ``dataset/mof_output.zip`` directly rather than
the hard-coded Windows paths in ``utils.MyDataset``.

Example
-------
    python candidate_space_bo/train_small_embedding.py \
        --objective CH4ABL --hidden 16 --epochs 8 --n-train 20000 \
        --out my_models/small/biGRU_CH4ABL_hd16.pth
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from models import GRUModel  # noqa: E402
from utils import collate_fn  # noqa: E402


def _read_zip_text(zip_path: Path, member: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as raw:
            return io.TextIOWrapper(raw, encoding="utf-8").read().splitlines()


def load_sequences_and_targets(objective: str):
    """Tokenized MOF sentences + objective values, aligned by row index."""
    seq_lines = _read_zip_text(REPO_ROOT / "dataset" / "MOFseq_output.zip", "MOFseq_output.txt")
    with (REPO_ROOT / "dataset" / "my_dict_output.json").open() as f:
        symbol2idx = json.load(f)["symbol2idx"]

    import csv
    with zipfile.ZipFile(REPO_ROOT / "dataset" / "mof_output.zip") as zf:
        with zf.open("mof_output.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            targets = []
            for row in reader:
                try:
                    targets.append(float(row[objective]))
                except (TypeError, ValueError, KeyError):
                    targets.append(np.nan)
    targets = np.asarray(targets, dtype=float)

    seqs = []
    for line in seq_lines:
        seqs.append([symbol2idx[w] for w in line.split()])
    n = min(len(seqs), len(targets))
    seqs, targets = seqs[:n], targets[:n]
    keep = np.isfinite(targets)
    seqs = [s for s, k in zip(seqs, keep) if k]
    targets = targets[keep]
    return seqs, targets, len(symbol2idx) + 1


class SeqDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs = seqs
        self.targets = targets

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.tensor(self.seqs[i], dtype=torch.long), torch.tensor(self.targets[i], dtype=torch.float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", default="CH4ABL")
    p.add_argument("--hidden", type=int, default=16, help="GRU hidden size; embedding is 2*hidden.")
    p.add_argument("--embedding-size", type=int, default=80)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train", type=int, default=20000,
                   help="Subsample this many MOFs for a fast CPU run; 0 uses all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    seqs, targets, vocab = load_sequences_and_targets(args.objective)
    rng = np.random.default_rng(args.seed)
    if args.n_train and len(seqs) > args.n_train:
        idx = rng.choice(len(seqs), size=args.n_train, replace=False)
        seqs = [seqs[i] for i in idx]
        targets = targets[idx]
    print(f"Training on {len(seqs)} MOFs | vocab {vocab} | objective {args.objective} "
          f"(hidden={args.hidden} -> {2*args.hidden}-d embedding)")

    # Standardize the target for stable training; the saved model predicts in the
    # standardized space, but get_hidden_layer_output (the embedding) is unaffected.
    y_mean, y_std = float(np.mean(targets)), float(np.std(targets) + 1e-8)
    targets_std = (targets - y_mean) / y_std

    n_val = max(1, int(0.1 * len(seqs)))
    perm = rng.permutation(len(seqs))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = SeqDataset([seqs[i] for i in tr_idx], targets_std[tr_idx])
    va = SeqDataset([seqs[i] for i in val_idx], targets_std[val_idx])
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GRUModel(vocab, args.embedding_size, args.hidden, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, yb in tr_loader:
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), yb)
            loss.backward()
            opt.step()
            tot += float(loss)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, yb in va_loader:
                preds.append(model(x).squeeze(-1).numpy())
                trues.append(yb.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        r = float(np.corrcoef(preds, trues)[0, 1]) if len(preds) > 1 else float("nan")
        print(f"  epoch {ep+1:02d}/{args.epochs}  train_mse={tot/max(1,len(tr_loader)):.4f}  val_r={r:.3f}")

    out = args.out or f"my_models/small/biGRU_{args.objective}_hd{args.hidden}.pth"
    out_path = (REPO_ROOT / out) if not Path(out).is_absolute() else Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save(model, out_path)
    print(f"Saved compact encoder ({2*args.hidden}-d embedding) to {out_path}")


if __name__ == "__main__":
    main()
