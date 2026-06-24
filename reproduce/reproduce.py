"""Cross-platform reproduction of the MOF-GRU training/evaluation pipeline.

The original ``training.py``/``test.py``/``utils.py`` use hard-coded Windows
paths (``dataset\\...``) and a large hyper-parameter grid, which makes the
project hard to run directly on Linux/CI.  This script wraps the *same* model
(``models.GRUModel``) and the *same* data files in a small, path-safe, and
parameterised driver so the results can be reproduced on any platform.

Examples
--------
Smoke test on 2000 rows for a single property::

    python reproduce/reproduce.py --properties gASA --limit 2000 --epochs 5

Train several properties on a subset and write parity plots + metrics::

    python reproduce/reproduce.py --properties gASA Density PLD LCD \
        --limit 20000 --epochs 20 --outdir reproduce/results
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset

# Make ``import models`` work regardless of the current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from models import GRUModel  # noqa: E402

DATASET_DIR = os.path.join(REPO_ROOT, "dataset")


def _resolve(*candidates):
    """Return the first existing path among ``candidates``."""
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "None of the candidate files exist: " + ", ".join(candidates)
    )


class MOFDataset(Dataset):
    """Path-safe re-implementation of ``utils.MyDataset``.

    Reads the tokenised MOF sequences, maps tokens to indices via the saved
    dictionary, and pairs each sequence with the requested target property.
    """

    def __init__(self, prop, limit=None):
        seq_path = _resolve(
            os.path.join(DATASET_DIR, "MOFseq_output.txt"),
            os.path.join(DATASET_DIR, "MOFseq_test.txt"),
        )
        csv_path = os.path.join(DATASET_DIR, "mof_output.csv")
        dict_path = os.path.join(DATASET_DIR, "my_dict_output.json")

        with open(seq_path, "r") as f:
            mof_list = [line.strip() for line in f]
        with open(dict_path) as f:
            symbol2idx = json.load(f)["symbol2idx"]

        df = pd.read_csv(csv_path)
        targets = df[prop].tolist()

        n = min(len(mof_list), len(targets))
        if limit is not None:
            n = min(n, limit)

        self.symbol2idx = symbol2idx
        self.MOF_seq = []
        self.MOF_P = []
        for i in range(n):
            tokens = mof_list[i].split()
            if not tokens:
                continue
            value = targets[i]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            self.MOF_seq.append([symbol2idx[t] for t in tokens])
            self.MOF_P.append(float(value))

    def __len__(self):
        return len(self.MOF_seq)

    def __getitem__(self, index):
        seq = torch.tensor(self.MOF_seq[index], dtype=torch.long)
        target = torch.tensor(self.MOF_P[index], dtype=torch.float)
        return seq, target

    def get_vocab_size(self):
        return len(self.symbol2idx)


def collate_fn(batch):
    max_len = max(len(seq) for seq, _ in batch)
    padded = []
    labels = []
    for seq, label in batch:
        buf = torch.zeros(max_len).long()
        buf[: len(seq)] = seq
        padded.append(buf)
        labels.append(label)
    return torch.stack(padded), torch.tensor(labels)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(-1)
            preds.extend(outputs.cpu().numpy().flatten().tolist())
            trues.extend(targets.numpy().flatten().tolist())
    return np.array(trues), np.array(preds)


def parity_plot(train, test, prop, metrics, out_path):
    plt.figure(figsize=(7, 7))
    lo = min(train[0].min(), test[0].min())
    hi = max(train[0].max(), test[0].max())
    plt.scatter(train[0], train[1], s=12, c="royalblue", alpha=0.5, label="Train")
    plt.scatter(test[0], test[1], s=12, c="red", alpha=0.6, label="Test")
    plt.plot([lo, hi], [lo, hi], "k-", linewidth=2)
    plt.xlabel(f"True {prop}")
    plt.ylabel(f"Predicted {prop}")
    plt.title(
        f"MOF-GRU: {prop}\n"
        f"Test R2={metrics['test_r2']:.3f}  "
        f"MAE={metrics['test_mae']:.3g}  "
        f"SRCC={metrics['test_srcc']:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_property(prop, args, device):
    print(f"\n=== Property: {prop} ===")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = MOFDataset(prop, limit=args.limit)
    print(f"Loaded {len(dataset)} samples (vocab={dataset.get_vocab_size()})")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn,
    )

    model = GRUModel(
        dataset.get_vocab_size() + 1, args.embedding_size,
        args.hidden_size, args.num_layers,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total, steps = 0.0, 0
        for batch, target in train_loader:
            batch = batch.to(device)
            target = target.to(device)
            pred = model(batch).squeeze(-1)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        avg = total / max(steps, 1)
        history.append(avg)
        print(f"  epoch {epoch + 1}/{args.epochs}  train_mse={avg:.4f}")
    elapsed = time.time() - start

    train_eval = evaluate(model, train_loader, device)
    test_eval = evaluate(model, test_loader, device)

    metrics = {
        "property": prop,
        "n_samples": len(dataset),
        "n_train": train_size,
        "n_test": test_size,
        "epochs": args.epochs,
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "train_seconds": round(elapsed, 2),
        "train_r2": float(r2_score(*train_eval)),
        "test_r2": float(r2_score(*test_eval)),
        "train_mae": float(mean_absolute_error(*train_eval)),
        "test_mae": float(mean_absolute_error(*test_eval)),
        "train_srcc": float(spearmanr(*train_eval).statistic),
        "test_srcc": float(spearmanr(*test_eval).statistic),
        "loss_history": history,
    }
    print(
        f"  -> test R2={metrics['test_r2']:.3f} "
        f"MAE={metrics['test_mae']:.3g} "
        f"SRCC={metrics['test_srcc']:.3f} "
        f"({elapsed:.1f}s)"
    )

    os.makedirs(args.outdir, exist_ok=True)
    parity_plot(
        train_eval, test_eval, prop, metrics,
        os.path.join(args.outdir, f"parity_{prop}.png"),
    )
    if args.save_model:
        torch.save(
            model.state_dict(),
            os.path.join(args.outdir, f"biGRU_{prop}.pt"),
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--properties", nargs="+",
        default=["gASA", "Density", "PLD", "LCD"],
        help="Target columns from mof_output.csv to model.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N rows (for quick runs).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--embedding-size", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=os.path.join("reproduce", "results"))
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (threads={torch.get_num_threads()})")

    os.makedirs(args.outdir, exist_ok=True)
    all_metrics = []
    for prop in args.properties:
        all_metrics.append(run_property(prop, args, device))

    summary_path = os.path.join(args.outdir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nWrote metrics to {summary_path}")


if __name__ == "__main__":
    main()
