"""Dispatch a follow-up Edison query for the MOF-GRU surrogate study (issue #3).

Context from the PR: over the *raw* 400-d MOF-GRU pooled hidden state, a
stationary Matern GP barely beats random while a deep-ensemble GNN head wins.
The follow-up question on the PR asks for advice on improving a GP/surrogate over
the learned embedding -- specifically dimensionality reduction (PCA, supervised
reductions), concatenating hand-engineered descriptors with the embedding,
retraining the encoder with a smaller embedding dimension, and training a VAE or
other ML-based embedding -- and then implementing the recommendation.

This script sends a single high-effort LITERATURE query asking for concrete,
well-cited guidance. Task IDs are appended to ``edison/tasks.json`` under the
``lit_followup`` key so ``fetch_results.py`` can retrieve the trajectory later.

Auth: ``EDISON_PLATFORM_API_KEY`` (fallback ``EDISON_API_KEY``), ``.strip()``-ed.
Endpoint: https://api.platform.edisonscientific.com
"""

import json
import os
import time
from pathlib import Path

from edison_client import EdisonClient, JobNames, TaskRequest

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "edison" / "tasks.json"

LIT_QUERY = (
    "I am doing Bayesian optimization / active learning over a *predefined candidate space* of "
    "metal-organic frameworks (MOFs). Each candidate is featurized with a frozen, pretrained "
    "bidirectional-GRU sequence encoder (MOF-GRU): the pooled hidden state is a 400-dimensional "
    "embedding (2 x hidden_size=200). Empirically, with a stationary Matern Gaussian process (with ARD "
    "length scales) fit on this *raw 400-d* embedding, EI-driven candidate selection barely beats random "
    "search, whereas a small deep-ensemble neural network head (the GRU's own Linear->ReLU->Linear "
    "regression head, retrained each round) used as the surrogate clearly wins. I want a high-effort, "
    "well-cited answer on how to make a *GP* (or a better-calibrated probabilistic surrogate) competitive "
    "over such a learned embedding, and more generally how to get a better optimization-ready embedding. "
    "Cover concretely: (1) Dimensionality reduction of the learned embedding before GP modeling -- when "
    "unsupervised PCA helps vs. not (my intuition is that the 400 GRU hidden units may carry roughly "
    "equal importance so PCA wouldn't compress much; is that right, and how to check via the PCA "
    "explained-variance spectrum?), and *supervised* alternatives (PLS regression, supervised PCA, "
    "linear discriminant / Fisher directions, neighborhood-components / metric learning, or learning a "
    "low-rank projection jointly with the property). (2) Why high-dimensional stationary GPs with ARD are "
    "hard to fit from tens of labels (curse of dimensionality, non-identifiable length scales) and the "
    "standard fixes: deep-kernel learning (DKL, Wilson et al.), GP on a learned feature extractor, "
    "additive/low-dimensional-structure GPs (ADD-GP, SAASBO sparse-axis-aligned priors, Eriksson & "
    "Jankowiak), random-feature / linear surrogates, and Bayesian neural networks / deep ensembles as "
    "drop-in probabilistic surrogates (compare calibration & sample efficiency to GPs). (3) Combining the "
    "learned embedding with hand-engineered global descriptors (pore geometry, density, porosity, surface "
    "area): concatenation, separate kernels (sum/product of kernels over each feature block), and feature "
    "selection. (4) Retraining the encoder with a *smaller* bottleneck embedding dimension (e.g. 8-64) so "
    "the surrogate lives in a compact space -- trade-offs vs. the full 400-d representation, and whether a "
    "bottleneck regularizes the optimization landscape. (5) Training a VAE or other ML-based embedding "
    "(beta-VAE, sequence VAE over the SELFIES token vocabulary, contrastive/SimCLR-style or self-supervised "
    "embeddings, or a jointly trained supervised autoencoder) to get a smooth, low-dimensional, "
    "optimization-friendly latent space, and how that compares to just reducing the existing embedding. "
    "(6) Practical recommendations and a ranked, actionable plan specifically for improving GP-based "
    "candidate-space BO over a frozen sequence-model embedding when only tens-to-hundreds of labels are "
    "available. Provide citations throughout."
)


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def main() -> None:
    key = (
        os.environ.get("EDISON_PLATFORM_API_KEY")
        or os.environ.get("EDISON_API_KEY")
        or ""
    ).strip()
    if not key:
        raise SystemExit("Set EDISON_PLATFORM_API_KEY (or EDISON_API_KEY).")

    client = EdisonClient(api_key=key)
    state = _load_state()

    if "lit_followup" not in state:
        task = TaskRequest(name=JobNames.LITERATURE_HIGH, query=LIT_QUERY)
        state["lit_followup"] = str(client.create_task(task))
        _save_state(state)
        print("LITERATURE_HIGH (followup) task:", state["lit_followup"])
        time.sleep(10)
    else:
        print("LITERATURE_HIGH (followup) already dispatched:", state["lit_followup"])

    print("State written to", STATE_PATH)


if __name__ == "__main__":
    main()
