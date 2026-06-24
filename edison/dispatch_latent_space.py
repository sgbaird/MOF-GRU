"""Dispatch Edison Scientific queries for MOF-GRU issue #3.

Sends:
  1. A high-effort LITERATURE query on optimizing over a learned latent space
     (VAE / GNN embeddings) instead of over global descriptor variables.
  2. An ANALYSIS query with the full MOF-GRU repository uploaded as a single
     zipped collection (per the file-management docs).

Task IDs are recorded to ``edison/tasks.json`` so the run is resumable and so
``fetch_results.py`` can retrieve the trajectories/artifacts later (possibly in
a subsequent session).

Auth: the API key is read from ``EDISON_PLATFORM_API_KEY`` (fallback
``EDISON_API_KEY``) and ``.strip()``-ed (the injected secret can carry a
trailing newline that otherwise yields a 403 on login).

Endpoint: https://api.platform.edisonscientific.com
"""

import json
import os
import time
from pathlib import Path

from edison_client import EdisonClient, JobNames, TaskRequest

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "edison" / "tasks.json"
# Files uploaded for the ANALYSIS task. The large redundant snapshot and VCS
# metadata are excluded; everything else in the repository is included.
UPLOAD_IGNORE = ["MOF-GRU.zip", ".git", ".git/**"]

LIT_QUERY = (
    "In the context of inverse design of metal-organic frameworks (MOFs), how can one "
    "optimize over a learned continuous latent space instead of over global/hand-engineered "
    "descriptor variables? Provide a high-effort, well-cited review covering: (1) Variational "
    "autoencoders (VAEs) over string/SELFIES/SMILES or graph representations that yield a direct, "
    "smooth latent space for Bayesian or gradient-based optimization (e.g., ChemVAE/Gomez-Bombarelli, "
    "junction-tree VAE, SELFIES-based VAEs, MOF-specific generative models such as SmVAE/Yao MOF "
    "generation, GHP-MOFassemble, and recent diffusion/transformer MOF generators); (2) using the "
    "hidden state / pooled embedding of a sequence model (e.g., a bidirectional GRU over a MOF "
    "'sentence' of topology+node+linker SELFIES tokens, as in MOF-GRU) as a latent space for "
    "optimization, and the pitfalls of optimizing a non-generative encoder's latent space "
    "(no decoder, holes/invalid regions, off-manifold extrapolation); (3) graph neural network "
    "latent embeddings of crystal/MOF structure and how to make them optimizable/invertible; "
    "(4) latent-space optimization methods: Bayesian optimization with Gaussian processes in latent "
    "space, latent-space BO with trust regions (e.g., LS-BO, LOL-BO, weighted retraining by "
    "Tripp et al.), and gradient ascent on a differentiable property predictor; (5) ensuring "
    "decodability/validity and constraining to the data manifold (jointly trained property "
    "predictors, semi-supervised VAEs, uncertainty/penalty terms). Compare these latent-space "
    "approaches to optimizing over global variables directly, and give concrete, actionable "
    "recommendations for the MOF-GRU codebase."
)

ANALYSIS_QUERY = (
    "Attached is the full MOF-GRU repository. It trains a bidirectional GRU (models.py: GRUModel) "
    "over MOF 'sentences' (topology, catenation, node, and SELFIES-tokenized linkers; see "
    "get_MOFseq.py and utils.py MyDataset) to regress global MOF properties (e.g., CH4ABL, N2ABL, "
    "PLD, LCD, Density, Porosity, ASA). The model exposes pooled hidden states via "
    "get_hidden_layer_output / get_fc_layer_output1. The question: how would one optimize over a "
    "latent space instead of over global descriptor variables? Analyze the code and: (1) explain "
    "where a latent space exists or could be added (GRU pooled hidden state vs. a true generative "
    "VAE with a decoder); (2) explain why the current encoder-only GRU hidden space is not directly "
    "invertible/decodable and what that means for optimization; (3) propose a concrete, minimal "
    "design to enable latent-space optimization - e.g., a sequence VAE over the same SELFIES/token "
    "vocabulary (reuse my_dict_output.json and MOFseq), a jointly trained property head, and "
    "latent-space Bayesian optimization / gradient ascent with decode-and-validate; (4) sketch the "
    "interfaces/functions to add and how they connect to existing code (MyDataset, collate_fn, "
    "GRUModel). Where useful, produce diagrams/plots illustrating the proposed latent-space "
    "optimization workflow. Provide an actionable summary tailored to this repository."
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

    # 1) High-effort literature query.
    if "lit" not in state:
        task = TaskRequest(name=JobNames.LITERATURE_HIGH, query=LIT_QUERY)
        state["lit"] = str(client.create_task(task))
        _save_state(state)
        print("LITERATURE_HIGH task:", state["lit"])
        time.sleep(10)  # gentle throttle to avoid 429
    else:
        print("LITERATURE_HIGH already dispatched:", state["lit"])

    # 2) Analysis query with the whole repository uploaded as a collection.
    if "analysis" not in state:
        resp = client.store_file_content(
            name="mofgru_repo_bundle",
            file_path=str(REPO_ROOT),
            description="Full MOF-GRU repository (code, dataset dictionary, model weights).",
            as_collection=True,
            ignore_patterns=UPLOAD_IGNORE,
        )
        entry_id = resp.data_storage.id
        state["analysis_entry"] = str(entry_id)
        _save_state(state)
        uri = f"data_entry:{entry_id}"
        print("Uploaded collection:", uri)

        task = TaskRequest(name=JobNames.ANALYSIS, query=ANALYSIS_QUERY)
        state["analysis"] = str(client.create_task(task, files=[uri]))
        _save_state(state)
        print("ANALYSIS task:", state["analysis"])
    else:
        print("ANALYSIS already dispatched:", state["analysis"])

    print("State written to", STATE_PATH)


if __name__ == "__main__":
    main()
