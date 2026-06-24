# Optimizing over a latent space instead of global variables

This note answers issue
[#3](https://github.com/sgbaird/MOF-GRU/issues/3): *"How would one optimize over
a latent space instead of global variables? I.e., some kind of latent-like
embedding of the GNN structure, or using a different representation (e.g., VAE)
with a direct latent space."*

It is grounded in the current MOF-GRU code:
[`models.py`](../models.py) (`GRUModel`), [`utils.py`](../utils.py)
(`MyDataset`, `collate_fn`), [`get_MOFseq.py`](../get_MOFseq.py) (SELFIES
tokenization of the MOF "sentence"), and [`training.py`](../training.py).

> **See also:**
> [`candidate-space-optimization.md`](candidate-space-optimization.md) covers the
> complementary *predefined-candidate-space / featurization* approach (optimize
> by selecting from a fixed library of real MOFs, no decoder needed). That route
> is lower-effort and lower-risk and is recommended as the first thing to try;
> the generative VAE route below is for proposing genuinely novel structures.

## 1. What "optimizing over global variables" means here

Today MOF-GRU is a **supervised regressor**. `GRUModel` encodes a MOF sentence
(`<T> topo cat <N> node <E_1> selfies… <E_2> selfies…`, built in
`get_MOFseq.py`) into a pooled bidirectional hidden state and maps it to a
single scalar property (`fc_1 → ReLU → fc_2`). To "optimize over global
variables" you would search directly over interpretable design knobs (topology,
node, linker identity, pore-limiting diameter, density, etc.) and either look up
or predict the property. That search space is **discrete, combinatorial, and
non-differentiable**, so optimization is limited to enumeration or
genetic/discrete heuristics.

The alternative is to optimize over a **continuous latent space** `z`: a learned
vector representation in which (a) nearby points decode to similar, valid MOFs
and (b) properties vary smoothly. Then powerful continuous optimizers — Bayesian
optimization (BO) with a Gaussian process, or gradient ascent on a
differentiable property head — can be applied, and the optimum is **decoded**
back into a concrete MOF.

## 2. Does MOF-GRU already have a latent space?

Partly. `GRUModel` exposes the pooled hidden state via
`get_hidden_layer_output(x)` and the post-`fc_1` activations via
`get_fc_layer_output1(x)` ([`models.py`](../models.py) lines 42–50). These are a
*usable embedding* of an existing MOF and are fine for analysis, similarity
search, or fitting a surrogate over **known** structures.

They are **not** a good optimization space as-is, for two reasons:

1. **No decoder / not invertible.** The GRU is an *encoder-only* discriminative
   model. There is no map from an arbitrary vector `z` back to a token sequence,
   so an optimizer can move to a `z*` that no real MOF produces and that cannot
   be turned into a structure.
2. **No regularized geometry.** Nothing constrains the hidden space to be
   smooth, bounded, or "hole-free". Optimizers happily exploit off-manifold
   regions where the property head extrapolates nonsensically (the classic
   "adversarial latent" failure of latent-space optimization).

So the embedding is a *byproduct* latent space, not a *generative* one.

## 3. The fix: a generative latent space (sequence VAE)

The standard, well-validated recipe (ChemVAE / Gómez-Bombarelli et al. 2018;
SELFIES, Krenn et al.; junction-tree VAE; MOF-specific generators such as
Yao et al. / SmVAE and GHP-MOFassemble) is to learn a **variational
autoencoder** over the *same* representation MOF-GRU already uses, with a
**jointly trained property predictor**:

```
            ┌────────────┐      μ, logσ²     ┌──────────┐
 tokens ───►│  encoder   │ ───► z ~ N(μ,σ) ─►│ decoder  │──► reconstructed tokens
 (MOFseq)   │ (bi-GRU)   │           │       └──────────┘
            └────────────┘           │
                                     └──────► property head ŷ  (CH4ABL, ASA, …)
```

Loss = reconstruction (token cross-entropy) + β·KL(q(z|x) ‖ N(0,I)) +
λ·prediction (MSE to the property). The KL term gives a **smooth, bounded,
decodable** latent space; the property term aligns the latent geometry with the
objective (semi-supervised / "joint" VAE). Reuse what already exists:

- The token vocabulary in [`dataset/my_dict_output.json`](../dataset) and the
  tokenized sequences from `get_MOFseq.py` / `MyDataset` — no new featurization.
- The bidirectional GRU encoder from `GRUModel` — keep it, but emit `μ` and
  `logσ²` instead of going straight to `fc_1`.
- Add a small autoregressive GRU **decoder** over the same vocabulary, plus the
  existing regression head fed from `z`.
- Because the tokens are **SELFIES**, almost every decoded string is a *valid*
  molecule (that is the entire point of SELFIES), which largely solves the
  validity problem for the linker portion of the sentence.

### Why a VAE over the GNN/structure embedding instead?

If you want to optimize over a *structural graph* embedding (the issue's "latent
embedding of the GNN"), the same principle applies but is harder: a plain GNN
property predictor is also encoder-only and not invertible. To make a GNN latent
optimizable you need a **graph generative model** (graph VAE / autoregressive
graph generator / crystal diffusion model such as CDVAE) so that `z` can be
decoded back to a structure. For MOFs, the sequence/SELFIES VAE above is the
lower-effort, higher-validity path because the building-block "sentence"
representation is already implemented in this repo.

## 4. How to optimize once you have `z`

With a regularized, decodable latent space, two complementary optimizers work:

- **Latent-space Bayesian optimization (recommended to start).** Encode the
  training MOFs to `z`, fit a GP from `z → property`, and run BO (e.g. with
  `BoTorch`/`Ax`) **inside a trust region** around the data. After each proposal,
  **decode → validate → (optionally) score** before trusting it. Trust regions
  and *weighted retraining* (Tripp et al., 2020) keep the search on-manifold and
  are the key to avoiding the off-manifold failure mode.
- **Gradient ascent on the property head.** Since the property head is
  differentiable in `z`, take a few gradient steps `z ← z + η·∂ŷ/∂z` from good
  starting points, with a penalty that keeps `‖z‖` near the prior (stay where the
  decoder is reliable), then decode and validate.

Always **decode-and-validate**: map `z*` back to tokens, parse the SELFIES
linkers, and reject anything that fails to assemble into a real MOF. This is the
guardrail that makes latent-space optimization trustworthy versus optimizing a
raw, unregularized encoder embedding.

## 5. Concrete, minimal path for this repo

1. Add `MOFVAE` to `models.py` reusing the existing bi-GRU encoder: emit
   `mu`/`logvar`, sample `z` (reparameterization), add a GRU decoder over the
   vocabulary, and keep `fc_1/fc_2` as a property head fed from `z`.
2. Add a `train_vae.py` mirroring `training.py` (same `MyDataset`/`collate_fn`,
   add reconstruction + KL + property losses; reuse `dataset/my_dict_output.json`
   for `vocab_size` and for decoding indices back to symbols).
3. Add `optimize_latent.py`: encode the dataset to `z`, fit a GP surrogate, run
   trust-region BO (or gradient ascent on the property head), and
   `decode → validate (SELFIES) → report` the top candidates.

This converts the current "predict a property of a given MOF" pipeline into a
"propose a MOF with a target property" pipeline, optimizing over a continuous,
decodable latent space rather than over discrete global variables.

## Edison Scientific queries for this issue

Two Edison Scientific queries were prepared per the repository's
`.github/copilot-instructions.md` and issue #3:

- A **high-effort literature** query (`JobNames.LITERATURE_HIGH`) on latent-space
  optimization for MOF inverse design (VAEs, GNN/graph generative latents, and
  latent-space BO).
- An **analysis** query (`JobNames.ANALYSIS`) with the full repository uploaded,
  asking how to add a latent space to this specific codebase.

Both are implemented as reproducible, resumable scripts in
[`edison/`](../edison): run [`dispatch_latent_space.py`](../edison/dispatch_latent_space.py)
to send them and [`fetch_results.py`](../edison/fetch_results.py) to poll and
save the answers/notebooks/figures into `edison/artifacts/`. See
[`edison/README.md`](../edison/README.md) for status and run instructions.
