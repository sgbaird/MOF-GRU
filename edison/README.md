# Edison Scientific queries — latent-space optimization (issue #3)

This directory contains reproducible scripts that send the two Edison Scientific
queries requested in
[issue #3](https://github.com/sgbaird/MOF-GRU/issues/3) and fetch their results:

| Script | Purpose |
| --- | --- |
| [`dispatch_latent_space.py`](dispatch_latent_space.py) | Sends a high-effort **LITERATURE** query and an **ANALYSIS** query (with the whole repo uploaded as a single zipped collection). Records task IDs to `tasks.json` so the run is resumable. |
| [`fetch_results.py`](fetch_results.py) | Polls each task to a terminal state and writes answers, notebooks, downloaded artifacts, and decoded inline figures into `artifacts/<lit|analysis>/`. |

## Status

> ✅ **Executed.** Both Edison tasks ran to `success` and their trajectories are
> committed under [`artifacts/`](artifacts):
>
> | Task | Job | ID | Output |
> | --- | --- | --- | --- |
> | Literature (high) | `LITERATURE_HIGH` | `ffe4cb56-c6a7-41d0-9182-ec8d79b5fa0d` | [`artifacts/lit/answer.md`](artifacts/lit/answer.md) (~49k chars, fully cited) |
> | Analysis | `ANALYSIS` | `d6673951-5d91-4049-be04-d2de3f9f6dcd` | [`artifacts/analysis/answer.md`](artifacts/analysis/answer.md), [`notebook.ipynb`](artifacts/analysis/notebook.ipynb), 3 decoded figures |
>
> The ANALYSIS run had the full repository uploaded as a single zipped collection
> and independently confirmed the dataset scale (**113,160 MOFs**, vocabulary
> 583) and the encoder-vs-generative-latent distinction documented in
> [`../docs/`](../docs).

## How to run

```bash
pip install edison-client
export EDISON_PLATFORM_API_KEY=...   # or EDISON_API_KEY

python edison/dispatch_latent_space.py   # creates tasks, writes edison/tasks.json
# high-effort literature can take ~15 min; analysis a few minutes
python edison/fetch_results.py           # polls + saves to edison/artifacts/
```

`dispatch_latent_space.py` is **idempotent**: if `tasks.json` already has task
IDs it will not resubmit, so `fetch_results.py` can be re-run later (even in a
new session) to retrieve finished trajectories.

## Notes / conventions used

- The API key is read from `EDISON_PLATFORM_API_KEY` (fallback `EDISON_API_KEY`)
  and `.strip()`-ed — the injected secret can carry a trailing newline that
  otherwise causes a `403 Forbidden` on login.
- The ANALYSIS upload uses
  `client.store_file_content(file_path=<repo>, as_collection=True)` (a single
  zipped collection); the large redundant `MOF-GRU.zip` and `.git/` are
  excluded via `ignore_patterns`.
- Endpoint: `https://api.platform.edisonscientific.com`
  (see `.github/copilot-instructions.md`).
