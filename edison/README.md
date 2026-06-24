# Edison Scientific queries — latent-space optimization (issue #3)

This directory contains reproducible scripts that send the two Edison Scientific
queries requested in
[issue #3](https://github.com/sgbaird/MOF-GRU/issues/3) and fetch their results:

| Script | Purpose |
| --- | --- |
| [`dispatch_latent_space.py`](dispatch_latent_space.py) | Sends a high-effort **LITERATURE** query and an **ANALYSIS** query (with the whole repo uploaded as a single zipped collection). Records task IDs to `tasks.json` so the run is resumable. |
| [`fetch_results.py`](fetch_results.py) | Polls each task to a terminal state and writes answers, notebooks, downloaded artifacts, and decoded inline figures into `artifacts/<lit|analysis>/`. |

## Status

> ⚠️ **Not yet executed.** When this branch was prepared, the Edison endpoint
> `api.platform.edisonscientific.com` was **not reachable from the agent
> sandbox** (DNS for `*.edisonscientific.com` was blocked, so authentication
> failed before any task could be created). The scripts below are ready to run
> as soon as the domain is allow-listed; `artifacts/` will be populated then.

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
