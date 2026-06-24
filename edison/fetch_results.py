"""Fetch Edison Scientific trajectories dispatched by ``dispatch_latent_space.py``.

Reads task IDs from ``edison/tasks.json``, polls each until it reaches a
terminal state, then writes the answers and every associated artifact into
``edison/artifacts/<lit|analysis>/``.

For LITERATURE tasks the formatted answer is saved as ``answer.md``. For
ANALYSIS tasks the answer (``answer.md``) and the notebook (``notebook.ipynb``)
are saved, any ``data_entry`` artifacts are downloaded, and inline base64 PNGs
embedded in notebook cell outputs are decoded to ``figure_*.png`` (Edison crows
frequently return figures only as inline images rather than downloadable
artifacts).
"""

import base64
import json
import os
import time
from pathlib import Path

from edison_client import EdisonClient

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "edison" / "tasks.json"
ARTIFACT_ROOT = REPO_ROOT / "edison" / "artifacts"

TERMINAL = {"success", "fail", "cancelled", "truncated", "error"}


def _client() -> EdisonClient:
    key = (
        os.environ.get("EDISON_PLATFORM_API_KEY")
        or os.environ.get("EDISON_API_KEY")
        or ""
    ).strip()
    if not key:
        raise SystemExit("Set EDISON_PLATFORM_API_KEY (or EDISON_API_KEY).")
    return EdisonClient(api_key=key)


def _status(dump: dict) -> str:
    for key in ("status", "state"):
        val = dump.get(key)
        if isinstance(val, str):
            return val.lower()
    return "unknown"


def _wait(client: EdisonClient, task_id: str, poll: int = 60, timeout: int = 5400) -> dict:
    deadline = time.time() + timeout
    while True:
        dump = client.get_task(task_id).model_dump()
        status = _status(dump)
        print(f"  {task_id}: {status}")
        if status in TERMINAL or time.time() > deadline:
            return dump
        time.sleep(poll)


def _save_answer(dump: dict, out_dir: Path) -> None:
    answer = dump.get("formatted_answer") or dump.get("answer")
    if not answer:
        frame = dump.get("environment_frame") or {}
        try:
            answer = frame["state"]["state"]["answer"]
        except (KeyError, TypeError):
            answer = None
    if answer:
        (out_dir / "answer.md").write_text(answer)
        print(f"  wrote {out_dir / 'answer.md'} ({len(answer)} chars)")


def _save_notebook(dump: dict, out_dir: Path) -> None:
    notebook = dump.get("notebook")
    if not notebook:
        return
    (out_dir / "notebook.ipynb").write_text(json.dumps(notebook, indent=1))
    n = 0
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            png = data.get("image/png")
            if not png:
                continue
            n += 1
            (out_dir / f"figure_{n:02d}.png").write_bytes(base64.b64decode(png))
    if n:
        print(f"  decoded {n} inline figure(s)")


def _download_entries(client: EdisonClient, dump: dict, out_dir: Path) -> None:
    raw = json.dumps(dump)
    seen = set()
    # Collect any UUIDs referenced as data entries in the response.
    for token in raw.replace('"', " ").split():
        token = token.strip(",")
        if token.count("-") == 4 and len(token) == 36 and token not in seen:
            seen.add(token)
    for uid in seen:
        try:
            result = client.fetch_data_from_storage(uid)
        except Exception:  # noqa: BLE001 - best effort over many candidate UUIDs
            continue
        paths = result if isinstance(result, list) else [result]
        for p in paths:
            p = Path(p)
            if p.is_file():
                (out_dir / p.name).write_bytes(p.read_bytes())
                print(f"  fetched artifact {p.name}")


def main() -> None:
    if not STATE_PATH.exists():
        raise SystemExit("No edison/tasks.json; run dispatch_latent_space.py first.")
    state = json.loads(STATE_PATH.read_text())
    client = _client()

    for label in ("lit", "analysis"):
        task_id = state.get(label)
        if not task_id:
            continue
        print(f"[{label}] {task_id}")
        out_dir = ARTIFACT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        dump = _wait(client, task_id)
        (out_dir / "task.json").write_text(
            json.dumps({"task_id": task_id, "status": _status(dump)}, indent=2)
        )
        _save_answer(dump, out_dir)
        _save_notebook(dump, out_dir)
        _download_entries(client, dump, out_dir)


if __name__ == "__main__":
    main()
