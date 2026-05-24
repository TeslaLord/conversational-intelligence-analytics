"""Plain-text log of conv_ids that have completed a pipeline batch.

Used by the pipeline to skip already-processed conversations on subsequent
incremental runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_processed(path: Path, conv_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for cid in conv_ids:
            f.write(f"{cid}\n")


def load_pipeline_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_pipeline_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))
