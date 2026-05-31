from __future__ import annotations

import datetime
import json
import os
import subprocess
from typing import Any

from .config import REPO_ROOT


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def git_sha() -> str:
    # git may legitimately be absent (e.g. a fresh Colab checkout): degrade to "unknown".
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _library_versions() -> dict:
    import torch
    import transformers
    import peft
    import trl

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
    }


def write_run_meta(out_dir: str, resolved_cfg: dict) -> None:
    ensure_dir(out_dir)
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "library_versions": _library_versions(),
        "config": resolved_cfg,
    }
    write_json(os.path.join(out_dir, "run_meta.json"), meta)
