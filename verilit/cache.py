"""A small on-disk cache for API responses (LLM calls, paper searches).

Free-tier APIs have daily limits, so every response is saved under cache/ and
a repeated request (same model and prompt, same search) is answered from disk.
Re-running a question or an evaluation then costs no quota, and the results
stay reproducible. Delete the cache/ folder to start fresh.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MISSING = object()  # returned by get() when nothing is cached (a cached value can be [] or None)


class DiskCache:
    def __init__(self, namespace: str, root: Path | None = None):
        root = root or Path(os.environ.get("VERILIT_CACHE_DIR", ROOT / "cache"))
        self.dir = root / namespace

    def _path(self, key: dict) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.dir / digest[:2] / f"{digest}.json"

    def get(self, key: dict):
        path = self._path(key)
        if not path.exists():
            return MISSING
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["value"]

    def set(self, key: dict, value) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "value": value}, fh, ensure_ascii=False)
        tmp.replace(path)
