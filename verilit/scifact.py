"""Reading the SciFact files (shared by the data and evaluation scripts)."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def pair_passages(pairs: list[dict], raw_dir: Path) -> list[list[str]]:
    """Candidate passages (full abstract + 3-sentence windows) for each processed pair."""
    from verilit.nli import make_passages

    corpus = {d["doc_id"]: d for d in read_jsonl(Path(raw_dir) / "corpus.jsonl")}
    return [make_passages(corpus[p["doc_id"]]["title"], corpus[p["doc_id"]]["abstract"]) for p in pairs]


def evidence_text(doc: dict) -> str:
    """A paper as one string: title + full abstract (what the Verifier and retrievers see)."""
    return " ".join([doc["title"].strip(), *[s.strip() for s in doc["abstract"]]])
