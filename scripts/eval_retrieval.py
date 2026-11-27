"""Evaluate retrieval on SciFact: semantic search vs. keyword (BM25) search.

For every SciFact claim that has evidence, both rankers search all 5,183 SciFact
abstracts with the claim as the query. Recall@k is the share of gold evidence
abstracts that appear in the top k, i.e. whether the retrieved text contains the
evidence needed to verify the claim. Hit@k is the share of claims with at least
one gold abstract in the top k. Claims without evidence (NEI) have no gold
abstract and are skipped.

In the VeriLit pipeline the ranker only chooses among a few dozen candidates
from Semantic Scholar / arXiv, so this 5,183-abstract search is the harder test.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --methods bm25          # keyword only (seconds)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verilit.retrieval import DEFAULT_EMBEDDING_MODEL, make_ranker
from verilit.scifact import evidence_text, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
KS = [1, 3, 5, 10, 20]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["train", "dev"], help="which SciFact claims to use")
    parser.add_argument("--methods", nargs="+", default=["bm25", "semantic"], choices=["bm25", "semantic"])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "scifact")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if not (args.raw / "corpus.jsonl").exists():
        raise SystemExit(f"No SciFact data in {args.raw}. Run: python scripts/download_scifact.py")
    corpus = read_jsonl(args.raw / "corpus.jsonl")
    doc_ids = [d["doc_id"] for d in corpus]
    texts = [evidence_text(d) for d in corpus]

    claims = [c for c in read_jsonl(args.raw / f"claims_{args.split}.jsonl") if c["evidence"]]
    queries = [c["claim"] for c in claims]
    gold = [{int(doc_id) for doc_id in c["evidence"]} for c in claims]
    n_gold = sum(len(g) for g in gold)
    print(f"{len(corpus)} abstracts, {len(claims)} {args.split} claims with evidence, {n_gold} gold abstracts")

    results = {"split": args.split, "claims": len(claims), "gold_abstracts": n_gold, "corpus": len(corpus), "methods": {}}
    for method in args.methods:
        start = time.time()
        kwargs = {"model_name": args.embedding_model} if method == "semantic" else {}
        ranker = make_ranker(method, **kwargs)
        if method == "semantic":
            ranker.fit(texts, show_progress=True)
        else:
            ranker.fit(texts)
        hits = ranker.search(queries, k=max(KS))
        retrieved = [[doc_ids[i] for i, _ in h] for h in hits]

        scores = {}
        for k in KS:
            found = sum(len(g & set(r[:k])) for g, r in zip(gold, retrieved))
            any_found = sum(bool(g & set(r[:k])) for g, r in zip(gold, retrieved))
            scores[f"recall@{k}"] = found / n_gold
            scores[f"hit@{k}"] = any_found / len(claims)
        scores["seconds"] = round(time.time() - start, 1)
        if method == "semantic":
            scores["embedding_model"] = args.embedding_model
        results["methods"][method] = scores

    print(f"\n{'':<10}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + "".join(f"{'Hit@' + str(k):>9}" for k in KS))
    for method, s in results["methods"].items():
        print(
            f"{method:<10}"
            + "".join(f"{s[f'recall@{k}']:>8.3f}" for k in KS)
            + "".join(f"{s[f'hit@{k}']:>9.3f}" for k in KS)
        )

    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"retrieval_{args.split}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
