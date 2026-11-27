"""Turn SciFact into labeled (evidence, claim) pairs for the Verifier.

Step 1 of the training pipeline in the proposal. Writes three files to
data/processed/scifact/:

    train.jsonl  SciFact train split minus the held-out part   -> fine-tuning
    val.jsonl    held-out part of the train split              -> choosing settings, calibration
    dev.jsonl    official SciFact dev split, untouched         -> final reported results

(SciFact's test labels are not public, so dev is the final evaluation set.)

One pair = one claim + one paper the claim cites:
  * the paper has SUPPORT rationales    -> SUPPORTED
  * the paper has CONTRADICT rationales -> CONTRADICTED
  * the paper is cited but has no rationale for the claim -> INSUFFICIENT

The evidence text is the paper's title + full abstract, because in VeriLit the
Verifier checks a claim against the abstract of the paper it cites (arXiv and
Semantic Scholar both return abstracts). The gold rationale sentences are kept
in each record for analysis and for the retrieval Recall@k evaluation.

The train/val split is done by claim, not by pair, so the same claim never
appears in both.

Usage:
    python scripts/prepare_scifact.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from verilit.labels import INSUFFICIENT, LABELS, SCIFACT_TO_LABEL

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def evidence_text(doc: dict) -> str:
    return " ".join([doc["title"].strip(), *[s.strip() for s in doc["abstract"]]])


def make_pairs(claims: list[dict], corpus: dict[int, dict]) -> list[dict]:
    pairs = []
    for claim in claims:
        for doc_id in claim["cited_doc_ids"]:
            doc = corpus.get(doc_id)
            if doc is None:
                print(f"warning: claim {claim['id']} cites doc {doc_id}, which is not in the corpus; skipped")
                continue
            rationales = claim["evidence"].get(str(doc_id), [])
            if rationales:
                label = SCIFACT_TO_LABEL[rationales[0]["label"]]
                sentence_ids = sorted({i for r in rationales for i in r["sentences"]})
            else:
                label = INSUFFICIENT
                sentence_ids = []
            pairs.append(
                {
                    "claim_id": claim["id"],
                    "claim": claim["claim"],
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "evidence": evidence_text(doc),
                    "label": label,
                    "rationale_sentence_ids": sentence_ids,
                    "rationale_sentences": [doc["abstract"][i] for i in sentence_ids],
                }
            )
    return pairs


def label_counts(pairs: list[dict]) -> dict[str, int]:
    counts = Counter(p["label"] for p in pairs)
    return {label: counts.get(label, 0) for label in LABELS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "scifact")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "scifact")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="share of train claims held out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.raw / "corpus.jsonl").exists():
        raise SystemExit(f"No SciFact data in {args.raw}. Run: python scripts/download_scifact.py")

    corpus = {doc["doc_id"]: doc for doc in read_jsonl(args.raw / "corpus.jsonl")}
    train_claims = read_jsonl(args.raw / "claims_train.jsonl")
    dev_claims = read_jsonl(args.raw / "claims_dev.jsonl")

    rng = random.Random(args.seed)
    shuffled = sorted(train_claims, key=lambda c: c["id"])
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * args.val_fraction)
    val_claims, fit_claims = shuffled[:n_val], shuffled[n_val:]

    splits = {
        "train": make_pairs(fit_claims, corpus),
        "val": make_pairs(val_claims, corpus),
        "dev": make_pairs(dev_claims, corpus),
    }
    n_claims = {"train": len(fit_claims), "val": len(val_claims), "dev": len(dev_claims)}

    args.out.mkdir(parents=True, exist_ok=True)
    stats = {"seed": args.seed, "val_fraction": args.val_fraction, "splits": {}}
    print(f"{'split':<7}{'claims':>8}{'pairs':>8}   " + "  ".join(f"{label:>12}" for label in LABELS))
    for name, pairs in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", pairs)
        counts = label_counts(pairs)
        stats["splits"][name] = {"claims": n_claims[name], "pairs": len(pairs), "labels": counts}
        print(f"{name:<7}{n_claims[name]:>8}{len(pairs):>8}   " + "  ".join(f"{counts[l]:>12}" for l in LABELS))

    with open(args.out / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
