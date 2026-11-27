"""Run VeriLit on the custom test set and build the blind labeling sheets.

Runs every question in eval/questions.csv and saves eval/runs/<id>.json (+ .md).
Questions that already have a run are skipped, so if the free LLM quota runs out,
run the script again later and it continues where it stopped.

When every question has a run, it writes one labeling sheet per annotator to
eval/labels/labels_<name>.csv. Each row is a claim with the abstract of the
paper it cites, without the Verifier's verdict. A claim is labeled in its
original form (that is also the no-verification baseline's claim) and, if the
feedback loop changed it and VeriLit verified it, in its final form. A shared
set of rows (default 30) goes to both annotators to measure agreement; the rest
are split between them. Existing sheets are never overwritten unless you pass
--force-sheets, so labels can't be lost by re-running.

Usage:
    python scripts/run_eval_set.py
    python scripts/run_eval_set.py --limit 2            # try the first two questions
    python scripts/run_eval_set.py --only-sheets        # just (re)build sheets from existing runs
Then label (see eval/LABELING.md) and score:
    python scripts/score_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from verilit.llm import LLMError
from verilit.pipeline import ROOT, Config, VeriLit, to_markdown

EVAL = ROOT / "eval"
SHEET_COLUMNS = ["item_id", "label", "notes", "claim", "paper_title", "paper_abstract", "paper_url", "question"]


def read_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row.get("question", "").strip()]


def build_items(runs: dict[str, dict]) -> list[dict]:
    """Claims to label: every original claim, plus the final form of verified claims the loop changed."""
    items = []
    for qid, rec in runs.items():
        for c in rec["claims"]:
            o = c["original"]
            items.append({"item_id": f"{qid}-{c['id']}-o", "qid": qid, "claim_id": c["id"], "kind": "original",
                          "text": o["text"], "paper_id": o["paper_id"]})
            f = c["final"]
            if c["status"] == "verified" and (f["text"], f["paper_id"]) != (o["text"], o["paper_id"]):
                items.append({"item_id": f"{qid}-{c['id']}-f", "qid": qid, "claim_id": c["id"], "kind": "final",
                              "text": f["text"], "paper_id": f["paper_id"]})
    return items


def build_sheets(runs: dict[str, dict], questions: dict[str, str], annotators: list[str], overlap: int,
                 seed: int, labels_dir: Path, force: bool) -> None:
    sheets = {name: labels_dir / f"labels_{name}.csv" for name in annotators}
    existing = [p for p in sheets.values() if p.exists()]
    if existing and not force:
        print(f"Labeling sheets already exist ({', '.join(p.name for p in existing)}); not overwriting. "
              "Use --force-sheets to rebuild them (this discards their labels).")
        return

    items = build_items(runs)
    rng = random.Random(seed)
    rng.shuffle(items)
    shared = items[: min(overlap, len(items))]
    rest = items[len(shared):]
    assignment = {it["item_id"]: list(annotators) for it in shared}
    for i, it in enumerate(rest):
        assignment[it["item_id"]] = [annotators[i % len(annotators)]]
    for it in items:
        it["annotators"] = assignment[it["item_id"]]

    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_dir / "items.json", "w", encoding="utf-8") as fh:
        json.dump({"annotators": annotators, "overlap": len(shared), "seed": seed, "items": items}, fh,
                  indent=2, ensure_ascii=False)

    for name, path in sheets.items():
        mine = [it for it in items if name in it["annotators"]]
        rng.shuffle(mine)  # so the shared rows aren't all at the top
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel reads UTF-8
            writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            for it in mine:
                paper = runs[it["qid"]]["papers"][it["paper_id"]]
                writer.writerow({
                    "item_id": it["item_id"], "label": "", "notes": "", "claim": it["text"],
                    "paper_title": paper["title"], "paper_abstract": paper["abstract"],
                    "paper_url": paper.get("url") or "", "question": questions[it["qid"]],
                })
        print(f"  {path.relative_to(ROOT)}: {len(mine)} rows")
    print(f"{len(items)} claims to label ({len(shared)} shared). Labeling guide: eval/LABELING.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=EVAL / "questions.csv")
    parser.add_argument("--runs", type=Path, default=EVAL / "runs")
    parser.add_argument("--labels", type=Path, default=EVAL / "labels")
    parser.add_argument("--max-claims", type=int, default=2, help="claims per sub-question (2 keeps labeling to ~200)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--annotators", nargs="+", default=["maryam", "nimra"])
    parser.add_argument("--overlap", type=int, default=30, help="rows labeled by every annotator")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--only-sheets", action="store_true")
    parser.add_argument("--force-sheets", action="store_true")
    args = parser.parse_args()

    questions = read_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    args.runs.mkdir(parents=True, exist_ok=True)

    if not args.only_sheets:
        todo = [q for q in questions if not (args.runs / f"{q['id']}.json").exists()]
        print(f"{len(questions) - len(todo)} of {len(questions)} questions already run; {len(todo)} to go.")
        pipeline = None
        for n, q in enumerate(todo, start=1):
            if pipeline is None:
                pipeline = VeriLit(Config(max_claims=args.max_claims))
            print(f"\n[{n}/{len(todo)}] {q['id']}: {q['question']}")
            try:
                record = pipeline.run(q["question"])
            except LLMError as e:
                sys.exit(f"\nStopped: {e}\nRun this script again later; finished questions are kept.")
            record["question_id"], record["area"] = q["id"], q.get("area", "")
            with open(args.runs / f"{q['id']}.json", "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, ensure_ascii=False)
            with open(args.runs / f"{q['id']}.md", "w", encoding="utf-8") as fh:
                fh.write(to_markdown(record))
            s = record["stats"]
            print(f"  -> {s['claims']} claims, {s['verified']} verified, {s['unverified']} unverified; "
                  f"{s['llm_requests']} LLM requests, {s['seconds']:.0f} s")

    runs = {}
    for q in questions:
        path = args.runs / f"{q['id']}.json"
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                runs[q["id"]] = json.load(fh)
    if len(runs) < len(questions):
        sys.exit(f"Only {len(runs)} of {len(questions)} questions have runs; not building sheets yet.")
    print("\nBuilding labeling sheets ...")
    build_sheets(runs, {q["id"]: q["question"] for q in questions}, args.annotators, args.overlap, args.seed,
                 args.labels, args.force_sheets)


if __name__ == "__main__":
    main()
