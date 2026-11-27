"""Score the custom test set from the hand labels (section 4 of the proposal).

Reads eval/labels/items.json, the annotators' sheets (labels_<name>.csv) and the
runs in eval/runs/, and reports:

  * annotator agreement on the shared rows (raw agreement and Cohen's kappa);
  * for VeriLit and the no-verification baseline:
      - unsupported-claim rate: share of presented claims the hand labels mark
        Contradicted or Insufficient (the baseline presents every original
        claim; VeriLit presents only the claims it verified, in final form);
      - citation correctness: share of presented citations whose paper supports
        the claim (each claim cites one paper, so this is 1 - unsupported rate);
      - evidence coverage: share of sub-questions with at least one presented
        claim the hand labels mark Supported (so dropping claims can't score well);
  * the feedback loop: retry success rate (retried claims that end verified, and
    how many of those the hand labels confirm) and average retries per review;
  * the Verifier against the hand labels on the original claims, with full-abstract
    scoring (what VeriLit used) and best-passage scoring (recorded for comparison).

Disagreements on shared rows: the first time, the script writes
eval/labels/disagreements.csv. Discuss each one, fill in `agreed_label`, and run
again; until then the first annotator's label is used. Kappa is always computed
from the sheets as labeled, before adjudication.

Usage:
    python scripts/score_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

from verilit.labels import CONTRADICTED, INSUFFICIENT, SUPPORTED
from verilit.pipeline import ROOT

EVAL = ROOT / "eval"


def show(path: Path) -> str:
    """Path relative to the project folder when inside it, else as given."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
LETTER = {"S": SUPPORTED, "C": CONTRADICTED, "I": INSUFFICIENT}


def norm_label(value: str) -> str | None:
    v = (value or "").strip().upper()
    return LETTER.get(v[:1]) if v else None


def read_sheet(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        return {row["item_id"]: norm_label(row.get("label", "")) for row in csv.DictReader(fh)}


def binary(pred: list[bool], gold: list[bool]) -> dict:
    p, r, f, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)
    acc = sum(a == b for a, b in zip(pred, gold)) / len(gold) if gold else 0.0
    return {"n": len(gold), "accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f)}


def rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=EVAL / "labels")
    parser.add_argument("--runs", type=Path, default=EVAL / "runs")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    with open(args.labels / "items.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    annotators, items = meta["annotators"], meta["items"]
    by_id = {it["item_id"]: it for it in items}
    sheets = {name: read_sheet(args.labels / f"labels_{name}.csv") for name in annotators}

    # -- agreement on shared rows ----------------------------------------------
    shared = [it for it in items if len(it["annotators"]) > 1]
    a, b = annotators[0], annotators[1] if len(annotators) > 1 else annotators[0]
    both = [it for it in shared if sheets[a].get(it["item_id"]) and sheets[b].get(it["item_id"])]
    la = [sheets[a][it["item_id"]] for it in both]
    lb = [sheets[b][it["item_id"]] for it in both]
    agreement = {
        "shared_rows": len(shared), "labeled_by_both": len(both),
        "raw_agreement": rate([x == y for x, y in zip(la, lb)]),
        "cohen_kappa": float(cohen_kappa_score(la, lb)) if len(set(la + lb)) > 1 else None,
    }

    # Adjudication of disagreements.
    dis_path = args.labels / "disagreements.csv"
    agreed: dict[str, str | None] = {}
    if dis_path.exists():
        with open(dis_path, encoding="utf-8-sig") as fh:
            agreed = {r["item_id"]: norm_label(r.get("agreed_label", "")) for r in csv.DictReader(fh)}
    disagreements = [it for it, x, y in zip(both, la, lb) if x != y]
    if disagreements and not dis_path.exists():
        with open(dis_path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["item_id", "agreed_label", a, b, "claim"])
            w.writeheader()
            for it in disagreements:
                w.writerow({"item_id": it["item_id"], "agreed_label": "", a: sheets[a][it["item_id"]],
                            b: sheets[b][it["item_id"]], "claim": it["text"]})
        print(f"Wrote {show(dis_path)} ({len(disagreements)} rows): fill in agreed_label and re-run.")

    def human(item_id: str) -> str | None:
        it = by_id.get(item_id)
        if it is None:
            return None
        labels = [sheets[n].get(item_id) for n in it["annotators"]]
        labels = [x for x in labels if x]
        if not labels:
            return None
        if len(set(labels)) > 1:
            return agreed.get(item_id) or labels[0]
        return labels[0]

    labeled = sum(human(it["item_id"]) is not None for it in items)

    # -- system metrics --------------------------------------------------------
    runs = {}
    for qid in sorted({it["qid"] for it in items}):
        with open(args.runs / f"{qid}.json", encoding="utf-8") as fh:
            runs[qid] = json.load(fh)

    systems = {"baseline": {"presented": [], "covered": []}, "verilit": {"presented": [], "covered": []}}
    verifier_rows = []
    retried, retry_human = 0, []
    retry_successes, total_retries = 0, 0
    for qid, rec in runs.items():
        sq_supported = {"baseline": {sq["sq_id"]: False for sq in rec["sub_questions"]},
                        "verilit": {sq["sq_id"]: False for sq in rec["sub_questions"]}}
        for c in rec["claims"]:
            h_orig = human(f"{qid}-{c['id']}-o")
            if h_orig:
                systems["baseline"]["presented"].append(h_orig)
                sq_supported["baseline"][c["sub_question"]] |= h_orig == SUPPORTED
                first = c["attempts"][0] if c["attempts"] else None
                if first:
                    verifier_rows.append((first, h_orig))
            final_id = f"{qid}-{c['id']}-f"
            # The final form is labeled separately only if the loop changed the claim.
            h_final = human(final_id) if final_id in by_id else h_orig
            if c["status"] == "verified":
                if h_final:
                    systems["verilit"]["presented"].append(h_final)
                    sq_supported["verilit"][c["sub_question"]] |= h_final == SUPPORTED
            if len(c["attempts"]) > 1:
                retried += 1
                total_retries += len(c["attempts"]) - 1
                if c["status"] == "verified":
                    retry_successes += 1
                    if h_final:
                        retry_human.append(h_final == SUPPORTED)
        for name in systems:
            systems[name]["covered"] += list(sq_supported[name].values())

    system_metrics = {}
    for name, s in systems.items():
        unsupported = rate([h != SUPPORTED for h in s["presented"]])
        system_metrics[name] = {
            "presented_claims_labeled": len(s["presented"]),
            "unsupported_claim_rate": unsupported,
            "contradicted_rate": rate([h == CONTRADICTED for h in s["presented"]]),
            "citation_correctness": None if unsupported is None else 1 - unsupported,
            "evidence_coverage": rate(s["covered"]),
        }

    loop = {
        "claims_retried": retried,
        "retry_success_rate": retry_successes / retried if retried else None,
        "retry_successes_confirmed_by_labels": rate(retry_human),
        "avg_retries_per_review": total_retries / len(runs) if runs else None,
    }

    # -- Verifier vs. hand labels (original claims) ------------------------------
    gold3 = [h for _, h in verifier_rows]
    pred3 = [a_["label"] for a_, _ in verifier_rows]
    gold_s = [h == SUPPORTED for h in gold3]
    thr = next(iter(runs.values()))["config"]["threshold"] if runs else 0.7
    evidence_used = next(iter(runs.values()))["config"].get("evidence", "abstract") if runs else "abstract"
    verifier = {
        "n": len(verifier_rows),
        "three_way_accuracy": rate([p == g for p, g in zip(pred3, gold3)]),
        "three_way_macro_f1": float(f1_score(gold3, pred3, average="macro", zero_division=0)) if gold3 else None,
        f"accept_vs_supported_{evidence_used}": binary([a_["accepted"] for a_, _ in verifier_rows], gold_s),
    }
    alt_rows = [(a_, h) for a_, h in verifier_rows if a_.get("alt_p_supported") is not None]
    if alt_rows:
        other = "passages" if evidence_used == "abstract" else "abstract"
        verifier[f"accept_vs_supported_{other}"] = binary(
            [a_["alt_p_supported"] >= thr for a_, _ in alt_rows], [h == SUPPORTED for _, h in alt_rows])

    result = {"items": len(items), "items_labeled": labeled, "agreement": agreement, "systems": system_metrics,
              "feedback_loop": loop, "verifier_vs_humans": verifier, "threshold": thr}
    args.results.mkdir(parents=True, exist_ok=True)
    with open(args.results / "custom_eval.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # -- print -------------------------------------------------------------------
    print(f"Labeled {labeled} of {len(items)} claims.")
    print(f"Agreement on {agreement['labeled_by_both']} shared rows: raw {fmt(agreement['raw_agreement'])}, "
          f"Cohen's kappa {fmt(agreement['cohen_kappa'])}\n")
    print(f"{'':<10}{'claims':>8}{'unsupported':>13}{'contradicted':>14}{'citation ok':>13}{'coverage':>10}")
    for name, m in system_metrics.items():
        print(f"{name:<10}{m['presented_claims_labeled']:>8}{fmt(m['unsupported_claim_rate']):>13}"
              f"{fmt(m['contradicted_rate']):>14}{fmt(m['citation_correctness']):>13}{fmt(m['evidence_coverage']):>10}")
    print(f"\nFeedback loop: {loop['claims_retried']} claims retried, success rate {fmt(loop['retry_success_rate'])} "
          f"(confirmed by labels: {fmt(loop['retry_successes_confirmed_by_labels'])}), "
          f"{fmt(loop['avg_retries_per_review'])} retries per review")
    print(f"\nVerifier vs. hand labels on {verifier['n']} original claims: 3-way accuracy "
          f"{fmt(verifier['three_way_accuracy'])}, macro-F1 {fmt(verifier['three_way_macro_f1'])}")
    for key, m in verifier.items():
        if key.startswith("accept_vs_supported_"):
            print(f"  accept (P(SUPPORTED) >= {thr}) vs. labeled Supported, {key.rsplit('_', 1)[-1]:<9}: "
                  f"precision {fmt(m['precision'])}  recall {fmt(m['recall'])}  F1 {fmt(m['f1'])}")
    print(f"\nSaved {show(args.results / 'custom_eval.json')}")


if __name__ == "__main__":
    main()
