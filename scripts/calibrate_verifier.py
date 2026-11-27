"""Calibrate the Verifier's confidence scores (step 4 of the training pipeline).

Uses temperature scaling (Guo et al., 2017): the model's logits are divided by
a single number T before the softmax. T > 1 makes an over-confident model less
sure of itself. It never changes which label is predicted, so accuracy and F1
stay the same; only the confidence scores change. T is fitted on the val split
by minimising the negative log-likelihood (NLL). The dev split is not used here.

The calibrated probability is VeriLit's confidence score. To help choose the
acceptance threshold for the pipeline, the script also shows, for several
thresholds, how many val pairs would be accepted as SUPPORTED and how many of
those are really supported.

Writes:
    models/verifier/calibration.json   read by NLIVerifier.load(..., use_calibration=True)
    results/calibration_val.json       the same numbers, kept in git for the report

Usage:
    python scripts/calibrate_verifier.py
Then evaluate on dev with the temperature applied:
    python scripts/eval_verifier.py --model models/verifier --calibrated --name calibrated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from verilit.labels import LABEL2ID, LABELS, SUPPORTED
from verilit.metrics import expected_calibration_error
from verilit.nli import CALIBRATION_FILE, NLIVerifier, softmax

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Average negative log-likelihood of the gold labels at this temperature."""
    log_probs = log_softmax(logits / temperature, axis=1)
    return float(-log_probs[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """The temperature (between 0.05 and 20) with the lowest NLL."""
    result = minimize_scalar(
        lambda log_t: nll(logits, y, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def scores(logits: np.ndarray, y: np.ndarray, temperature: float) -> dict:
    probs = softmax(logits / temperature)
    return {
        "nll": nll(logits, y, temperature),
        "ece": expected_calibration_error(probs, y),
        "accuracy": float((probs.argmax(axis=1) == y).mean()),
    }


def threshold_table(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    """Accept as SUPPORTED when P(SUPPORTED) >= threshold: how many, and how many are right."""
    sup = LABEL2ID[SUPPORTED]
    is_supported = y == sup
    rows = []
    for t in THRESHOLDS:
        accepted = probs[:, sup] >= t
        n = int(accepted.sum())
        rows.append(
            {
                "threshold": t,
                "accepted": n,
                # of the accepted pairs, the share that are really SUPPORTED
                "precision": float(is_supported[accepted].mean()) if n else None,
                # of the really SUPPORTED pairs, the share that get accepted
                "recall": float(accepted[is_supported].mean()) if is_supported.any() else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "verifier",
                        help="folder of the fine-tuned Verifier (from finetune_verifier.py)")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "scifact")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if not args.model.is_dir():
        raise SystemExit(f"{args.model} is not a folder. Run scripts/finetune_verifier.py first.")
    val_path = args.data / "val.jsonl"
    if not val_path.exists():
        raise SystemExit(f"{val_path} not found. Run: python scripts/prepare_scifact.py")
    with open(val_path, encoding="utf-8") as fh:
        pairs = [json.loads(line) for line in fh if line.strip()]

    verifier = NLIVerifier.load(str(args.model), max_length=args.max_length, batch_size=args.batch_size)
    print(f"model: {args.model}   device: {verifier.device}   val pairs: {len(pairs)}")
    logits = verifier.predict_logits([p["evidence"] for p in pairs], [p["claim"] for p in pairs])
    y = np.array([LABEL2ID[p["label"]] for p in pairs])

    temperature = fit_temperature(logits, y)
    before, after = scores(logits, y, 1.0), scores(logits, y, temperature)
    table = threshold_table(softmax(logits / temperature), y)

    result = {
        "temperature": temperature,
        "fitted_on": "val",
        "pairs": len(pairs),
        "model": str(args.model),
        "before": before,
        "after": after,
        "accept_supported_thresholds": table,
        "labels": LABELS,
    }
    with open(args.model / CALIBRATION_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    args.results.mkdir(parents=True, exist_ok=True)
    with open(args.results / "calibration_val.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"\ntemperature T = {temperature:.3f}  (fitted on val)\n")
    print(f"{'':<8}{'NLL':>8}{'ECE':>8}{'accuracy':>10}")
    for name, s in (("before", before), ("after", after)):
        print(f"{name:<8}{s['nll']:>8.3f}{s['ece']:>8.3f}{s['accuracy']:>10.3f}")

    print("\nAccept as SUPPORTED when calibrated P(SUPPORTED) >= threshold (val split):")
    print(f"{'threshold':>9}{'accepted':>10}{'precision':>11}{'recall':>8}")
    for row in table:
        prec = f"{row['precision']:.3f}" if row["precision"] is not None else "-"
        rec = f"{row['recall']:.3f}" if row["recall"] is not None else "-"
        print(f"{row['threshold']:>9.2f}{row['accepted']:>10}{prec:>11}{rec:>8}")
    n_sup = int((y == LABEL2ID[SUPPORTED]).sum())
    print(f"(val has only {n_sup} SUPPORTED pairs, so treat these as rough guides.)")

    print(f"\nSaved {args.model / CALIBRATION_FILE} and {args.results / 'calibration_val.json'}")
    print(f"Next: python scripts/eval_verifier.py --model {args.model} --calibrated --name calibrated")


if __name__ == "__main__":
    main()
