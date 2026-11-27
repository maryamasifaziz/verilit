"""Evaluate an NLI model as the VeriLit Verifier on a SciFact split.

Baseline (step 2 of the training pipeline): an off-the-shelf NLI model, no fine-tuning.
    python scripts/eval_verifier.py

The fine-tuned Verifier uses the same script, pointed at its folder:
    python scripts/eval_verifier.py --model models/verifier --name finetuned

And after calibrate_verifier.py, with the fitted temperature applied:
    python scripts/eval_verifier.py --model models/verifier --calibrated --name calibrated

Writes results/<name>_<split>/metrics.json and predictions.jsonl. Reports
accuracy, macro precision/recall/F1 over the three labels, per-label scores,
the confusion matrix and expected calibration error (ECE).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verilit.labels import LABELS
from verilit.metrics import classification_metrics, format_metrics
from verilit.nli import NLIVerifier

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name or local folder")
    parser.add_argument("--name", default="baseline", help="label for the results folder")
    parser.add_argument("--split", default="dev", choices=["train", "val", "dev"])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "scifact")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--calibrated", action="store_true",
                        help="apply the temperature saved by calibrate_verifier.py in the model folder")
    parser.add_argument("--temperature", type=float, default=None, help="use this temperature instead (overrides)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="only the first N pairs (quick test)")
    args = parser.parse_args()

    path = args.data / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/prepare_scifact.py")
    with open(path, encoding="utf-8") as fh:
        pairs = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    verifier = NLIVerifier.load(
        args.model, max_length=args.max_length, batch_size=args.batch_size, use_calibration=args.calibrated
    )
    if args.temperature is not None:
        verifier.temperature = args.temperature
    print(
        f"model: {args.model}   device: {verifier.device}   split: {args.split} ({len(pairs)} pairs)   "
        f"temperature: {verifier.temperature:.3f}"
    )

    start = time.time()
    probs = verifier.predict_proba([p["evidence"] for p in pairs], [p["claim"] for p in pairs])
    seconds = time.time() - start

    metrics = classification_metrics([p["label"] for p in pairs], probs)
    metrics.update(
        model=args.model,
        split=args.split,
        temperature=verifier.temperature,
        max_length=args.max_length,
        seconds=round(seconds, 1),
    )

    out_dir = args.results / f"{args.name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for pair, row in zip(pairs, probs):
            fh.write(
                json.dumps(
                    {
                        "claim_id": pair["claim_id"],
                        "doc_id": pair["doc_id"],
                        "claim": pair["claim"],
                        "gold": pair["label"],
                        "pred": LABELS[int(row.argmax())],
                        "confidence": float(row.max()),
                        "probs": {label: float(p) for label, p in zip(LABELS, row)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(format_metrics(metrics))
    print(f"\n{seconds:.0f}s. Saved to {out_dir}")


if __name__ == "__main__":
    main()
