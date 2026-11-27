"""Evaluation metrics for the Verifier (section 4 of the proposal)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from verilit.labels import LABEL2ID, LABELS


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE: the gap between confidence and accuracy, averaged over confidence bins.

    0 means the model's confidence matches how often it is right.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def classification_metrics(gold_labels: list[str], probs: np.ndarray) -> dict:
    """Accuracy, macro precision/recall/F1, per-label scores, confusion matrix and ECE."""
    y_true = np.array([LABEL2ID[label] for label in gold_labels])
    y_pred = probs.argmax(axis=1)
    label_ids = list(range(len(LABELS)))

    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="macro", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "ece": expected_calibration_error(probs, y_true),
        "per_label": {
            label: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        # rows = gold label, columns = predicted label, both in LABELS order
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "labels": LABELS,
    }


def format_metrics(m: dict) -> str:
    """A short plain-text summary for the terminal."""
    lines = [
        f"pairs: {m['n']}",
        f"accuracy: {m['accuracy']:.3f}   macro P/R/F1: "
        f"{m['macro_precision']:.3f} / {m['macro_recall']:.3f} / {m['macro_f1']:.3f}   ECE: {m['ece']:.3f}",
        "",
        f"{'label':<14}{'P':>7}{'R':>7}{'F1':>7}{'n':>6}",
    ]
    for label, s in m["per_label"].items():
        lines.append(f"{label:<14}{s['precision']:>7.3f}{s['recall']:>7.3f}{s['f1']:>7.3f}{s['support']:>6}")
    lines += ["", "confusion matrix (rows = gold, columns = predicted):"]
    short = [label[:5] for label in m["labels"]]
    lines.append(" " * 14 + "".join(f"{s:>8}" for s in short))
    for label, row in zip(m["labels"], m["confusion_matrix"]):
        lines.append(f"{label:<14}" + "".join(f"{v:>8}" for v in row))
    return "\n".join(lines)
