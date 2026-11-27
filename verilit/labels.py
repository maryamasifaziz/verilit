"""Verdict labels shared by the data scripts, the Verifier and the pipeline.

Every probability array in this project uses the order of LABELS, so column 0
is always SUPPORTED, column 1 CONTRADICTED and column 2 INSUFFICIENT.
"""

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT"  # "Insufficient Evidence" in the proposal

LABELS = [SUPPORTED, CONTRADICTED, INSUFFICIENT]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# SciFact's own label names.
SCIFACT_TO_LABEL = {
    "SUPPORT": SUPPORTED,
    "CONTRADICT": CONTRADICTED,
    "NEI": INSUFFICIENT,  # no evidence in the cited abstract
}


def label_from_model_name(name: str) -> str | None:
    """Map a model's label name (from config.id2label) to a VeriLit label.

    Off-the-shelf NLI models use entailment / contradiction / neutral;
    a fine-tuned VeriLit Verifier is saved with the LABELS names directly.
    Returns None for names it cannot interpret (e.g. "LABEL_0").
    """
    n = name.strip().lower()
    if n.startswith("entail") or n in {"supported", "support"}:
        return SUPPORTED
    if n.startswith("contradict"):
        return CONTRADICTED
    if n.startswith("neutral") or n in {"insufficient", "nei", "not enough info"}:
        return INSUFFICIENT
    return None
