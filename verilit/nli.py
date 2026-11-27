"""The Verifier's model: an NLI cross-encoder that scores (evidence, claim) pairs.

The same class loads an off-the-shelf NLI model from Hugging Face (the
baseline) or a fine-tuned checkpoint from a local folder, and always returns
scores in the order of verilit.labels.LABELS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verilit.labels import LABELS, label_from_model_name

CALIBRATION_FILE = "calibration.json"  # written into the model folder by calibrate_verifier.py


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NLIVerifier:
    """Scores whether an evidence passage supports, contradicts or does not address a claim.

    The evidence is the NLI premise and the claim is the hypothesis. Long
    evidence is truncated; the claim is never cut.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        temperature: float = 1.0,
    ):
        """Wrap an already-loaded model (used during fine-tuning). Use NLIVerifier.load() otherwise."""
        self.device = device or pick_device()
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature  # 1.0 = uncalibrated
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        # column_order[i] = the model output index that holds LABELS[i]
        self.column_order = column_order(model.config.id2label)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        use_calibration: bool = False,
        evidence: str = "abstract",
    ):
        """Load a Hugging Face model name or a local folder (e.g. models/verifier).

        use_calibration=True reads the temperature that calibrate_verifier.py
        saved in the model folder, so confidence scores are calibrated. The
        pipeline should always load the Verifier this way. `evidence` picks the
        calibration fitted for full abstracts ("abstract") or passages ("passages").
        """
        temperature = 1.0
        if use_calibration:
            path = Path(model_name_or_path) / calibration_file(evidence)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run: python scripts/calibrate_verifier.py --evidence {evidence}")
            with open(path, encoding="utf-8") as fh:
                temperature = float(json.load(fh)["temperature"])
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, device, max_length, batch_size, temperature)

    def encode(self, evidence: list[str], claims: list[str]):
        """Tokenize (evidence, claim) pairs; long evidence is truncated, the claim is kept whole."""
        return self.tokenizer(
            evidence,
            claims,
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def predict_logits(self, evidence: list[str], claims: list[str], show_progress: bool = True) -> np.ndarray:
        """Raw logits, shape (n, 3), columns in LABELS order. Calibration uses these."""
        if len(evidence) != len(claims):
            raise ValueError("evidence and claims must have the same length")
        was_training = self.model.training
        self.model.eval()
        # Batch similar lengths together so less time is spent on padding.
        order = np.argsort([len(e) + len(c) for e, c in zip(evidence, claims)])
        out = np.zeros((len(claims), len(LABELS)), dtype=np.float32)
        batches = range(0, len(order), self.batch_size)
        for start in tqdm(batches, desc="NLI", disable=not show_progress):
            idx = order[start : start + self.batch_size]
            enc = self.encode([evidence[i] for i in idx], [claims[i] for i in idx])
            logits = self.model(**enc).logits.float().cpu().numpy()
            out[idx] = logits[:, self.column_order]
        self.model.train(was_training)
        return out

    def predict_logits_passages(
        self, passages: list[list[str]], claims: list[str], show_progress: bool = True
    ) -> np.ndarray:
        """Logits, shape (n, 3), when each claim has several candidate passages.

        Every (passage, claim) pair is scored, and for each claim the passage
        that is most sure it found relevant evidence (lowest INSUFFICIENT
        probability) is kept. Long abstracts dilute the evidence; a short
        window around the relevant sentences is easier for the model to judge.
        """
        flat_ev = [p for plist in passages for p in plist]
        flat_cl = [c for plist, c in zip(passages, claims) for _ in plist]
        logits = self.predict_logits(flat_ev, flat_cl, show_progress)
        p_nei = softmax(logits)[:, LABELS.index("INSUFFICIENT")]
        out = np.zeros((len(claims), len(LABELS)), dtype=np.float32)
        start = 0
        for i, plist in enumerate(passages):
            end = start + len(plist)
            out[i] = logits[start + int(p_nei[start:end].argmin())]
            start = end
        return out

    def predict_proba_passages(self, passages: list[list[str]], claims: list[str], show_progress: bool = True):
        return softmax(self.predict_logits_passages(passages, claims, show_progress) / self.temperature)

    def predict_proba(self, evidence: list[str], claims: list[str], show_progress: bool = True) -> np.ndarray:
        """Probabilities, shape (n, 3), columns in LABELS order, using self.temperature.

        A temperature > 1 softens over-confident predictions; calibrate_verifier.py
        fits it on the validation split (step 4 of the training pipeline).
        """
        return softmax(self.predict_logits(evidence, claims, show_progress) / self.temperature)

    def verify(self, evidence: str, claim: str) -> tuple[str, float]:
        """Convenience for the pipeline: (label, confidence) for a single pair."""
        probs = self.predict_proba([evidence], [claim], show_progress=False)[0]
        best = int(probs.argmax())
        return LABELS[best], float(probs[best])


def make_passages(title: str, sentences: list[str], window: int = 3) -> list[str]:
    """Candidate evidence passages for one paper: the full abstract plus every window of
    `window` consecutive sentences, each prefixed with the title."""
    title = title.strip()
    sentences = [s.strip() for s in sentences if s.strip()]
    full = " ".join([title, *sentences])
    if len(sentences) <= window:
        return [full]
    windows = [" ".join([title, *sentences[i : i + window]]) for i in range(len(sentences) - window + 1)]
    return [full, *windows]


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for abstracts (split after . ! ? followed by a capital or digit)."""
    import re

    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text.strip()) if s]


def calibration_file(evidence: str) -> str:
    return CALIBRATION_FILE if evidence == "abstract" else f"calibration_{evidence}.json"


def column_order(id2label: dict) -> list[int]:
    """For each label in LABELS, the index of the model output that holds it.

    Works for off-the-shelf NLI models (entailment / contradiction / neutral, in
    any order) and for VeriLit checkpoints saved with the LABELS names.
    """
    found: dict[str, int] = {}
    for idx, name in id2label.items():
        label = label_from_model_name(str(name))
        if label is not None:
            found[label] = int(idx)
    missing = [label for label in LABELS if label not in found]
    if missing:
        raise ValueError(
            f"Cannot map the model's labels {dict(id2label)} to {LABELS}; missing {missing}. "
            "Use a 3-way NLI model (entailment / contradiction / neutral)."
        )
    return [found[label] for label in LABELS]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

