"""Fine-tune the Verifier's NLI model on SciFact (step 3 of the training pipeline).

Starts from the same off-the-shelf NLI model as the baseline and trains it on
data/processed/scifact/train.jsonl. After every epoch it scores the val split,
and the epoch with the best val macro-F1 is saved to models/verifier/. If you
give several learning rates, each is trained in turn and the best one overall
is kept, so the val split chooses both the learning rate and the number of
epochs. The dev split is not used here; it is kept for the final results.

Use a GPU (a free Google Colab T4 is enough). On a 4-core CPU it takes about
an hour per learning rate.

Usage:
    python scripts/finetune_verifier.py                         # one learning rate (2e-5)
    python scripts/finetune_verifier.py --lr 1e-5 2e-5 3e-5     # also choose the learning rate
    python scripts/finetune_verifier.py --limit 16 --epochs 1   # 1-minute check that it runs
Then:
    python scripts/eval_verifier.py --model models/verifier --name finetuned
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from verilit.labels import LABEL2ID, LABELS
from verilit.metrics import classification_metrics
from verilit.nli import NLIVerifier, pick_device

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_pairs(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/prepare_scifact.py")
    with open(path, encoding="utf-8") as fh:
        pairs = [json.loads(line) for line in fh if line.strip()]
    return pairs[:limit] if limit else pairs


def load_model(args, device: str) -> NLIVerifier:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    verifier = NLIVerifier(model, tokenizer, device, args.max_length, args.eval_batch_size)
    # Keep the pretrained NLI head (its entailment / contradiction / neutral
    # outputs already mean SUPPORTED / CONTRADICTED / INSUFFICIENT) and only
    # rename the outputs, so the saved model reports VeriLit's label names.
    model.config.id2label = {verifier.column_order[i]: label for i, label in enumerate(LABELS)}
    model.config.label2id = {label: idx for idx, label in model.config.id2label.items()}
    return verifier


def train_one_lr(args, lr: float, train_pairs, val_pairs, device: str, best: dict, history: list) -> None:
    set_seed(args.seed)
    verifier = load_model(args, device)
    model, order = verifier.model, verifier.column_order

    # Targets as the model's own output indices.
    targets = torch.tensor([order[LABEL2ID[p["label"]]] for p in train_pairs])

    class_weights = None
    if args.balanced:
        counts = torch.bincount(targets, minlength=len(LABELS)).float()
        class_weights = (counts.sum() / (len(LABELS) * counts.clamp(min=1))).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    batches_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_updates), total_updates)

    use_amp = args.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed)
    val_labels = [p["label"] for p in val_pairs]

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        perm = rng.permutation(len(train_pairs))
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for b in range(batches_per_epoch):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            enc = verifier.encode([train_pairs[i]["evidence"] for i in idx], [train_pairs[i]["claim"] for i in idx])
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), targets[idx].to(device))
            total_loss += loss.item() * len(idx)
            scaler.scale(loss / args.grad_accum).backward()
            if (b + 1) % args.grad_accum == 0 or b + 1 == batches_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        probs = verifier.predict_proba([p["evidence"] for p in val_pairs], [p["claim"] for p in val_pairs],
                                       show_progress=False)
        m = classification_metrics(val_labels, probs)
        record = {
            "lr": lr,
            "epoch": epoch,
            "train_loss": total_loss / len(train_pairs),
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_ece": m["ece"],
            "seconds": round(time.time() - start, 1),
        }
        history.append(record)

        improved = m["macro_f1"] > best.get("val_macro_f1", -1.0)
        if improved:
            best.update(record)
            model.save_pretrained(args.out)
            verifier.tokenizer.save_pretrained(args.out)
        print(
            f"lr {lr:g}  epoch {epoch}/{args.epochs}  train loss {record['train_loss']:.4f}  "
            f"val acc {m['accuracy']:.3f}  val macro-F1 {m['macro_f1']:.3f}  val ECE {m['ece']:.3f}  "
            f"({record['seconds']:.0f}s){'  <- best so far, saved' if improved else ''}"
        )

    del verifier, model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="starting NLI model (same as the baseline)")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "scifact")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "verifier")
    parser.add_argument("--lr", type=float, nargs="+", default=[2e-5], help="one or more learning rates to try")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup", type=float, default=0.1, help="share of updates used for LR warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--balanced", action="store_true",
                        help="weight the loss by inverse label frequency (CONTRADICTED is the rarest)")
    parser.add_argument("--fp16", action="store_true", help="mixed precision on GPU (faster; turn off if loss becomes nan)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="only the first N train and val pairs (quick test)")
    args = parser.parse_args()

    device = pick_device()
    train_pairs = read_pairs(args.data / "train.jsonl", args.limit)
    val_pairs = read_pairs(args.data / "val.jsonl", args.limit)
    print(f"model: {args.model}   device: {device}   train pairs: {len(train_pairs)}   val pairs: {len(val_pairs)}")
    if device == "cpu" and not args.limit:
        print("warning: no GPU found. On a CPU, training takes about an hour per learning rate; "
              "use Colab, or --limit 16 --epochs 1 for a quick test.")

    args.out.mkdir(parents=True, exist_ok=True)
    best: dict = {}
    history: list[dict] = []
    for lr in args.lr:
        train_one_lr(args, lr, train_pairs, val_pairs, device, best, history)

    log = {
        "base_model": args.model,
        "best": best,
        "history": history,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
    }
    with open(args.out / "training_log.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    print(
        f"\nBest on val: lr {best['lr']:g}, epoch {best['epoch']}, macro-F1 {best['val_macro_f1']:.3f}. "
        f"Saved to {args.out}\nNext: python scripts/eval_verifier.py --model {args.out} --name finetuned"
    )


if __name__ == "__main__":
    main()
