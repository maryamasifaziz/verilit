"""Run VeriLit on one research question.

Usage:
    python scripts/run_pipeline.py "How does retrieval-augmented generation reduce hallucination in LLMs?"
    python scripts/run_pipeline.py "..." --no-verify        # the no-verification baseline
    python scripts/run_pipeline.py "..." --retriever bm25   # keyword retrieval instead of semantic

Needs GEMINI_API_KEY (or VERILIT_LLM=groq and GROQ_API_KEY), and the fine-tuned
Verifier in models/verifier (unless --no-verify). Writes runs/<time>-<question>-<mode>.json
(everything) and .md (the report). Responses are cached, so running the same question
again, or the baseline after VeriLit, reuses earlier LLM calls and searches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from verilit.llm import LLMError
from verilit.pipeline import ROOT, Config, VeriLit, to_markdown


def slug(text: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--no-verify", action="store_true", help="baseline: skip the Verifier and feedback loop")
    parser.add_argument("--retriever", default="semantic", choices=["semantic", "bm25"])
    parser.add_argument("--papers", type=int, default=5, help="papers per sub-question")
    parser.add_argument("--max-claims", type=int, default=3, help="claims per sub-question")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.7, help="accept when calibrated P(SUPPORTED) >= this")
    parser.add_argument("--verifier", default=str(ROOT / "models" / "verifier"))
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    config = Config(
        verify=not args.no_verify, retriever=args.retriever, papers_per_question=args.papers,
        max_claims=args.max_claims, max_retries=args.max_retries, threshold=args.threshold,
        verifier_path=args.verifier,
    )
    if config.verify and not Path(config.verifier_path).is_dir():
        sys.exit(f"No Verifier at {config.verifier_path}. Train it (scripts/finetune_verifier.py) or use --no-verify.")

    try:
        record = VeriLit(config).run(args.question)
    except LLMError as e:
        sys.exit(f"LLM error: {e}")

    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(args.question)}-{record['mode']}"
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    markdown = to_markdown(record)
    with open(base.with_suffix(".md"), "w", encoding="utf-8") as fh:
        fh.write(markdown)

    s = record["stats"]
    print("\n" + markdown)
    print(f"{s['claims']} claims: {s['verified']} verified ({s['verified_first_try']} on the first try, "
          f"{s['retry_successes']} after retries), {s['unverified']} unverified, {s['unchecked']} unchecked. "
          f"LLM requests: {s['llm_requests']} (+{s['llm_cache_hits']} from cache). {s['seconds']:.0f} s.")
    print(f"Saved {base.with_suffix('.md')} and .json")


if __name__ == "__main__":
    main()
