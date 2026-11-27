"""The VeriLit pipeline: question → Planner → Retriever → Claim Generator → Verifier → Synthesizer.

Claims the Verifier does not accept go through the feedback loop, up to
max_retries times:
  * INSUFFICIENT (no support in the cited abstract) → re-retrieve: search for
    the claim itself and check it against the best new papers;
  * CONTRADICTED, or still insufficient after re-retrieval → revise: the LLM
    rewrites the claim to say only what the abstract states.
A claim still not accepted after the limit is marked unverified: it is listed
in the report but not used in the review.

With verify=False the pipeline is the no-verification baseline from the
proposal: same LLM, same papers, same claims, no Verifier and no loop.

run() returns a JSON-ready record of everything (sub-questions, papers, every
claim with all its attempts, the review, stats); to_markdown() renders it.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from verilit.agents import (
    INSUFFICIENT, ClaimGenerator, Planner, Retriever, SubQuestion, Synthesizer, Verdict, Verifier,
)
from verilit.llm import LLM
from verilit.sources import Paper, PaperSearch

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    verify: bool = True  # False = the no-verification baseline
    retriever: str = "semantic"  # or "bm25"
    papers_per_question: int = 5
    max_sub_questions: int = 4
    max_claims: int = 3  # per sub-question
    max_retries: int = 2  # feedback-loop attempts per claim
    threshold: float = 0.7  # accept a claim when calibrated P(SUPPORTED) >= threshold
    evidence: str = "abstract"  # Verifier scores the full abstract, or "passages" (best 3-sentence window)
    verifier_path: str = str(ROOT / "models" / "verifier")


def _attempt(action: str, text: str | None, paper_id: str, v: Verdict | None) -> dict:
    return {
        "action": action,
        "text": text,
        "paper_id": paper_id,
        "label": v.label if v else None,
        "raw_label": v.raw_label if v else None,
        "confidence": round(v.confidence, 4) if v else None,
        "p_supported": round(v.p_supported, 4) if v else None,
        "accepted": v.accepted if v else False,
        "alt_p_supported": round(v.alt_p_supported, 4) if v and v.alt_p_supported is not None else None,
    }


class VeriLit:
    def __init__(self, config: Config | None = None, llm: LLM | None = None,
                 search: PaperSearch | None = None, verbose: bool = True, log_fn=print):
        self.config = config or Config()
        self.verbose = verbose
        self.log_fn = log_fn  # where progress messages go (print, or e.g. the demo UI)
        self.llm = llm or LLM()
        self.search = search or PaperSearch(verbose=verbose)
        c = self.config
        self.planner = Planner(self.llm, c.max_sub_questions)
        self.retriever = Retriever(self.search, c.retriever, c.papers_per_question)
        self.generator = ClaimGenerator(self.llm, c.max_claims)
        self.verifier = Verifier(c.verifier_path, c.threshold, c.evidence) if c.verify else None
        self.synthesizer = Synthesizer(self.llm)

    def reconfigure(self, config: Config) -> None:
        """Apply new per-run settings without reloading any model (used by the demo server)."""
        if config.verify != (self.verifier is not None) or config.verifier_path != self.config.verifier_path:
            raise ValueError("reconfigure can't switch verification on/off or change the Verifier; make a new VeriLit")
        self.config = config
        self.planner.max_sub_questions = config.max_sub_questions
        self.retriever.method = config.retriever
        self.retriever.k = config.papers_per_question
        self.generator.max_claims = config.max_claims
        if self.verifier is not None:
            self.verifier.threshold = config.threshold
            self.verifier.set_evidence(config.evidence)

    def log(self, msg: str) -> None:
        if self.verbose:
            self.log_fn(msg)

    # -- main entry point ----------------------------------------------------

    def run(self, question: str) -> dict:
        start = time.time()
        llm_before, hits_before = self.llm.requests, self.llm.cache_hits
        search_before = dict(self.search.requests)
        papers: dict[str, Paper] = {}

        self.log("Planner: splitting the question ...")
        sub_questions = self.planner.run(question)
        by_id = {sq.sq_id: sq for sq in sub_questions}

        claims: list[dict] = []
        sq_records = []
        for sq in sub_questions:
            self.log(f"{sq.sq_id}: {sq.question}")
            found = self.retriever.run(sq)
            papers.update({p.paper_id: p for p in found})
            self.log(f"  Retriever: {len(found)} papers")
            generated = self.generator.run(sq, found)
            self.log(f"  Claim Generator: {len(generated)} claims")
            sq_records.append({**asdict(sq), "paper_ids": [p.paper_id for p in found]})
            for c in generated:
                claims.append({
                    "id": f"C{len(claims) + 1}",
                    "sub_question": sq.sq_id,
                    "original": {"text": c.text, "paper_id": c.paper_id},
                    "attempts": [],
                    "final": {"text": c.text, "paper_id": c.paper_id, "label": None, "confidence": None},
                    "status": "unchecked",
                })

        if self.verifier is not None:
            self.log("Verifier: checking claims ...")
            for rec in claims:
                self._verify(rec, by_id[rec["sub_question"]], papers)
                f = rec["final"]
                self.log(f"  {rec['id']} {rec['status']:<10} {f['label'] or '':<12} "
                         f"(retries: {len(rec['attempts']) - 1}) {f['text'][:70]}")

        used = [c for c in claims if c["status"] in ("verified", "unchecked")]
        ref_of: dict[str, int] = {}
        for c in used:
            ref_of.setdefault(c["final"]["paper_id"], len(ref_of) + 1)
        references = [{"n": n, "paper_id": pid, "citation": papers[pid].citation()} for pid, n in ref_of.items()]
        gaps = [sq.question for sq in sub_questions if not any(c["sub_question"] == sq.sq_id for c in used)]

        self.log("Synthesizer: writing the review ...")
        review = self.synthesizer.run(
            question, sub_questions,
            [(c["final"]["text"], ref_of[c["final"]["paper_id"]], c["sub_question"]) for c in used], gaps,
        )
        cited = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", review) for n in re.findall(r"\d+", group)}
        invalid = sorted(cited - set(ref_of.values()))

        retried = [c for c in claims if len(c["attempts"]) > 1]
        stats = {
            "claims": len(claims),
            "verified": sum(c["status"] == "verified" for c in claims),
            "unverified": sum(c["status"] == "unverified" for c in claims),
            "unchecked": sum(c["status"] == "unchecked" for c in claims),
            "verified_first_try": sum(c["status"] == "verified" and len(c["attempts"]) == 1 for c in claims),
            "claims_retried": len(retried),
            "retry_successes": sum(c["status"] == "verified" for c in retried),
            "total_retries": sum(len(c["attempts"]) - 1 for c in retried),
            "sub_questions_covered": len(sub_questions) - len(gaps),
            "sub_questions": len(sub_questions),
            "invalid_citations_in_review": invalid,
            "llm_requests": self.llm.requests - llm_before,
            "llm_cache_hits": self.llm.cache_hits - hits_before,
            "search_requests": {k: self.search.requests[k] - search_before[k] for k in search_before},
            "seconds": round(time.time() - start, 1),
        }
        return {
            "question": question,
            "mode": "verilit" if self.verifier is not None else "baseline",
            "config": asdict(self.config),
            "llm": {"provider": self.llm.provider, "model": self.llm.model},
            "sub_questions": sq_records,
            "claims": claims,
            "references": references,
            "evidence_gaps": gaps,
            "review": review,
            "stats": stats,
            "papers": {pid: p.to_dict() for pid, p in papers.items()},
        }

    # -- the feedback loop ---------------------------------------------------

    def _verify(self, rec: dict, sq: SubQuestion, papers: dict[str, Paper]) -> None:
        text, pid = rec["original"]["text"], rec["original"]["paper_id"]
        v = self.verifier.run(text, papers[pid])
        rec["attempts"].append(_attempt("initial", text, pid, v))
        re_retrieved = False
        retries = 0
        while not v.accepted and retries < self.config.max_retries:
            retries += 1
            if v.label == INSUFFICIENT and not re_retrieved:
                re_retrieved = True
                candidates = self.retriever.find_evidence(text, exclude={pid})
                if candidates:
                    papers.update({p.paper_id: p for p in candidates})
                    verdicts = self.verifier.run_many(text, candidates)
                    best = max(range(len(candidates)), key=lambda i: verdicts[i].p_supported)
                    rec["attempts"].append(_attempt("re-retrieve", text, candidates[best].paper_id, verdicts[best]))
                    # Switch to the new paper only if it supports the claim better than the current one;
                    # otherwise keep the current paper for the revision step.
                    if verdicts[best].p_supported > v.p_supported:
                        pid, v = candidates[best].paper_id, verdicts[best]
                    continue
            new_text = self.generator.revise(sq, text, papers[pid], v.label)
            if not new_text:
                rec["attempts"].append(_attempt("revise", None, pid, None))
                break
            text = new_text
            v = self.verifier.run(text, papers[pid])
            rec["attempts"].append(_attempt("revise", text, pid, v))
        rec["final"] = {"text": text, "paper_id": pid, "label": v.label, "confidence": round(v.confidence, 4)}
        rec["status"] = "verified" if v.accepted else "unverified"


# -- report ------------------------------------------------------------------


def _cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def to_markdown(record: dict) -> str:
    papers = {pid: Paper.from_dict(d) for pid, d in record["papers"].items()}
    ref_of = {r["paper_id"]: r["n"] for r in record["references"]}
    s = record["stats"]
    mode = "VeriLit (with verification)" if record["mode"] == "verilit" else "Baseline (no verification)"
    out = [
        f"# {record['question']}",
        "",
        f"*{mode} · {record['llm']['provider']} {record['llm']['model']} · "
        f"{s['claims']} claims, {s['verified']} verified, {s['unverified']} unverified · {s['seconds']:.0f} s*",
        "",
        "## Literature review",
        "",
        record["review"],
        "",
        "## References",
        "",
        *[f"[{r['n']}] {r['citation']}" + "  " for r in record["references"]],
    ]
    if record["mode"] == "verilit":
        out += [
            "",
            "## Claim verification",
            "",
            "| | Claim | Source | Verdict | Confidence | Retries |",
            "|---|---|---|---|---:|---:|",
        ]
        for c in record["claims"]:
            f = c["final"]
            src = f"[{ref_of[f['paper_id']]}]" if f["paper_id"] in ref_of else _cell(papers[f["paper_id"]].title[:50])
            mark = "✓" if c["status"] == "verified" else "✗"
            out.append(f"| {mark} | {_cell(f['text'])} | {src} | {f['label']} | {f['confidence']:.2f} | "
                       f"{len(c['attempts']) - 1} |")
        unverified = [c for c in record["claims"] if c["status"] == "unverified"]
        if unverified:
            out += ["", "## Unverified claims (not used in the review)", ""]
            for c in unverified:
                f = c["final"]
                out.append(f"- {f['text']} — *{papers[f['paper_id']].title}* ({f['label']}, {f['confidence']:.2f})")
    if record["evidence_gaps"]:
        out += ["", "## Evidence gaps", "", *[f"- {g}" for g in record["evidence_gaps"]]]
    if s["invalid_citations_in_review"]:
        out += ["", f"> Warning: the review cites reference numbers that do not exist: {s['invalid_citations_in_review']}"]
    return "\n".join(out) + "\n"
