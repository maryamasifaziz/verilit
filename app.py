"""VeriLit demo: ask a research question and see a verified literature review.

Run:
    streamlit run app.py

Needs GEMINI_API_KEY (or VERILIT_LLM=groq + GROQ_API_KEY) for live questions, and the
fine-tuned Verifier in models/verifier for verification. "Browse saved runs" works with
neither: it shows earlier results from eval/runs/ and runs/ without calling any API.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from verilit.llm import LLM, LLMError
from verilit.pipeline import ROOT, Config, VeriLit, to_markdown
from verilit.sources import PaperSearch

st.set_page_config(page_title="VeriLit", page_icon="🔎", layout="wide")

STATUS = {
    "verified": ("✅", "Verified"),
    "unverified": ("❌", "Unverified"),
    "unchecked": ("⚪", "Not checked"),
}
ACTION = {"initial": "Checked against its cited paper", "re-retrieve": "Searched for better evidence",
          "revise": "Rewrote the claim to match the abstract"}
EXAMPLES = [
    "How does retrieval-augmented generation reduce hallucination in large language models?",
    "How effective are deep learning methods for malware detection?",
    "What are prompt injection attacks on LLM-based applications and how can they be mitigated?",
]


# -- cached resources (loaded once per app session) ---------------------------

@st.cache_resource(show_spinner="Connecting to the LLM ...")
def shared_clients() -> tuple[LLM, PaperSearch]:
    """One LLM client and one search client, so their rate limits are shared by both pipelines."""
    return LLM(), PaperSearch(verbose=False)


@st.cache_resource(show_spinner="Loading models (first time only) ...")
def get_pipeline(verify: bool, retriever: str, papers: int, max_claims: int, max_retries: int,
                 threshold: float, evidence: str) -> VeriLit:
    llm, search = shared_clients()
    config = Config(verify=verify, retriever=retriever, papers_per_question=papers, max_claims=max_claims,
                    max_retries=max_retries, threshold=threshold, evidence=evidence)
    return VeriLit(config, llm=llm, search=search)


# -- rendering -------------------------------------------------------------------

def render(record: dict) -> None:
    s = record["stats"]
    papers = record["papers"]
    ref_of = {r["paper_id"]: r["n"] for r in record["references"]}
    verified_mode = record["mode"] == "verilit"

    cols = st.columns(5 if verified_mode else 3)
    cols[0].metric("Claims", s["claims"])
    if verified_mode:
        cols[1].metric("Verified", s["verified"])
        cols[2].metric("Unverified", s["unverified"])
        cols[3].metric("Retries", s["total_retries"])
    cols[-1].metric("Time", f"{s['seconds']:.0f} s", help=f"{s['llm_requests']} LLM requests "
                    f"(+{s['llm_cache_hits']} from cache)")

    st.subheader("Literature review")
    st.markdown(record["review"])
    if s.get("invalid_citations_in_review"):
        st.warning(f"The review cites reference numbers that don't exist: {s['invalid_citations_in_review']}")

    with st.expander(f"References ({len(record['references'])})", expanded=False):
        for r in record["references"]:
            st.markdown(f"**[{r['n']}]** {r['citation']}")

    if record["evidence_gaps"]:
        st.info("**Evidence gaps** (no verified claim): " + "; ".join(record["evidence_gaps"]))

    st.subheader("Claims")
    by_sq = {sq["sq_id"]: sq["question"] for sq in record["sub_questions"]}
    for sq_id, sq_text in by_sq.items():
        claims = [c for c in record["claims"] if c["sub_question"] == sq_id]
        if not claims:
            continue
        st.markdown(f"**{sq_id}.** *{sq_text}*")
        for c in claims:
            icon, word = STATUS[c["status"]]
            f = c["final"]
            paper = papers[f["paper_id"]]
            ref = f"[{ref_of[f['paper_id']]}] " if f["paper_id"] in ref_of else ""
            with st.container(border=True):
                left, right = st.columns([5, 1])
                left.markdown(f"{icon} {f['text']}")
                link = f"[{paper['title']}]({paper['url']})" if paper.get("url") else paper["title"]
                left.caption(f"{ref}{link} ({paper.get('year') or 'n.d.'})")
                if f.get("confidence") is not None:
                    right.markdown(f"**{word}**")
                    right.caption(f"{f['label'].title()} · {f['confidence']:.2f}")
                else:
                    right.markdown(f"**{word}**")
                if len(c["attempts"]) > 0:
                    with st.expander(f"How it was verified ({len(c['attempts']) - 1} retries)"):
                        for a in c["attempts"]:
                            step = ACTION.get(a["action"], a["action"])
                            if a["text"] is None:
                                st.markdown(f"- {step}: the abstract says nothing relevant, so the claim was dropped.")
                                continue
                            title = papers[a["paper_id"]]["title"]
                            alt = (f", best passage {a['alt_p_supported']:.2f}"
                                   if a.get("alt_p_supported") is not None else "")
                            st.markdown(f"- **{step}** → {a['label'].title()} "
                                        f"(P(supported) {a['p_supported']:.2f}{alt})  \n"
                                        f"  *{a['text']}*  \n  Paper: {title}")

    st.download_button("Download report (.md)", to_markdown(record), file_name="verilit-report.md",
                       key=f"md-{record['mode']}-{id(record)}")
    st.download_button("Download full record (.json)", json.dumps(record, indent=2, ensure_ascii=False),
                       file_name="verilit-record.json", key=f"json-{record['mode']}-{id(record)}")


def run_live(pipeline: VeriLit, question: str, label: str) -> dict | None:
    with st.status(f"{label}: working ...", expanded=True) as status:
        pipeline.log_fn = lambda msg: status.write(msg.strip())
        try:
            record = pipeline.run(question)
        except LLMError as e:
            status.update(label=f"{label}: stopped", state="error")
            st.error(str(e))
            return None
        s = record["stats"]
        status.update(label=f"{label}: done in {s['seconds']:.0f} s ({s['llm_requests']} LLM requests)",
                      state="complete", expanded=False)
    return record


# -- page ------------------------------------------------------------------------

st.title("🔎 VeriLit")
st.caption("A self-verifying multi-agent research assistant: every claim in the review is checked "
           "against the abstract of the paper it cites.")

with st.sidebar:
    mode = st.radio("Mode", ["Ask a question", "Browse saved runs"])
    if mode == "Ask a question":
        st.header("Settings")
        compare = st.checkbox("Also show the baseline (no verification)", value=True,
                              help="Same LLM, papers and claims, without the Verifier. Costs about one "
                                   "extra LLM request, since the rest is cached.")
        threshold = st.slider("Accept a claim when P(supported) ≥", 0.5, 0.95, 0.7, 0.05)
        max_retries = st.slider("Retries per claim", 0, 3, 2)
        papers = st.slider("Papers per sub-question", 3, 8, 5)
        max_claims = st.slider("Claims per sub-question", 1, 4, 3)
        retriever = st.selectbox("Retrieval", ["semantic", "bm25"])
        evidence = st.selectbox("Verifier evidence", ["abstract", "passages"],
                                help="Full abstract (default) or the best 3-sentence passage.")

if mode == "Browse saved runs":
    files = sorted((ROOT / "eval" / "runs").glob("*.json")) + sorted((ROOT / "runs").glob("*.json"), reverse=True)
    if not files:
        st.info("No saved runs yet. Ask a question, or run scripts/run_eval_set.py.")
        st.stop()
    records = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rec = json.load(fh)
        tag = "VeriLit" if rec["mode"] == "verilit" else "baseline"
        folder = f.parent.relative_to(ROOT).as_posix()
        records[f"{folder}/{f.stem} · {tag} · {rec['question'][:80]}"] = rec
    choice = st.selectbox("Saved run", list(records))
    rec = records[choice]
    st.header(rec["question"])
    render(rec)
    st.stop()

verifier_dir = ROOT / "models" / "verifier"
if not verifier_dir.is_dir():
    st.warning("No Verifier found in models/verifier, so only the baseline (no verification) can run. "
               "Train it with scripts/finetune_verifier.py.")

question = st.text_input("Research question", placeholder=EXAMPLES[0])
st.caption("Examples: " + " · ".join(f"*{q}*" for q in EXAMPLES))
if st.button("Run VeriLit", type="primary", disabled=not question.strip()):
    try:
        shared_clients()
    except LLMError as e:
        st.error(f"{e} Set the key in your terminal before running `streamlit run app.py`.")
        st.stop()
    settings = dict(retriever=retriever, papers=papers, max_claims=max_claims, max_retries=max_retries,
                    threshold=threshold, evidence=evidence)
    results = {}
    if verifier_dir.is_dir():
        results["VeriLit"] = run_live(get_pipeline(True, **settings), question, "VeriLit")
    if compare or not verifier_dir.is_dir():
        results["Baseline (no verification)"] = run_live(get_pipeline(False, **settings), question, "Baseline")
    st.session_state["results"] = {k: v for k, v in results.items() if v}
    st.session_state["question"] = question

if st.session_state.get("results"):
    st.header(st.session_state["question"])
    tabs = st.tabs(list(st.session_state["results"]))
    for tab, rec in zip(tabs, st.session_state["results"].values()):
        with tab:
            render(rec)
