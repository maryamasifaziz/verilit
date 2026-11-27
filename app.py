"""VeriLit demo: a web interface for the pipeline.

Run:
    python app.py              # opens http://127.0.0.1:8000 in your browser

Asking a question needs GEMINI_API_KEY (or VERILIT_LLM=groq + GROQ_API_KEY) and, for
verification, the fine-tuned Verifier in models/verifier. "Saved runs" needs neither: it
shows earlier results from eval/runs/ and runs/ without calling any API.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from verilit.llm import LLM, PRESETS, LLMError
from verilit.pipeline import ROOT, Config, VeriLit, to_markdown
from verilit.sources import PaperSearch

WEB = ROOT / "web"
RUN_DIRS = {"eval": ROOT / "eval" / "runs", "runs": ROOT / "runs"}
SAFE_STEM = re.compile(r"^[A-Za-z0-9._-]+$")

app = FastAPI(title="VeriLit", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")

_run_lock = threading.Lock()  # one question at a time: the free-tier rate limits are shared
_build_lock = threading.Lock()
_state: dict = {}  # the LLM client, search client and the two pipelines, built on first use


def _pipelines() -> tuple[VeriLit, VeriLit | None]:
    """The VeriLit pipeline (if a Verifier exists) and the baseline, sharing clients and models."""
    with _build_lock:
        if "baseline" not in _state:
            llm, search = LLM(), PaperSearch(verbose=False)
            verilit = VeriLit(Config(verify=True), llm=llm, search=search) if _verifier_ready() else None
            baseline = VeriLit(Config(verify=False), llm=llm, search=search)
            if verilit is not None:
                baseline.retriever = verilit.retriever  # share the embedding model
            _state.update(verilit=verilit, baseline=baseline)
        return _state["verilit"], _state["baseline"]


def _verifier_ready() -> bool:
    return (Path(Config().verifier_path) / "config.json").exists()


def _llm_status() -> dict:
    provider = os.environ.get("VERILIT_LLM", "gemini")
    preset = PRESETS.get(provider)
    if preset is None:
        return {"ready": False, "message": f"Unknown VERILIT_LLM={provider!r}"}
    model = os.environ.get("VERILIT_LLM_MODEL") or preset["model"]
    if not os.environ.get(preset["key_env"]):
        return {"ready": False, "provider": provider, "model": model,
                "message": f"Set {preset['key_env']} in the terminal before starting the app."}
    return {"ready": True, "provider": provider, "model": model}


def _summary(rec: dict, folder: str, stem: str) -> dict:
    s = rec["stats"]
    return {"id": f"{folder}/{stem}", "folder": folder, "stem": stem, "question": rec["question"],
            "mode": rec["mode"], "area": rec.get("area", ""), "claims": s["claims"], "verified": s["verified"],
            "unverified": s["unverified"]}


def _save(record: dict) -> str:
    out = RUN_DIRS["runs"]
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", record["question"].lower()).strip("-")[:50].strip("-")
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}-{record['mode']}"
    with open(out / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    with open(out / f"{stem}.md", "w", encoding="utf-8") as fh:
        fh.write(to_markdown(record))
    return f"runs/{stem}"


# -- routes ------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    return {"llm": _llm_status(), "verifier": _verifier_ready(), "busy": _run_lock.locked()}


@app.get("/api/runs")
def list_runs():
    items = []
    for folder, d in RUN_DIRS.items():
        for f in sorted(d.glob("*.json"), reverse=(folder == "runs")):
            try:
                with open(f, encoding="utf-8") as fh:
                    items.append(_summary(json.load(fh), folder, f.stem))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    return items


@app.get("/api/runs/{folder}/{stem}")
def get_run(folder: str, stem: str):
    if folder not in RUN_DIRS or not SAFE_STEM.match(stem):
        raise HTTPException(404)
    path = RUN_DIRS[folder] / f"{stem}.json"
    if not path.is_file():
        raise HTTPException(404)
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)
    return {"id": f"{folder}/{stem}", "record": record, "markdown": to_markdown(record)}


@app.get("/api/ask")
def ask(question: str = Query(..., min_length=5, max_length=500),
        threshold: float = Query(0.7, ge=0.5, le=0.95), max_retries: int = Query(2, ge=0, le=3),
        papers: int = Query(5, ge=3, le=8), max_claims: int = Query(3, ge=1, le=4),
        retriever: str = Query("semantic", pattern="^(semantic|bm25)$"),
        evidence: str = Query("abstract", pattern="^(abstract|passages)$"), compare: bool = True):
    """Run a question; progress and results stream back as server-sent events."""
    events: queue.Queue = queue.Queue()

    def send(kind: str, data: dict) -> None:
        events.put((kind, data))

    def work() -> None:
        if not _run_lock.acquire(blocking=False):
            send("failed", {"message": "Another question is still running. Wait for it to finish."})
            events.put(None)
            return
        try:
            llm = _llm_status()
            if not llm["ready"]:
                raise LLMError(llm["message"])
            verilit, baseline = _pipelines()
            settings = dict(retriever=retriever, papers_per_question=papers, max_claims=max_claims,
                            max_retries=max_retries, threshold=threshold, evidence=evidence)
            runs = []
            if verilit is not None:
                runs.append(("verilit", verilit, Config(verify=True, **settings)))
            if compare or verilit is None:
                runs.append(("baseline", baseline, Config(verify=False, **settings)))
            for mode, pipeline, config in runs:
                pipeline.reconfigure(config)
                pipeline.log_fn = lambda msg, mode=mode: send("log", {"mode": mode, "message": msg.strip()})
                record = pipeline.run(question)
                run_id = _save(record)
                send("result", {"mode": mode, "id": run_id, "record": record, "markdown": to_markdown(record)})
        except LLMError as e:
            send("failed", {"message": str(e)})
        except Exception as e:  # shown in the page instead of a silent hang
            send("failed", {"message": f"{type(e).__name__}: {e}"})
        finally:
            if _run_lock.locked():
                _run_lock.release()
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            try:
                item = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                yield "event: done\ndata: {}\n\n"
                return
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print(f"VeriLit demo at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
