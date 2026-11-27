// VeriLit demo page. All text from papers and the LLM is inserted as text, never as HTML.
"use strict";

const $ = (sel) => document.querySelector(sel);
const EXAMPLES = [
  "How does retrieval-augmented generation reduce hallucination in large language models?",
  "How effective are deep learning methods for malware detection?",
  "What are prompt injection attacks on LLM-based applications and how can they be mitigated?",
];
const STATUS = { verified: "Verified", unverified: "Unverified", unchecked: "Not checked" };
const ACTION = {
  initial: "Checked against its cited paper",
  "re-retrieve": "Searched for better evidence",
  revise: "Rewrote the claim to match the abstract",
};
const STEPS = ["planner", "retriever", "generator", "verifier", "synthesizer"];

// Build an element: h("p", {class: "x"}, "text", child, ...). Strings become text nodes.
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const state = { results: {}, mode: "verilit", source: null, runs: [] };

// -- status -------------------------------------------------------------------------
async function loadStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const box = $("#status");
    box.replaceChildren(
      h("span", { class: `pill ${s.llm.ready ? "ok" : "off"}`, title: s.llm.message || "" },
        s.llm.ready ? `${s.llm.provider} · ${s.llm.model}` : "No LLM key"),
      h("span", { class: `pill ${s.verifier ? "ok" : "off"}`,
        title: s.verifier ? "" : "Train it with scripts/finetune_verifier.py" },
        s.verifier ? "Verifier ready" : "No Verifier"),
    );
    if (!s.llm.ready) showError(`${s.llm.message} Saved runs still work.`);
  } catch { /* the page still works for saved runs */ }
}

// -- views ----------------------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on);
  });
  $("#view-ask").hidden = name !== "ask";
  $("#view-saved").hidden = name !== "saved";
  const showResult = state.source === name && Object.keys(state.results).length > 0;
  $("#result").hidden = !showResult;
  if (name === "saved" && state.runs.length === 0) loadRuns();
}

// -- asking ----------------------------------------------------------------------------
function settings() {
  return {
    threshold: $("#threshold").value, max_retries: $("#retries").value, papers: $("#papers").value,
    max_claims: $("#claims").value, retriever: $("#retriever").value, evidence: $("#evidence").value,
    compare: $("#compare").checked,
  };
}

function showError(msg) { const e = $("#error"); e.textContent = msg; e.hidden = false; }

function setStep(step) {
  const idx = STEPS.indexOf(step);
  document.querySelectorAll("#stepper li").forEach((li, i) => {
    li.classList.toggle("done", idx >= 0 && i < idx);
    li.classList.toggle("active", i === idx);
  });
}

function stepFor(message) {
  if (message.startsWith("Planner")) return "planner";
  if (/^S\d+:/.test(message) || message.startsWith("Retriever")) return "retriever";
  if (message.startsWith("Claim Generator")) return "generator";
  if (message.startsWith("Verifier") || /^C\d+ /.test(message)) return "verifier";
  if (message.startsWith("Synthesizer")) return "synthesizer";
  return null;
}

function ask(question) {
  const params = new URLSearchParams({ question, ...settings() });
  const es = new EventSource(`/api/ask?${params}`);
  const started = Date.now();
  const timer = setInterval(() => { $("#progress-time").textContent = `${Math.round((Date.now() - started) / 1000)} s`; }, 500);
  let currentMode = null;

  state.results = {};
  state.source = "ask";
  $("#result").hidden = true;
  $("#error").hidden = true;
  $("#progress").hidden = false;
  $("#log").replaceChildren();
  $("#run").disabled = true;
  setStep("planner");

  const finish = () => { es.close(); clearInterval(timer); $("#run").disabled = false; };

  es.addEventListener("log", (ev) => {
    const { mode, message } = JSON.parse(ev.data);
    if (mode !== currentMode) {
      currentMode = mode;
      $("#progress-label").textContent = mode === "verilit" ? "VeriLit is working" : "Baseline (no verification) is working";
      $("#log").append(h("li", { class: "mode" }, mode === "verilit" ? "VeriLit" : "Baseline"));
      setStep("planner");
    }
    const step = stepFor(message);
    if (step) setStep(step);
    const log = $("#log");
    log.append(h("li", {}, message));
    log.scrollTop = log.scrollHeight;
  });
  es.addEventListener("result", (ev) => {
    const data = JSON.parse(ev.data);
    state.results[data.mode] = data;
    if (data.mode === "verilit" || !state.results.verilit) {
      state.mode = data.mode;
    }
    render();
  });
  es.addEventListener("failed", (ev) => { showError(JSON.parse(ev.data).message); finish(); });
  es.addEventListener("done", () => {
    finish();
    setStep(null);
    document.querySelectorAll("#stepper li").forEach((li) => li.classList.add("done"));
    $("#progress-label").textContent = "Done";
    state.runs = []; // refresh the saved list next time it's opened
  });
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    showError("Lost the connection to the VeriLit server. Is it still running?");
    finish();
  };
}

// -- saved runs ------------------------------------------------------------------------
async function loadRuns() {
  state.runs = await (await fetch("/api/runs")).json();
  renderRuns();
}

function renderRuns() {
  const q = $("#filter").value.trim().toLowerCase();
  const items = state.runs.filter((r) => !q || r.question.toLowerCase().includes(q) || r.id.toLowerCase().includes(q));
  $("#runlist").replaceChildren(...items.map((r) => h("li", {},
    h("button", { onclick: () => openRun(r.id) },
      h("span", { class: "q" }, r.question),
      h("span", { class: "meta" },
        h("span", { class: `tag ${r.mode}` }, r.mode === "verilit" ? "VeriLit" : "Baseline"),
        r.folder === "eval" ? h("span", { class: "tag" }, `test set · ${r.stem}${r.area ? " · " + r.area : ""}`) : h("span", { class: "tag" }, "asked here"),
        r.mode === "verilit" ? `${r.verified} of ${r.claims} claims verified` : `${r.claims} claims`)))));
  if (items.length === 0) $("#runlist").append(h("li", { class: "hint" }, "No saved runs match."));
}

async function openRun(id) {
  const data = await (await fetch(`/api/runs/${id}`)).json();
  state.results = { [data.record.mode]: data };
  state.mode = data.record.mode;
  state.source = "saved";
  render();
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// -- rendering a result -----------------------------------------------------------------
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function reviewParagraphs(text, refs, invalid) {
  const byN = new Map(refs.map((r) => [r.n, r]));
  return text.split(/\n\s*\n/).filter((p) => p.trim()).map((para) => {
    const p = h("p");
    let last = 0;
    for (const m of para.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
      p.append(para.slice(last, m.index));
      for (const n of m[1].split(",").map((x) => Number(x.trim()))) {
        const ref = byN.get(n);
        p.append(h("button", {
          class: `cite${ref ? "" : " bad"}`, type: "button",
          title: ref ? ref.citation : "This reference number does not exist",
          onclick: () => flashRef(n),
        }, String(n)));
      }
      last = m.index + m[0].length;
    }
    p.append(para.slice(last));
    return p;
  });
}

function flashRef(n) {
  const li = document.getElementById(`ref-${n}`);
  if (!li) return;
  li.scrollIntoView({ behavior: "smooth", block: "center" });
  li.classList.add("flash");
  setTimeout(() => li.classList.remove("flash"), 1400);
}

function meter(pSupported, threshold) {
  const pct = Math.round(pSupported * 100);
  return h("div", { class: "meter" },
    h("div", { class: "meter-row" }, h("span", {}, "P(supported)"),
      h("span", {}, h("b", {}, pSupported.toFixed(2)), `  ·  accept at ${threshold.toFixed(2)}`)),
    h("div", { class: `bar ${pSupported >= threshold ? "pass" : ""}`, role: "img",
      "aria-label": `P(supported) ${pSupported.toFixed(2)}, threshold ${threshold.toFixed(2)}` },
      h("i", { style: `width:${pct}%` }), h("u", { style: `left:calc(${threshold * 100}% - 1px)` })));
}

function trail(c, papers) {
  const steps = c.attempts.map((a) => {
    if (a.text === null) {
      return h("li", { class: "no" }, h("div", { class: "step" }, ACTION[a.action] || a.action),
        "The abstract says nothing relevant, so the claim was dropped.");
    }
    const paper = papers[a.paper_id];
    const alt = a.alt_p_supported != null ? ` · best passage ${a.alt_p_supported.toFixed(2)}` : "";
    return h("li", { class: a.accepted ? "ok" : "no" },
      h("div", { class: "step" }, `${ACTION[a.action] || a.action} → ${(a.label || "").toLowerCase()}`),
      h("div", { class: "num" }, `P(supported) ${a.p_supported.toFixed(2)}${alt}`),
      a.action !== "initial" ? h("div", { class: "txt" }, `“${a.text}”`) : null,
      h("div", {}, paper ? paper.title : a.paper_id));
  });
  const retries = c.attempts.length - 1;
  return h("details", { class: "trail" },
    h("summary", {}, retries ? `How it was verified · ${retries} ${retries === 1 ? "retry" : "retries"}` : "How it was verified"),
    h("ol", {}, steps));
}

function claimCard(c, rec, refOf) {
  const f = c.final;
  const paper = rec.papers[f.paper_id];
  const last = c.attempts.length ? c.attempts.at(-1) : null;
  const lastScored = [...c.attempts].reverse().find((a) => a.p_supported != null);
  return h("article", { class: `card ${c.status}` },
    h("div", { class: "card-top" },
      h("p", { class: "claim" }, f.text),
      h("span", { class: `badge ${c.status}` }, STATUS[c.status])),
    h("div", { class: "source" },
      refOf.has(f.paper_id) ? h("span", { class: "ref" }, `[${refOf.get(f.paper_id)}]`) : null,
      paper && paper.url ? h("a", { href: paper.url, target: "_blank", rel: "noopener noreferrer" }, paper.title) : (paper ? paper.title : f.paper_id),
      paper ? ` · ${paper.year || "n.d."}${paper.venue ? " · " + paper.venue : ""}` : ""),
    lastScored ? meter(lastScored.p_supported, rec.config.threshold) : null,
    last ? trail(c, rec.papers) : null);
}

function stat(value, label, cls = "") {
  return h("div", { class: `stat ${cls}` }, h("b", {}, value), h("span", {}, label));
}

function render() {
  const data = state.results[state.mode];
  if (!data) return;
  const rec = data.record;
  const s = rec.stats;
  const isVeri = rec.mode === "verilit";
  const refOf = new Map(rec.references.map((r) => [r.paper_id, r.n]));
  const modes = ["verilit", "baseline"];

  const seg = h("div", { class: "seg", role: "tablist" }, modes.map((m) => h("button", {
    class: state.mode === m ? "on" : "", disabled: !state.results[m], type: "button",
    title: state.results[m] ? "" : (state.source === "saved" ? "Not saved for this run" : "Still running, or not requested"),
    onclick: () => { state.mode = m; render(); },
  }, m === "verilit" ? "VeriLit" : "Baseline")));

  const stats = isVeri
    ? [stat(s.claims, "claims"), stat(s.verified, "verified", "good"), stat(s.unverified, "unverified", s.unverified ? "bad" : ""),
       stat(s.total_retries, "retries"), stat(`${s.sub_questions_covered}/${s.sub_questions}`, "sub-questions covered"),
       stat(`${Math.round(s.seconds)} s`, `${s.llm_requests} LLM requests`)]
    : [stat(s.claims, "claims, none checked"), stat(`${s.sub_questions_covered}/${s.sub_questions}`, "sub-questions covered"),
       stat(`${Math.round(s.seconds)} s`, `${s.llm_requests} LLM requests`)];

  const subqs = rec.sub_questions.map((sq) => {
    const cards = rec.claims.filter((c) => c.sub_question === sq.sq_id).map((c) => claimCard(c, rec, refOf));
    return cards.length ? [h("h3", {}, h("span", { class: "sq" }, sq.sq_id), sq.question), ...cards] : [];
  });

  const paper = h("div", { class: "paper" },
    h("p", { class: "eyebrow" }, isVeri ? "Literature review · verified claims only" : "Literature review · no verification"),
    h("div", { class: "review" }, reviewParagraphs(rec.review, rec.references, s.invalid_citations_in_review || [])),
    rec.evidence_gaps.length ? h("div", { class: "gaps" }, h("b", {}, "Evidence gaps: "), rec.evidence_gaps.join(" · ")) : null,
    h("div", { class: "refs" }, h("p", { class: "eyebrow" }, `References (${rec.references.length})`),
      h("ol", {}, rec.references.map((r) => h("li", { id: `ref-${r.n}`, value: r.n }, r.citation)))));

  $("#result").replaceChildren(
    h("div", { class: "result-head" },
      h("div", {},
        h("h1", {}, rec.question),
        h("div", { class: "result-meta" },
          `${rec.llm.provider} ${rec.llm.model} · ${rec.config.retriever} retrieval · accept at P(supported) ≥ ${rec.config.threshold}`,
          data.id ? ` · ${data.id}` : "")),
      h("div", { class: "result-tools" }, seg,
        h("button", { class: "ghost", type: "button", onclick: () => download("verilit-report.md", data.markdown, "text/markdown") }, "Report .md"),
        h("button", { class: "ghost", type: "button", onclick: () => download("verilit-record.json", JSON.stringify(rec, null, 2), "application/json") }, "Record .json"))),
    h("div", { class: "stats" }, stats),
    h("div", { class: "columns" }, paper, h("div", { class: "claims" }, h("p", { class: "eyebrow" }, `Claims (${rec.claims.length})`), subqs)));
  $("#result").hidden = false;
}

// -- wiring ------------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
$("#examples").replaceChildren(...EXAMPLES.map((q) => h("button", { class: "chip", type: "button", onclick: () => { $("#question").value = q; $("#question").focus(); } }, q)));
for (const [id, digits] of [["threshold", 2], ["retries", 0], ["papers", 0], ["claims", 0]]) {
  const input = $(`#${id}`), out = $(`#${id}-out`);
  input.addEventListener("input", () => { out.textContent = Number(input.value).toFixed(digits); });
}
$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim() || $("#question").placeholder;
  $("#question").value = q;
  ask(q);
});
$("#filter").addEventListener("input", renderRuns);
loadStatus();
