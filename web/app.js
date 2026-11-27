// VeriLit demo page. All text from papers and the LLM is inserted as text, never as HTML.
"use strict";

const $ = (sel) => document.querySelector(sel);
const SVG = "http://www.w3.org/2000/svg";
const EXAMPLES = [
  "How does retrieval-augmented generation reduce hallucination in large language models?",
  "How effective are deep learning methods for malware detection?",
  "What are prompt injection attacks on LLM-based applications and how can they be mitigated?",
];
const STATUS = {
  verified: { word: "Verified", cls: "ok", icon: "i-check" },
  unverified: { word: "Unverified", cls: "bad", icon: "i-x" },
  unchecked: { word: "Not checked", cls: "grey", icon: "i-minus" },
};
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
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// An icon from the sprite in index.html.
function icon(name, extraClass = "") {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", `ic ${extraClass}`.trim());
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG, "use");
  use.setAttribute("href", `#${name}`);
  svg.append(use);
  return svg;
}

// A button that shows/hides the element it controls (aria-expanded).
function toggle(button, panel) {
  button.addEventListener("click", () => {
    const open = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!open));
    panel.hidden = open;
  });
}

let uid = 0;
const nextId = (p) => `${p}-${++uid}`;
const state = { results: {}, mode: "verilit", source: null, runs: [] };

// -- status ---------------------------------------------------------------------------
async function loadStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const pill = (ok, text, title) => h("span", { class: `pill ${ok ? "" : "bad"}`, title },
      h("span", { class: `dot ${ok ? "" : "bad"}` }), text);
    $("#status").replaceChildren(
      pill(s.llm.ready, s.llm.ready ? `${s.llm.provider} · ${s.llm.model}` : "No LLM key", s.llm.message || ""),
      pill(s.verifier, s.verifier ? "Verifier ready" : "Verifier not ready",
        s.verifier ? "" : "Train it with scripts/finetune_verifier.py"));
    if (!s.llm.ready) showError(s.llm.message);
  } catch { /* the page still works for saved runs */ }
}

// -- views -----------------------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    if (t.dataset.view === name) t.setAttribute("aria-current", "page");
    else t.removeAttribute("aria-current");
  });
  $("#view-ask").hidden = name !== "ask";
  $("#view-saved").hidden = name !== "saved";
  $("#result").hidden = !(state.source === name && Object.keys(state.results).length);
  if (name === "saved" && state.runs.length === 0) loadRuns();
}

function showError(msg) { $("#error-msg").textContent = msg; $("#error").hidden = false; }

// -- asking ----------------------------------------------------------------------------
function settings() {
  return {
    threshold: $("#threshold").value, max_retries: $("#retries").value, papers: $("#papers").value,
    max_claims: $("#claims").value, retriever: $("#retriever").value, evidence: $("#evidence").value,
    compare: $("#compare").checked,
  };
}

function setStep(step, allDone = false) {
  const idx = allDone ? STEPS.length : STEPS.indexOf(step);
  document.querySelectorAll("#stepper li").forEach((li, i) => {
    const done = i < idx, active = i === idx;
    li.className = done ? "done" : active ? "active" : "waiting";
    if (active) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
    const marker = li.querySelector(".marker");
    marker.replaceChildren(done ? icon("i-check") : active ? "" : String(i + 1));
    li.querySelector(".state").textContent = done ? "Done" : active ? "Running" : "Waiting";
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

function logLine(started, message) {
  const s = Math.round((Date.now() - started) / 1000);
  const t = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  const parts = message.split(/\b(SUPPORTED|CONTRADICTED|INSUFFICIENT)\b/);
  return h("div", {}, h("span", { class: "t" }, t),
    parts.map((p) => p === "SUPPORTED" ? h("span", { class: "okk" }, p)
      : (p === "CONTRADICTED" || p === "INSUFFICIENT") ? h("span", { class: "bd" }, p) : p));
}

function ask(question) {
  const params = new URLSearchParams({ question, ...settings() });
  const es = new EventSource(`/api/ask?${params}`);
  const started = Date.now();
  const timer = setInterval(() => {
    $("#progress-time").textContent = `${Math.round((Date.now() - started) / 1000)} s elapsed`;
  }, 500);
  let currentMode = null;

  state.results = {};
  state.source = "ask";
  $("#result").hidden = true;
  $("#error").hidden = true;
  $("#progress").hidden = false;
  $("#progress").classList.remove("finished");
  $("#progress-text").textContent = "VeriLit is working";
  $("#log").replaceChildren();
  $("#run").disabled = true;
  setStep("planner");

  const finish = () => { es.close(); clearInterval(timer); $("#run").disabled = false; };

  es.addEventListener("log", (ev) => {
    const { mode, message } = JSON.parse(ev.data);
    const log = $("#log");
    if (mode !== currentMode) {
      currentMode = mode;
      $("#progress-text").textContent = mode === "verilit" ? "VeriLit is working" : "Baseline (no verification) is working";
      log.append(h("div", { class: "mode" }, mode === "verilit" ? "VeriLit" : "Baseline, without verification"));
      setStep("planner");
    }
    const step = stepFor(message);
    if (step) setStep(step);
    log.append(logLine(started, message));
    log.scrollTop = log.scrollHeight;
  });
  es.addEventListener("result", (ev) => {
    const data = JSON.parse(ev.data);
    state.results[data.mode] = data;
    if (data.mode === "verilit" || !state.results.verilit) state.mode = data.mode;
    render();
  });
  es.addEventListener("failed", (ev) => { showError(JSON.parse(ev.data).message); finish(); });
  es.addEventListener("done", () => {
    finish();
    setStep(null, true);
    $("#progress").classList.add("finished");
    $("#progress-text").textContent = "Done";
    state.runs = []; // refresh the saved list the next time it opens
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

function runTags(r) {
  return [
    h("span", { class: `tag ${r.mode === "verilit" ? "v" : ""}` }, r.mode === "verilit" ? "VeriLit" : "Baseline"),
    h("span", { class: "tag" }, r.folder === "eval" ? `test set · ${r.stem}${r.area ? " · " + r.area : ""}` : "asked here"),
  ];
}

function renderRuns() {
  const q = $("#filter").value.trim().toLowerCase();
  const items = state.runs.filter((r) => !q || `${r.question} ${r.id} ${r.area} ${r.mode}`.toLowerCase().includes(q));
  $("#runlist").replaceChildren(...items.map((r) => h("li", {},
    h("button", { type: "button", class: "runcard", onclick: () => openRun(r.id) },
      h("span", { class: "q" }, r.question),
      h("span", { class: "tags" }, runTags(r)),
      h("span", { class: "foot" },
        r.mode === "verilit"
          ? [icon("i-check"), h("b", {}, `${r.verified} of ${r.claims}`), " claims verified"]
          : [icon("i-minus"), h("b", {}, r.claims), " claims · not checked"])))));
  if (items.length === 0) $("#runlist").append(h("li", { class: "empty" }, "No saved runs match."));
}

async function openRun(id) {
  const data = await (await fetch(`/api/runs/${id}`)).json();
  state.results = { [data.record.mode]: data };
  state.mode = data.record.mode;
  state.source = "saved";
  render();
  $("#result").scrollIntoView({ block: "start" });
}

// -- rendering a result -------------------------------------------------------------------
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function citeLink(n, ref) {
  return h("a", {
    class: `cite ${ref ? "" : "missing"}`, href: `#ref-${n}`,
    title: ref ? ref.citation : "This reference number does not exist",
    "aria-label": `Reference ${n}`,
  }, `[${n}]`);
}

function reviewParagraphs(text, refs) {
  const byN = new Map(refs.map((r) => [r.n, r]));
  return text.split(/\n\s*\n/).filter((p) => p.trim()).map((para) => {
    const p = h("p");
    let last = 0;
    for (const m of para.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
      p.append(para.slice(last, m.index));
      m[1].split(",").map((x) => Number(x.trim())).forEach((n, i) => {
        if (i) p.append(" ");
        p.append(citeLink(n, byN.get(n)));
      });
      last = m.index + m[0].length;
    }
    p.append(para.slice(last));
    return p;
  });
}

function confidence(pSupported, threshold) {
  const pass = pSupported >= threshold;
  return h("div", { class: "conf" },
    h("div", { class: "conf-label" }, "P(supported) ", h("b", {}, pSupported.toFixed(2)), ` · accept at ${threshold.toFixed(2)}`),
    h("div", {
      class: `bar ${pass ? "" : "bad"}`, role: "img",
      style: `--p:${Math.round(pSupported * 100)}%;--t:${threshold * 100}%`,
      "aria-label": `P(supported) ${pSupported.toFixed(2)}, ${pass ? "at or above" : "below"} accept threshold ${threshold.toFixed(2)}`,
    }, h("span", { class: "fill" }), h("span", { class: "tick" })));
}

function notChecked(threshold) {
  return h("div", { class: "conf" },
    h("div", { class: "conf-label" }, "P(supported) ", h("b", {}, "—"), " · not verified"),
    h("div", { class: "bar empty", role: "img", style: `--t:${threshold * 100}%`, "aria-label": "Not verified" },
      h("span", { class: "tick" })));
}

function timeline(c, papers, maxRetries) {
  const items = c.attempts.map((a) => {
    if (a.text === null) {
      return h("li", { class: "bad" }, `${ACTION[a.action] || a.action}: the abstract says nothing relevant, so the claim was dropped.`);
    }
    const ok = a.accepted;
    const paper = papers[a.paper_id];
    return h("li", { class: ok ? "ok" : "bad" },
      ACTION[a.action] || a.action, h("span", { class: "arrow" }, "→"),
      h("span", { class: `o ${ok ? "ok" : "bad"}` }, `${(a.label || "").toLowerCase()} (P ${a.p_supported.toFixed(2)})`),
      a.alt_p_supported != null ? h("span", { class: "paper-line" }, `Best 3-sentence passage: P ${a.alt_p_supported.toFixed(2)}`) : null,
      a.action === "revise" ? h("em", {}, `“${a.text}”`) : null,
      a.action === "re-retrieve" && paper ? h("span", { class: "paper-line" }, `Paper: ${paper.title}`) : null);
  });
  if (c.status === "unverified") {
    items.push(h("li", {}, c.attempts.length - 1 >= maxRetries
      ? "Retry limit reached. The claim stays Unverified and is left out of the review."
      : "No further retry could help. The claim stays Unverified and is left out of the review."));
  }
  return h("ol", { class: "timeline" }, items);
}

function claimCard(c, rec, refOf) {
  const f = c.final;
  const st = STATUS[c.status];
  const paper = rec.papers[f.paper_id];
  const scored = [...c.attempts].reverse().find((a) => a.p_supported != null);
  const retries = Math.max(0, c.attempts.length - 1);
  const ref = refOf.get(f.paper_id);

  let how = null;
  if (c.attempts.length) {
    const panelId = nextId("how");
    const open = c.status === "unverified";
    const btn = h("button", { type: "button", class: "how", "aria-expanded": String(open), "aria-controls": panelId },
      icon("i-chev"), `How it was verified · ${retries ? `${retries} ${retries === 1 ? "retry" : "retries"}` : "no retries"}`);
    const panel = h("div", { class: "how-panel", id: panelId, hidden: !open }, timeline(c, rec.papers, rec.config.max_retries));
    toggle(btn, panel);
    how = [btn, panel];
  }

  return h("article", { class: `claim ${st.cls}` },
    h("div", { class: "claim-top" }, h("span", { class: "cid" }, c.id),
      h("span", { class: `badge ${st.cls}` }, icon(st.icon), st.word)),
    h("p", { class: "claim-text" }, f.text),
    h("p", { class: "source" },
      ref ? [citeLink(ref, { citation: rec.references.find((r) => r.n === ref).citation }), " "] : null,
      paper && paper.url ? h("a", { class: "title", href: paper.url, target: "_blank", rel: "noopener noreferrer" }, paper.title)
        : (paper ? paper.title : f.paper_id),
      paper ? ` · ${paper.year || "n.d."}${paper.venue ? " · " + paper.venue : ""}` : ""),
    scored ? confidence(scored.p_supported, rec.config.threshold) : notChecked(rec.config.threshold),
    how);
}

function stat(value, label, cls = "") {
  return h("li", { class: `stat ${cls}` }, h("div", { class: "num" }, value), h("div", { class: "lbl2" }, label));
}

function render() {
  const data = state.results[state.mode];
  if (!data) return;
  const rec = data.record;
  const s = rec.stats;
  const isVeri = rec.mode === "verilit";
  const refOf = new Map(rec.references.map((r) => [r.paper_id, r.n]));

  const seg = h("div", { class: "seg", role: "group", "aria-label": "Result variant" },
    ["verilit", "baseline"].map((m) => h("button", {
      type: "button", "aria-pressed": String(state.mode === m), disabled: !state.results[m],
      title: state.results[m] ? "" : (state.source === "saved" ? "Not saved for this run" : "Still running, or not requested"),
      onclick: () => { state.mode = m; render(); },
    }, m === "verilit" ? "VeriLit" : "Baseline")));

  const stats = isVeri
    ? [stat(s.claims, "claims"), stat(s.verified, "verified", "ok"), stat(s.unverified, "unverified", s.unverified ? "bad" : ""),
       stat(s.total_retries, "retries"), stat(`${s.sub_questions_covered}/${s.sub_questions}`, "sub-questions covered"),
       stat(`${Math.round(s.seconds)} s`, `${s.llm_requests} LLM requests`)]
    : [stat(s.claims, "claims, none checked"), stat(`${s.sub_questions_covered}/${s.sub_questions}`, "sub-questions covered"),
       stat(`${Math.round(s.seconds)} s`, `${s.llm_requests} LLM requests`)];

  const paper = h("article", { class: "card paper", "aria-labelledby": "review-title" },
    h("span", { class: "kicker" }, isVeri ? "VeriLit · verified synthesis" : "Baseline · no verification"),
    h("h3", { id: "review-title" }, "Literature review"),
    h("div", { class: "prose" }, reviewParagraphs(rec.review, rec.references)),
    rec.evidence_gaps.length ? h("aside", { class: "gaps", "aria-labelledby": "gaps-title" }, icon("i-gap"),
      h("div", {}, h("h4", { id: "gaps-title" }, "Evidence gaps"),
        h("ul", {}, rec.evidence_gaps.map((g) => h("li", {}, `No verified claim for: ${g}`))))) : null,
    h("section", { class: "refs", "aria-labelledby": "refs-title" },
      h("h4", { id: "refs-title" }, "References"),
      h("ol", {}, rec.references.map((r) => h("li", { id: `ref-${r.n}` }, r.citation)))));

  const groups = rec.sub_questions.map((sq) => {
    const claims = rec.claims.filter((c) => c.sub_question === sq.sq_id);
    if (!claims.length) return null;
    const verified = claims.filter((c) => c.status === "verified").length;
    const groupId = nextId("group");
    const btn = h("button", { type: "button", class: "sq-btn", "aria-expanded": "true", "aria-controls": groupId },
      h("span", { class: "sq-id" }, sq.sq_id), h("span", { class: "txt" }, sq.question), icon("i-chev"));
    const body = h("div", { id: groupId }, claims.map((c) => claimCard(c, rec, refOf)));
    toggle(btn, body);
    return h("section", { class: "group" },
      h("h4", {}, btn),
      h("p", { class: "sq-sum" }, isVeri ? `${claims.length} claims · ${verified} verified` : `${claims.length} claims · not checked`),
      body);
  });

  const claimsCol = h("div", { class: "claims" },
    h("div", { class: "claims-head" }, h("h3", {}, "Claims"),
      h("div", { class: "legend", "aria-label": "Legend" },
        Object.values(STATUS).map((st) => h("span", { class: `badge ${st.cls}` }, icon(st.icon), st.word)))),
    isVeri ? null : h("div", { class: "base-note" }, icon("i-info"),
      h("span", {}, "In the Baseline view the Verifier is skipped, so claims are shown exactly as the generator wrote them.")),
    groups);

  $("#result").replaceChildren(
    h("div", { class: "rhead" },
      h("span", { class: "kicker" }, `Literature review${data.id ? " · " + data.id : ""}`),
      h("h2", { id: "result-title" }, rec.question),
      h("p", { class: "meta" }, `${rec.llm.model} · ${rec.config.retriever} retrieval · accept at P(supported) ≥ ${rec.config.threshold}`)),
    h("div", { class: "toolbar" }, seg,
      h("div", { class: "btns" },
        h("button", { type: "button", class: "btn sm", onclick: () => download("verilit-report.md", data.markdown, "text/markdown") }, icon("i-down"), "Report .md"),
        h("button", { type: "button", class: "btn sm", onclick: () => download("verilit-record.json", JSON.stringify(rec, null, 2), "application/json") }, icon("i-down"), "Record .json"))),
    h("ul", { class: "stats", "aria-label": "Run summary" }, stats),
    h("div", { class: "cols" }, paper, claimsCol));
  $("#result").hidden = false;
}

// -- wiring ---------------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
document.querySelectorAll("[data-toggle]").forEach((b) => toggle(b, document.getElementById(b.getAttribute("aria-controls"))));
$("#examples").replaceChildren(h("span", { class: "hint" }, "Try:"),
  ...EXAMPLES.map((q) => h("button", { type: "button", class: "chip", onclick: () => { $("#question").value = q; $("#question").focus(); } }, q)));
for (const [id, digits] of [["threshold", 2], ["retries", 0], ["papers", 0], ["claims", 0]]) {
  const input = $(`#${id}`), out = $(`#${id}-out`);
  input.addEventListener("input", () => { out.textContent = Number(input.value).toFixed(digits); });
}
$("#error-dismiss").addEventListener("click", () => { $("#error").hidden = true; });
$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim() || EXAMPLES[0];
  $("#question").value = q;
  ask(q);
});
$("#filter").addEventListener("input", renderRuns);
loadStatus();
