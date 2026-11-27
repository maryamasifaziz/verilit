# VeriLit: Self-Verifying Multi-Agent Research Assistant

VeriLit answers a research question with a short literature review. Before it presents a
claim, it checks that the claim is supported by the paper it cites.

```
question → Planner → Retriever → Claim Generator → Verifier ─┬─ supported → Synthesizer → review
                         ▲                                    ├─ not supported → back to Retriever (up to the retry limit)
                         └────────────────────────────────────┘
                                                              └─ still unsupported → marked "unverified"
```

Team: Maryam Asif (23k-0505), Nimra Ashraf (23k-0867)

## Setup

Python 3.10 or newer.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Google Colab** (use this for fine-tuning: Runtime → Change runtime type → T4 GPU).
Clone the repo, or upload `verilit.zip` and run `!unzip -q verilit.zip`, then:
```python
%cd verilit
!pip install -q -e .
```

The first run downloads the NLI model from Hugging Face (about 570 MB); later runs reuse it.

## Verifier

### 1. Data and baseline

```bash
python scripts/download_scifact.py      # SciFact → data/raw/scifact/
python scripts/prepare_scifact.py       # claim–evidence pairs → data/processed/scifact/
python scripts/eval_verifier.py         # off-the-shelf NLI baseline on the dev split
```

The baseline took 72 s on a 4-core CPU; it's faster on a GPU. Add `--limit 20` for a quick
test run. Results go to `results/baseline_dev/metrics.json`.

### 2. Fine-tuning (on a GPU)

```bash
python scripts/finetune_verifier.py                       # lr 2e-5, 5 epochs
python scripts/eval_verifier.py --model models/verifier --name finetuned
```

The script trains on `train`, scores `val` after every epoch and saves the best epoch to
`models/verifier/`, with a `training_log.json` of every epoch's val scores. To let `val`
choose the learning rate as well, give several: `--lr 1e-5 2e-5 3e-5`. Other options:
`--balanced` (weights the loss toward the rarer CONTRADICTED label) and `--fp16` (faster on
GPU). Before a long run, check that everything works with
`--limit 16 --epochs 1` (about a minute, even on a CPU).

Download `models/verifier/` from Colab before the session ends (it is about 570 MB), or
save it to Google Drive.

### 3. Calibration

```bash
python scripts/calibrate_verifier.py
python scripts/eval_verifier.py --model models/verifier --calibrated --name calibrated
```

Temperature scaling: the model's logits are divided by one number T, fitted on `val`, so its
probabilities match how often it is right. Predictions (and so accuracy and F1) don't change;
only the confidence scores do. T is saved to `models/verifier/calibration.json`. In the
pipeline, load the Verifier with `NLIVerifier.load("models/verifier", use_calibration=True)`
and its confidence scores are calibrated automatically.

The script also prints, for thresholds 0.5–0.9, how many `val` pairs would be accepted as
SUPPORTED and how many of those really are. Use it to choose the pipeline's acceptance
threshold (`val` has only 47 SUPPORTED pairs, so treat these numbers as rough guides):

| accept if P(SUPPORTED) ≥ | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|
| precision (accepted pairs really SUPPORTED) | 0.769 | 0.806 | 0.844 | 0.885 | 1.000 |
| recall (SUPPORTED pairs accepted) | 0.638 | 0.617 | 0.574 | 0.489 | 0.298 |

### 4. Full abstract vs. best passage

On a long abstract the evidence for a claim can be one sentence among many, which dilutes it.
With `--evidence passages`, the Verifier scores the claim against every window of 3
consecutive sentences (plus the full abstract) and keeps the window that is most sure it found
relevant evidence (lowest INSUFFICIENT probability). It has its own calibration:

```bash
python scripts/calibrate_verifier.py --evidence passages
python scripts/eval_verifier.py --model models/verifier --calibrated --evidence passages --name calibrated_passages
```

The choice was made on `val` before looking at dev: macro-F1 0.791 (full abstract) vs. 0.781
(passages), so **full-abstract scoring is the default**. Dev agrees:

| SciFact dev, calibrated | accuracy | macro F1 | ECE | SUPPORTED P / R |
|---|---:|---:|---:|---:|
| Full abstract (default) | **0.756** | **0.738** | **0.062** | 0.815 / 0.768 |
| Best passage | 0.741 | 0.728 | 0.086 | 0.767 / 0.812 |

SciFact can't show whether passages help on AI papers, where the Verifier wasn't trained (one
early VeriLit claim, stated word for word in a long abstract, scored 0.36 on the full abstract
and 0.89 on its best passage). So the pipeline records the passage score of every claim too,
and `scripts/score_eval.py` compares both against the hand labels on the custom test set.

### How the SciFact pairs are built

| | |
|---|---|
| **One pair** | one claim + one paper the claim cites |
| **Evidence text** | the paper's title + full abstract (the same kind of text the Retriever gets from arXiv / Semantic Scholar) |
| **Labels** | SUPPORT → `SUPPORTED`, CONTRADICT → `CONTRADICTED`, cited but no rationale → `INSUFFICIENT` |
| **NLI mapping** | evidence = premise, claim = hypothesis; entailment / contradiction / neutral → the three labels above |
| **Splits** | `train` (fine-tuning) and `val` (settings + calibration) are a split of SciFact train **by claim**, so no claim is in both; `dev` is the official dev split, kept untouched for final results (SciFact's test labels are not public) |

| split | claims | pairs | SUPPORTED | CONTRADICTED | INSUFFICIENT |
|---|---:|---:|---:|---:|---:|
| train | 688 | 780 | 323 | 167 | 290 |
| val | 121 | 139 | 47 | 27 | 65 |
| dev | 300 | 340 | 138 | 71 | 131 |

Each record also keeps the gold rationale sentences, which the retrieval Recall@k
evaluation uses.

## Retriever

```bash
python scripts/eval_retrieval.py
```

`verilit/retrieval.py` has two rankers with the same interface: `SemanticRanker` (embedding
search with `BAAI/bge-small-en-v1.5`, cosine similarity) and `BM25Ranker` (keyword search,
the baseline). The Retriever agent uses them to pick the most relevant abstracts for each
sub-question from the candidates Semantic Scholar and arXiv return.

The evaluation searches all 5,183 SciFact abstracts with each dev claim that has evidence
(188 claims, 209 gold abstracts). Recall@k is the share of gold evidence abstracts in the top
k; Hit@k is the share of claims with at least one in the top k.

| SciFact dev | R@1 | R@3 | R@5 | R@10 | R@20 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keyword (BM25) | 0.656 | 0.785 | 0.828 | 0.909 | 0.923 | 0.883 | 0.952 |
| Semantic (bge-small-en-v1.5) | **0.699** | **0.833** | **0.866** | **0.943** | **0.967** | **0.910** | **0.968** |

On a 4-core CPU, BM25 takes about 3 s and semantic search about 6 minutes (almost all of it
embedding the 5,183 abstracts once). In the pipeline it only embeds a few dozen candidates
per sub-question, which takes a second or two.

## Running VeriLit

The Planner, Claim Generator and Synthesizer use a free-tier LLM through its
OpenAI-compatible API. Set your key first (PowerShell; `setx` keeps it for new terminals):

```powershell
$env:GEMINI_API_KEY = "your-key"      # this terminal only
setx GEMINI_API_KEY "your-key"        # permanently (open a new terminal afterwards)
```

Then ask a question:

```bash
python scripts/run_pipeline.py "How does retrieval-augmented generation reduce hallucination in large language models?"
python scripts/run_pipeline.py "..." --no-verify        # the no-verification baseline
```

Each run writes `runs/<time>-<question>-<mode>.md` (the report: review, references, a table
of every claim with its verdict, confidence and retries, unverified claims, evidence gaps) and
a `.json` with everything, including every attempt of every claim. It needs the fine-tuned
Verifier in `models/verifier/` (with `calibration.json`), except with `--no-verify`.

| Setting | What it does |
|---|---|
| `VERILIT_LLM=groq` + `GROQ_API_KEY` | use Groq instead of Gemini (e.g. when Gemini's daily quota runs out) |
| `VERILIT_LLM_MODEL` | override the model (default `gemini-3.5-flash-lite`, Groq `llama-3.1-8b-instant`) |
| `S2_API_KEY` | your Semantic Scholar key; without it Semantic Scholar often refuses (429) and VeriLit uses arXiv only for 10 minutes at a time |
| `--retriever bm25` | keyword ranking instead of semantic |
| `--threshold 0.7` | accept a claim when its calibrated P(SUPPORTED) is at least this |
| `--max-retries 2`, `--papers 5`, `--max-claims 3` | feedback-loop attempts per claim, papers per sub-question, claims per sub-question |

**Free-tier budget.** A question takes about 7 LLM requests (plan, one claim set per
sub-question, the review, plus a revision per retried claim), roughly 30–60 seconds.
Every LLM reply and search result is cached in `cache/`, so re-running a question, or running
the baseline after VeriLit, only sends the requests that changed (the baseline needs one, for
its review). Delete `cache/` to start fresh. The Groq preset has not been tested yet.

**Feedback loop.** A claim the Verifier does not accept (calibrated P(SUPPORTED) below the
threshold) is retried up to `--max-retries` times: if the cited abstract gives insufficient
evidence, VeriLit first searches for the claim itself and checks it against the best new
papers (switching papers only if one supports it better); a contradicted claim, or one still
unsupported, is rewritten by the LLM to say only what the abstract states. Claims still not
accepted are listed as unverified and left out of the review. Reference numbers are assigned
by the code, not the LLM, and the report warns if the review cites a number that doesn't exist.

## Custom test set

25 research questions across AI, ML, NLP and cybersecurity (`eval/questions.csv`), run through
VeriLit with 2 claims per sub-question, then labeled by hand.

```bash
python scripts/run_eval_set.py    # runs every question (resumes if stopped), then builds the labeling sheets
python scripts/score_eval.py      # once the sheets are labeled
```

1. **Run.** `run_eval_set.py` saves `eval/runs/Q01.json` … `Q25.json` (+ `.md` reports). If the
   free LLM quota runs out, run it again later; finished questions are skipped.
2. **Label.** It writes `eval/labels/labels_maryam.csv` and `labels_nimra.csv`: one row per
   claim with the abstract of the paper it cites, shuffled, *without* the Verifier's verdict.
   Every original claim is labeled (these are also the baseline's claims), plus the final form
   of claims the feedback loop changed and verified. 30 rows are in both sheets. Put **S**, **C**
   or **I** in the `label` column, following `eval/LABELING.md`. Label alone, without looking at
   the reports. The script never overwrites existing sheets.
3. **Score.** `score_eval.py` reports annotator agreement (Cohen's kappa on the shared rows;
   it writes `eval/labels/disagreements.csv` for you to settle by discussion), and for VeriLit
   vs. the no-verification baseline: unsupported-claim rate, citation correctness, evidence
   coverage; retry success rate and retries per review; and the Verifier's agreement with the
   hand labels, with full-abstract and best-passage scoring. Results go to `results/custom_eval.json`.

The baseline is the same LLM, papers and claims without the Verifier, so the baseline's claims
are exactly VeriLit's original claims and need no separate run (`run_pipeline.py --no-verify`
produces the baseline's written review if you want it for the report).

## Repository layout

```
verilit/              Python package
  labels.py           the three verdict labels and their mappings (shared by everything)
  nli.py              NLIVerifier: loads any NLI model, returns probabilities in label order
  metrics.py          accuracy, macro P/R/F1, confusion matrix, calibration error (ECE)
  retrieval.py        SemanticRanker and BM25Ranker
  scifact.py          reading the SciFact files
  llm.py              LLM client (Gemini / Groq presets, rate limiting, retries, caching)
  sources.py          Semantic Scholar and arXiv search
  agents.py           Planner, Retriever, Claim Generator, Verifier, Synthesizer
  pipeline.py         runs the agents with the feedback loop; to_markdown() renders the report
  cache.py            on-disk cache for API responses
scripts/
  download_scifact.py
  prepare_scifact.py
  eval_verifier.py
  finetune_verifier.py
  calibrate_verifier.py
  eval_retrieval.py
  run_pipeline.py     run VeriLit on one question
  run_eval_set.py     run the custom test set and build the labeling sheets
  score_eval.py       score the custom test set from the hand labels
eval/                 questions.csv, LABELING.md, runs/ (one record per question), labels/
results/              metrics, training log, calibration, retrieval (predictions.jsonl is git-ignored)
data/, models/        git-ignored; re-create with the scripts above
runs/, cache/         git-ignored; pipeline outputs and cached API responses
```

Planned additions:

| Where | What |
|---|---|
| `app.py` | the demo UI |

## Results

| Verifier | accuracy | macro F1 | ECE |
|---|---:|---:|---:|
| Off-the-shelf NLI (baseline) | 0.482 | 0.431 | 0.458 |
| Fine-tuned (lr 2e-5, best epoch 5 of 5) | 0.756 | 0.738 | 0.161 |
| Fine-tuned + calibrated (T = 1.908) | 0.756 | 0.738 | 0.062 |
| Fine-tuned + calibrated, best passage (T = 2.223) | 0.741 | 0.728 | 0.086 |

SciFact dev split, 340 pairs. The fine-tuned rows come from a CPU run with the default
settings (seed 42); a GPU run gives slightly different numbers. During training, val ECE rose
from 0.065 after epoch 1 to 0.148 after epoch 5 (the model gets more over-confident as it
trains). Calibration fixes most of that without changing any prediction. CONTRADICTED is the
weakest label (F1 0.623 vs. 0.79–0.80 for the other two); `--balanced` is worth trying for it.

## Data sources

- SciFact: Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020 — <https://github.com/allenai/scifact>
- Semantic Scholar Academic Graph API — <https://www.semanticscholar.org/product/api>
- arXiv API — <https://info.arxiv.org/help/api/>
