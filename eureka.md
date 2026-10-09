# AI Creative Benchmark Finder — Full Build Plan (for the AI Agent)

> **Mission:** Build a Python system that (1) **discovers** the latest AI-creativity benchmarks from papers, repos and datasets, and (2) **runs** them against any AI model through its real API key, producing a traceable creativity profile and leaderboard.
>
> **Hard rules (non-negotiable):**
> 1. **Real API calls only.** No mocked responses, no fake scores, no placeholder numbers, no "simulated" runs. If a call fails, log the failure and retry or skip — never fill in a value.
> 2. **Every score must be traceable** to a saved raw API request/response (timestamp, model ID, request ID, prompt hash).
> 3. **Real-time sources.** Benchmark discovery queries live sources (arXiv, Semantic Scholar, Hugging Face, GitHub) at run time, not a hardcoded list. The seed list below is only a starting point.
> 4. **Python 3.11+**, async where it matters, fully typed, tested.
> 5. **Never hardcode API keys.** Read from environment variables or a `.env` file that is git-ignored.

---

## 0. Assumptions (change if wrong)

- "Benchmark finder" = both **finding benchmarks** (literature/tool discovery) **and measuring a model** on them.
- The user supplies API keys for the model(s) under test (OpenAI, Anthropic, Google, Mistral, OpenRouter, any OpenAI-compatible endpoint, etc.).
- Creativity is measured on **text first**. Image/multimodal creativity is Phase 6 (optional).
- Model names change often → **never hardcode model IDs**; they come from `config.yaml` or the CLI.

---

## 1. What "creativity" means here (measurement framework)

Creativity is not one number. Measure these dimensions separately, then combine:

| Dimension | What it captures | Example measures |
|---|---|---|
| **Divergent thinking** | Many, varied, original ideas | Divergent Association Task (DAT), Alternative Uses Task (AUT) |
| **Diversity / anti-mode-collapse** | Does the model give different answers across samples? | NoveltyBench-style distinct-output counts, embedding dispersion, Self-BLEU |
| **Creative writing quality** | Story/poem quality judged against a rubric | EQ-Bench Creative Writing v3 style rubric + pairwise Elo |
| **Associative creativity** | Connecting distant concepts, verifiable | CREATE (2026) |
| **Creative problem solving** | Repurposing objects, working around missing tools | CreativityBench (2026, affordance-based) |
| **Humanlike calibration** | Where the model sits vs. human norms | Human percentile comparison on DAT/AUT |

Key research findings the design must respect (verify each by reading the source before relying on it):

- **AGC-Bench (arXiv 2607.01152, July 2026):** a unified meta-benchmark built from a review of ~3,100 papers and ~497 benchmarks; uses *Judge Response Theory* to correct LLM-judge leniency/severity; reports a single general creativity factor across 83 LLMs; finds that **prompting a model to "be creative" helps far more than turning on reasoning**; the top human still beats the top LLM. → **Design implication:** always test at least two prompt conditions (neutral vs. "be creative") and treat judge calibration as a first-class feature.
- **"Has the creativity of LLMs peaked?" (Journal of Creativity, 2025):** 14 LLMs on DAT + AUT; large *intra-model* variability (same prompt, wildly different quality) and very few responses reach the top 10% of human scores. → **Design implication:** sample many times per prompt and report distributions + confidence intervals, never a single generation.
- **Jerbi et al., Scientific Reports (Jan 2026):** 100k+ humans vs. LLMs; some models beat the *average* human on DAT/creative writing, top 10% of humans still win; temperature and prompting matter. → **Design implication:** run a temperature sweep and report human-percentile placement.
- **NoveltyBench (arXiv 2504.05228):** 1,100 prompts; evaluates sets of outputs for both diversity and quality; finds frontier models lack diversity. → use its metric ideas (utility over k generations).
- **EQ-Bench Creative Writing v3 (GitHub: EQ-bench/creative-writing-bench):** hybrid rubric + Elo, bias mitigation for length/position/verbosity; repo notes API cost is roughly $10 per model with its recommended judge. → use as a reference implementation and cost baseline.
- **CREATE (arXiv 2603.09970)**, **CreativityBench (arXiv 2605.02910, preprint)**, **CreBench (AAAI 2026, multimodal)**, **WritingBench (arXiv 2503.05244)**, **WebNovelBench (EACL 2026 Findings)** → candidate suites for the registry.

> The agent must re-verify these (title, date, license, code availability) from the live source before ingesting. Treat this list as seeds, not truth.

---

## 2. System architecture

```
creative-benchmark-finder/
├── pyproject.toml
├── config.yaml                # models under test, judges, budgets, sources
├── .env.example               # OPENAI_API_KEY=..., ANTHROPIC_API_KEY=..., GOOGLE_API_KEY=...
├── src/cbf/
│   ├── providers/             # real API adapters
│   │   ├── base.py            # LLMProvider interface
│   │   ├── openai_compat.py   # OpenAI + any OpenAI-compatible endpoint
│   │   ├── anthropic.py
│   │   ├── google.py
│   │   └── registry.py
│   ├── discovery/             # the "Finder"
│   │   ├── sources/           # arxiv.py, semantic_scholar.py, huggingface.py, github.py, rss.py
│   │   ├── extractor.py       # LLM-assisted metadata extraction from paper/repo
│   │   ├── ranker.py          # relevance + credibility + recency scoring
│   │   └── registry_db.py     # SQLite: discovered benchmarks + status
│   ├── benchmarks/            # the "Runner"
│   │   ├── base.py            # Benchmark interface
│   │   ├── dat.py             # Divergent Association Task
│   │   ├── aut.py             # Alternative Uses Task
│   │   ├── diversity.py       # NoveltyBench-style
│   │   ├── writing_elo.py     # rubric + pairwise judging
│   │   ├── associative.py     # CREATE-style
│   │   ├── affordance.py      # CreativityBench-style
│   │   └── adapters/          # auto-generated wrappers for newly found benchmarks
│   ├── judging/
│   │   ├── panel.py           # multi-judge panel
│   │   ├── bias.py            # position swap, length control
│   │   └── calibration.py     # judge leniency/severity correction
│   ├── metrics/               # embeddings, distinct-n, self-BLEU, bootstrap CIs
│   ├── runner/                # async orchestration, rate limits, retries, budget guard
│   ├── storage/               # raw call log (JSONL), results DB (SQLite/Parquet)
│   ├── report/                # leaderboard, HTML/Markdown report, charts
│   └── cli.py                 # `cbf discover`, `cbf run`, `cbf report`
├── data/
│   ├── prompts/               # versioned prompt sets
│   └── human_norms/           # published human baselines (with source + license noted)
├── runs/                      # one folder per run: manifest + raw logs + results
└── tests/
```

### 2.1 Provider layer (the "through their API keys" part)

```python
# src/cbf/providers/base.py
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class LLMRequest:
    model: str
    messages: list[dict]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 1024
    seed: int | None = None
    extra: dict = field(default_factory=dict)   # e.g. reasoning settings

@dataclass
class LLMResponse:
    text: str
    model: str
    request_id: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    raw: dict            # full provider payload, stored verbatim
    timestamp_utc: str

class LLMProvider(Protocol):
    name: str
    async def complete(self, req: LLMRequest) -> LLMResponse: ...
    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...
    def cost_usd(self, resp: LLMResponse) -> float: ...
```

Requirements:
- One adapter per vendor using the **official SDK or plain `httpx`** against the live endpoint. An OpenAI-compatible adapter covers many other vendors via `base_url`.
- A **key check** on startup: one cheap real call per provider; fail loudly if it errors.
- **Automatic model listing** from each provider's models endpoint where available, so the user picks a model that actually exists.
- **Capability flags** per model (supports temperature? reasoning toggle? system prompt?) discovered from docs/errors, not assumed.
- **Rate limiting + exponential backoff** (respect `Retry-After`), **concurrency cap**, and a **hard budget guard** (`max_usd` per run; abort cleanly when reached).
- **Raw call log:** append every request/response to `runs/<run_id>/calls.jsonl` (mask the key). This is the proof that nothing was simulated.

### 2.2 Embeddings
- DAT needs word semantic distance. Original DAT uses GloVe vectors; the agent should download the real GloVe file once (cached, license noted) **or** use a provider embedding API. Pick one, document it, and keep it fixed across all models so scores are comparable.
- Diversity metrics need sentence embeddings: use a local model (e.g., `sentence-transformers`) or an embedding API; record the exact model/version in the run manifest.

---

## 3. The Finder — real-time benchmark discovery

**Goal:** keep a living registry of creativity benchmarks, updated from live sources.

### 3.1 Sources (all queried live)
| Source | How | Notes |
|---|---|---|
| arXiv | arXiv API (`export.arxiv.org/api/query`), categories `cs.CL`, `cs.AI`, `cs.HC` | queries like "creativity benchmark LLM", "divergent thinking language model", "creative writing evaluation", "novelty diversity generation" |
| Semantic Scholar | Graph API | citation counts, "papers citing AGC-Bench / NoveltyBench / EQ-Bench" |
| Hugging Face | Hub API: datasets + papers | tags/keywords: creativity, creative-writing, divergent |
| GitHub | REST search | repos with "creativity benchmark", "creative writing bench"; read README, license, stars, last commit |
| Leaderboards | EQ-Bench site, AGC-Bench public leaderboard | scrape only if terms allow; otherwise link out |
| RSS / blogs | lab and researcher blogs | used for leads, then verified against the paper |

### 3.2 Pipeline
1. **Query** all sources with a rotating keyword set + "cited-by" expansion of known anchors (AGC-Bench, NoveltyBench, EQ-Bench, CREATE).
2. **Dedupe** by arXiv ID / DOI / repo URL.
3. **Extract** (LLM-assisted, then rule-checked): benchmark name, what it measures, task format, metric, judge used, dataset size, license, code URL, data URL, date, peer-review status. Extraction must quote the source location (section/line) so a human can check it.
4. **Score** each candidate: `relevance` (is it really about creativity?), `runnability` (code + data public? license OK?), `credibility` (venue, citations, reproducibility, peer review vs. preprint), `recency`.
5. **Classify status:** `seed` / `candidate` / `adapter_drafted` / `approved` / `rejected` (with reason).
6. **Adapter generation:** for approved benchmarks the agent writes a wrapper implementing `Benchmark` (Section 4). **A human approves** each new adapter before it is used in a published leaderboard (prevents silently wrong implementations).
7. **Schedule:** `cbf discover --since 7d` runs weekly (cron/GitHub Actions) and writes a "what's new" digest.

### 3.3 Finder guardrails
- Never trust a paper's number without the source. Store the URL + retrieval date for every fact.
- Flag preprints as unreviewed.
- Respect `robots.txt`, API terms and rate limits; use official APIs over scraping.
- Respect dataset licenses; record them in the registry.

---

## 4. The Runner — benchmark suites

All suites implement one interface:

```python
class Benchmark(Protocol):
    id: str
    version: str
    def prompts(self) -> list[Item]: ...
    async def run(self, provider, model, cfg) -> RawResults: ...   # real API calls
    def score(self, raw: RawResults) -> Scores: ...                # pure function, deterministic
```

### 4.1 Tier A — cheap, objective, run first
1. **DAT (Divergent Association Task):** ask the model for 10 words as unrelated as possible; validate words (single, common, nouns); score = average pairwise semantic distance × 100. Run ≥50 samples per condition. Compare to published human norms (cite source + date in `data/human_norms/`).
2. **AUT (Alternative Uses Task):** for ~10 everyday objects (brick, paperclip, …) ask for unusual uses. Score **fluency** (count), **flexibility** (clusters), **originality** (embedding distance from the common-use centroid + LLM-judge rating), **elaboration**.
3. **Diversity suite (NoveltyBench-style):** for each prompt sample *k* generations (e.g., k=10–20); compute distinct clusters via embedding similarity threshold, mean pairwise distance, distinct-n, Self-BLEU; combine with a quality score so diversity of *bad* answers doesn't count.

### 4.2 Tier B — judge-based
4. **Creative writing (EQ-Bench v3 style):** fixed prompt set; rubric scoring + **pairwise comparisons** → Elo/Bradley-Terry. Reference the open-source repo; reproduce its protocol, then extend.
5. **Judge panel (essential):**
   - ≥ 2–3 judges from **different vendors**; the model under test is **never** its own sole judge (self-preference bias).
   - **Position swap** every pairwise comparison; **length-controlled** scoring; blind to model identity.
   - Report **inter-judge agreement** (Krippendorff's α or Spearman). If low, flag the score as unreliable.
   - Implement **judge calibration** (leniency/severity per judge, inspired by AGC-Bench's Judge Response Theory) once ≥ 3 judges and enough items exist.
   - Keep a small **human-rated gold set** (even 30–50 items rated by the user/friends) to sanity-check judges.

### 4.3 Tier C — structured creativity
6. **Associative creativity (CREATE-style):** connect real-world entities with verifiable paths; objective grading where possible.
7. **Affordance repurposing (CreativityBench-style):** "tool X is missing — solve the task with these objects"; judge feasibility + novelty. (Source is a preprint → mark as experimental.)
8. **TTCT-style verbal tasks:** consequences, product improvement, unusual questions — scored for fluency/flexibility/originality/elaboration.

### 4.4 Experimental conditions (run every Tier A/B suite under each)
| Factor | Levels |
|---|---|
| Prompt framing | neutral · "be creative / be original" · diversity-forcing (in-context regeneration) |
| Temperature | e.g., 0.2 · 0.7 · 1.0 · model max (where supported) |
| Reasoning mode | off · on (where the model supports it) |
| Samples | ≥ 30–50 per cell for stats |

This lets the tool report *why* a model scores as it does (e.g., "creativity comes from prompting, not from reasoning").

### 4.5 Scoring and the final "Creativity Profile"
- Normalize each metric to 0–100 using documented references (human norms or min/max across the tested pool). Record the normalization method.
- Report **per-dimension scores with 95% bootstrap CIs**, a **radar chart**, and an **optional composite** (default: equal weights; user-configurable). Always show the components next to the composite.
- Report **human percentile** where human data exists (DAT, AUT).
- Report **intra-model variance** (spread across samples) as its own metric — it was a key finding in the literature.
- Report **cost** and **latency** per model.

---

## 5. Data integrity and anti-"simulation" safeguards

1. `calls.jsonl` holds every real request/response; the results table stores the `call_id`s each score came from.
2. A `cbf verify <run_id>` command recomputes all scores from `calls.jsonl` only and fails if anything differs → proves reproducibility.
3. **Run manifest** per run: model IDs, provider versions, code git hash, prompt-set hashes, embedding model, judge IDs, temperature/seed settings, start/end time, total cost.
4. Unit tests may use recorded fixtures **only inside `tests/`**; the production code path must refuse to run with fixtures (check an env flag and fail).
5. Contamination note: public benchmark prompts may be in training data. Include a small **private prompt set** (agent generates fresh prompts per run, from a seeded template) and compare public vs. fresh performance.
6. Seeds/determinism: where APIs support `seed`, set it; where they don't, document that results vary and rely on multiple samples.

---

## 6. Phased roadmap (with acceptance criteria)

| Phase | Deliverable | Done when |
|---|---|---|
| **1. Foundation** (Days 1–3) | Repo, config, provider adapters (≥ 2 vendors), raw call log, budget guard, CLI skeleton | `cbf ping` makes one real call per configured provider and logs it |
| **2. Tier A suites** (Days 4–8) | DAT, AUT, diversity metrics, bootstrap CIs | `cbf run --suite tier_a --model X` yields scores + CIs from real calls; `cbf verify` passes |
| **3. Judging** (Days 9–14) | Judge panel, bias controls, pairwise Elo writing benchmark, agreement stats | Scores from ≥ 2 judge vendors with reported agreement; position-swap implemented |
| **4. Finder v1** (Days 15–21) | arXiv + Semantic Scholar + HF + GitHub crawlers, extractor, ranker, registry DB, weekly digest | `cbf discover` returns ranked, source-cited candidates, incl. the seed benchmarks rediscovered |
| **5. Adapters + Tier C** (Days 22–30) | Agent drafts adapters for approved benchmarks; CREATE-style + affordance suites; human-approval gate | At least 2 newly discovered benchmarks run end-to-end after human approval |
| **6. Reports & UI** (Days 31–38) | Leaderboard (Markdown/HTML), radar charts, run-to-run comparison, optional Streamlit/FastAPI dashboard | One command produces a shareable report with CIs, costs, caveats |
| **7. Hardening** (ongoing) | Judge calibration, contamination checks, optional multimodal (CreBench-style), CI, docs | Test coverage > 80% on scoring code; reproducibility check in CI |

---

## 7. Recommended Python stack

- **Core:** `python>=3.11`, `httpx`, `asyncio`, `pydantic`, `typer` (CLI), `tenacity` (retries), `python-dotenv`
- **Provider SDKs:** `openai`, `anthropic`, `google-genai` (or plain `httpx`)
- **Data/metrics:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `sentence-transformers`, `nltk`/`sacrebleu` (Self-BLEU), `krippendorff`
- **Storage:** `sqlite` (via `sqlmodel`) + JSONL; `pyarrow` for Parquet
- **Discovery:** `arxiv`, `semanticscholar`, `huggingface_hub`, `PyGithub`, `feedparser`, `beautifulsoup4`
- **Reporting:** `jinja2`, `matplotlib`/`plotly`, optional `streamlit`
- **Quality:** `pytest`, `ruff`, `mypy`, `pre-commit`
- Optional: an evaluation framework such as HELM or lm-evaluation-harness for standardized wrappers — AGC-Bench itself converts benchmark codebases into HELM-style benchmarks, so HELM compatibility is worth considering.

---

## 8. Risks and how the plan handles them

| Risk | Mitigation |
|---|---|
| LLM-as-judge bias (length, position, self-preference) | multi-vendor panel, position swap, length control, calibration, human gold set |
| Creativity is subjective | report dimensions separately, show CIs, never present one number as "the" truth |
| API cost blow-ups | per-run `max_usd`, dry-run cost *estimator* (token math only, no fake outputs), cheap Tier A first |
| Model outputs vary run to run | many samples, CIs, report variance |
| Benchmark contamination | fresh private prompts, public-vs-fresh comparison |
| Preprints that don't replicate | credibility score, "experimental" label, human approval gate |
| Key leakage | env vars only, `.gitignore`, log masking |
| ToS / rate limits | official APIs, backoff, concurrency caps, license tracking |
| Models change names/versions | no hardcoded IDs; record exact ID + date in every manifest |

---

## 9. Instructions the human should give the AI agent (copy-paste)

> You are building the "AI Creative Benchmark Finder" in Python 3.11+ following `AI_Creative_Benchmark_Finder_Plan.md`.
> Work phase by phase. At the end of each phase: run tests, run `cbf verify`, and summarize what works, what failed, and what you could not verify.
> **Never** fabricate API outputs, scores, citations, or benchmark metadata. If a source is unreachable or a fact can't be verified, say so and mark it `unverified`.
> Before adding any benchmark to the registry, read its primary source (paper + repo) and record URL, retrieval date, license and code/data availability.
> Ask for my approval before (a) spending more than the configured budget, (b) adding a new adapter to the published leaderboard, (c) scraping any site whose terms are unclear.
> Keep secrets in `.env`; never print keys.

---

## 10. First 10 concrete tasks for the agent

1. Scaffold the repo (`pyproject.toml`, `src/cbf`, `tests`, `.gitignore` incl. `.env`, `runs/`).
2. Implement `LLMProvider` + OpenAI-compatible adapter + Anthropic adapter; add `cbf ping`.
3. Implement raw-call JSONL logging and budget guard.
4. Implement DAT: prompt, word validation, embedding distance, scoring, bootstrap CI.
5. Implement AUT with fluency/flexibility/originality.
6. Implement the diversity metrics (embedding clustering, distinct-n, Self-BLEU).
7. Implement `cbf verify`.
8. Build the judge panel with position-swap and agreement stats.
9. Build the arXiv + Semantic Scholar crawler; rediscover AGC-Bench, NoveltyBench, CREATE, CreativityBench; store with source URLs.
10. Produce the first real leaderboard report for two models, with costs and caveats.

---

## 11. Seed reading list (agent must open and verify each)

- AGC-Bench: Measuring Artificial General Creativity — arXiv 2607.01152 (Jul 2026)
- CREATE: Testing LLMs for Associative Creativity — arXiv 2603.09970 (Mar 2026)
- CreativityBench — arXiv 2605.02910 (preprint, Apr 2026)
- CreBench (multimodal creativity evaluation) — AAAI 2026
- NoveltyBench: Evaluating Language Models for Humanlike Diversity — arXiv 2504.05228
- WritingBench — arXiv 2503.05244
- WebNovelBench — EACL 2026 Findings
- EQ-Bench / Creative Writing Benchmark v3 — github.com/EQ-bench/creative-writing-bench
- "Has the creativity of large language models peaked?" — Journal of Creativity (2025)
- Jerbi et al., LLM vs. 100k+ humans on divergent creativity — Scientific Reports (Jan 2026)
- Classic psychometrics: Torrance Tests of Creative Thinking; Guilford's Alternative Uses; Olson et al. (2021) Divergent Association Task

*Plan prepared 9 Oct 2026. Benchmark details above come from live searches on that date and must be re-verified by the agent before use.*
