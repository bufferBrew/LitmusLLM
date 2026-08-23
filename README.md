# LitmusLLM

A local-first LLM evaluation dashboard. Pick a model, pick what to measure,
press start — and get scored, explained results without anything leaving your
machine.

Built on [DeepEval](https://github.com/confident-ai/deepeval) for the metrics,
[Ollama](https://ollama.com) for local models, and
[LiteLLM](https://github.com/BerriAI/litellm) for optional cloud comparisons.

- **Run evals** on any pulled Ollama model across 11 DeepEval metrics
- **Watch progress live** and stop mid-run without losing what's been scored
- **Compare models** side by side — local against local — with ranks and a bar chart
- **Published benchmark reference** for frontier and open-weight models, so you
  can see where the ceiling is without paying for a single cloud API call
- **Model cards** for everything Ollama has pulled, with pull/delete buttons
- **Bring your own data** via CSV, or use the built-in 15-case starter set
- **Export** any run to JSON, CSV or a Markdown report

Everything is stored in one SQLite file. No accounts, no services, no cloud
calls unless you explicitly select a cloud model.

---

## Quick start (local)

Requires **Python 3.10+** and Ollama.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000>.

If you have no models yet:

```bash
ollama pull llama3.1:8b
```

LitmusLLM starts `ollama serve` for you if it's installed and not running. If
that fails it tells you exactly what to do rather than hanging.

## Quick start (Docker)

```bash
docker compose up -d
docker compose exec ollama ollama pull llama3.1:8b
```

Open <http://localhost:8000>.

Two things worth knowing before you reach for Docker:

- **On Apple Silicon, containerised Ollama runs on CPU.** Docker Desktop can't
  pass through the Metal GPU, so a 8B model that answers in two seconds
  natively can take a minute or more in the container. On a Mac, prefer the
  local install above and let LitmusLLM talk to your host Ollama. The NVIDIA
  GPU stanza in `docker-compose.yml` (commented out) does work on Linux.
- **Model storage.** By default models live in the `ollama-models` volume,
  separate from any models you've pulled on the host. To share the host's
  model store instead, replace the volume line in `docker-compose.yml` with
  `- ~/.ollama:/root/.ollama` — and don't run host Ollama and the container at
  the same time, since they'd both write the same store.

Eval history persists in `./data`, which is bind-mounted into the container.

---

## How an evaluation actually works

Worth understanding, because it explains the settings:

1. **Generate.** Your prompt goes to the *model under test*, which produces an
   answer. Local models are reached over Ollama's OpenAI-compatible endpoint
   (`/v1/chat/completions`), cloud models through LiteLLM — the same request
   shape either way.
2. **Judge.** Each metric then asks a *judge model* to score that answer and
   explain why.

The judge is a separate choice from the model under test, and it matters more
than people expect:

- **Judge quality caps result quality.** A 1.5B judge produces noisy scores no
  matter how good the model being tested is.
- **Keep the judge fixed across runs you intend to compare.** Changing it
  changes the numbers. A comparison run enforces this automatically: every
  model in it is graded by the same judge.
- **7B is the practical floor; 14B is comfortable.** DeepEval requires the
  judge to emit strict JSON. Small models frequently wrap it in prose or
  markdown — LitmusLLM strips fences, extracts the JSON object and retries
  three times, but a model that simply can't do it will fail with a message
  telling you to pick a bigger judge.

### Prompting and context

When you select a metric that grades against retrieved context (Faithfulness,
Contextual Recall / Precision / Relevancy), LitmusLLM includes the dataset's
`context` in the prompt — otherwise you'd be scoring the model on material it
never saw.

**Hallucination is deliberately excluded from that rule.** Its whole purpose is
to check whether the model invents facts when it *isn't* handed the answer, so
its context is used only as the judge's ground truth.

---

## The metrics

| Metric | What it measures | Needs |
| --- | --- | --- |
| **Answer Relevancy** | Does the answer address the question actually asked? | — |
| **Faithfulness** | Are all claims supported by the retrieved context? | context |
| **Hallucination** ↓ | Did the model fabricate unsupported information? | context |
| **Toxicity** ↓ | Is the response harmful, offensive or abusive? | — |
| **Bias** ↓ | Demographic, political or ideological slant | — |
| **Contextual Recall** | Did retrieval surface what the answer needed? | expected output + context |
| **Contextual Precision** | Did retrieval rank relevant chunks above noise? | expected output + context |
| **Contextual Relevancy** | How much retrieved context was actually useful? | context |
| **G-Eval (Correctness)** | Custom chain-of-thought rubric, graded by the judge | expected output |
| **Summarization Score** | Alignment + coverage for summarisation tasks | — |
| **Tool Call Accuracy** | Did the agent call the tools it should have? | tool traces |

**↓ marks metrics where lower is better.** LitmusLLM knows the direction of each
metric, so comparison rankings put 0.0 Toxicity in first place rather than last.

Metrics that need a field they don't have are **skipped per test case with a
recorded reason**, not silently zeroed and not fatal — so mixing
Answer Relevancy with Faithfulness on a partly-annotated dataset gives you real
numbers for the cases that qualify.

### Adding a custom metric

Add one entry to `METRICS` in [`metrics_catalog.py`](metrics_catalog.py), then
one branch in `build_metric()` in [`eval_runner.py`](eval_runner.py). The UI,
the API, the dashboard and the comparison ranking all read from the catalogue,
so nothing else needs changing.

For a rubric of your own, copy the `g_eval_correctness` entry and rewrite its
`criteria` string — G-Eval will grade against whatever you describe:

```python
MetricSpec(
    key="g_eval_tone",
    label="G-Eval (Tone)",
    description="Scores whether the response keeps a professional, non-condescending tone.",
    threshold=0.7,
    higher_is_better=True,
    category="Custom rubric",
    advanced=True,
    extra={
        "criteria": "Judge whether the actual output is professional and never condescending...",
        "params": ("input", "actual_output"),
    },
),
```

---

## Datasets

The built-in **LitmusLLM starter set** has 15 cases spanning plain Q&A,
context-grounded Q&A and summarisation.

Upload your own CSV from the home page:

| Column | Required | Notes |
| --- | --- | --- |
| `input` | **yes** | the prompt / question |
| `expected_output` | no | reference answer |
| `context` | no | separate multiple chunks with `\|` or newlines |
| `tools_called` | no | comma/pipe-separated tool names |
| `expected_tools` | no | comma/pipe-separated tool names |

Unknown columns are ignored, so exports from other tools usually load as-is.
Rows are stored in SQLite (not just on disk), so a dataset stays readable even
if you move the original file. Download the starter set from the home page for
a ready-made template.

---

## Cloud models

Set the keys you want in `.env` and restart. Models without a key are greyed
out in the UI rather than failing at run time.

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...
TOGETHER_API_KEY=...        # for the hosted Llama 405B and Qwen2.5 72B entries
```

The flagship list lives in `FLAGSHIP_MODELS` in
[`model_registry.py`](model_registry.py) — it's plain data, so add or remove
entries freely. Anything LiteLLM can route works.

> **A note on the preset list.** It contains the ten models from the original
> spec plus the two current Anthropic models. `claude-3-5-sonnet-20241022` and
> `claude-3-opus-20240229` are 2024 ids that have since been retired from the
> API; they're kept and tagged `legacy` so the list matches the spec, but
> **Claude Opus 5** and **Claude Sonnet 5** are the entries you actually want.

Cloud models cost money per call. A comparison run prompts every selected model
once per test case, then the judge grades each answer — so a 15-case dataset
with 3 metrics is a few hundred calls. Start small.

---

## Published benchmarks (the Reference page)

You usually don't want to *run* evals against cloud models — it costs money per
call and the API keys are a hassle. The **Reference** page instead shows figures
somebody else already measured, so you can see where the frontier sits for free.

### The thing to understand before you use it

**Reference numbers and your eval scores are not comparable, and LitmusLLM never
plots them together.** This is the whole reason the feature is a separate page
and a separate panel rather than extra bars on the comparison chart:

| | Your eval runs | The Reference page |
| --- | --- | --- |
| What's measured | *your* prompts, *your* dataset | ten fixed public benchmarks |
| How | an LLM judge you chose | a third party's harness |
| Scale | 0–1 per metric | 0–100 aggregate index |
| Who ran it | you, just now | Artificial Analysis, on a date |

A model scoring `0.91` Answer Relevancy on your dataset and a model scoring `63`
on the index have not been compared with each other in any meaningful sense.
Adjacent bars would imply otherwise, so the UI refuses to draw them.

### Where the numbers come from

One third-party aggregator running **one methodology across every model**, rather
than vendor self-reported figures. Vendors publish different benchmarks with
different shot counts and prompting, so their numbers aren't comparable to each
other either — Anthropic's model docs, for instance, publish no MMLU/GPQA table
at all.

Every row stores its source URL and retrieval date, both shown in the UI.
Leaderboards move; re-check before quoting.

### The coverage gap

The index tracks *current* models. The 2024-era small models most people run
locally — `llama3.1:8b`, `qwen2.5:14b`, `gemma2:2b`, `phi3.5`, `llama3.2:3b` —
have aged off it and **have no row**. That blank is deliberate: an invented
number would be worse than none. The modern small open-weight entries
(Qwen3.5 4B, Ministral 3 8B, Phi-4 Mini) are the closest available anchors for
that size class, and rows in a family you actually run are highlighted.

### Adding your own figures

The `/reference` page has an add form and a CSV export. Seed data lives in
`SEED_ROWS` in [`benchmarks.py`](benchmarks.py) — plain data, edit freely.
Please fill in the source URL when you add a row.

> **Still want to run real evals against cloud models?** That path still works —
> set an API key in `.env` and the model becomes selectable on `/` and
> `/compare`. Those results *are* measured by LitmusLLM and *are* comparable to
> your local runs, because it's the same dataset, judge and metric. It just
> costs money.

---

## Stopping a run

**Stop** is cooperative, not a kill. The cancel flag is checked between test
cases and between metrics, so a run stops after the currently in-flight judge
call returns — usually a few seconds, longer with a big judge on CPU.

Everything already scored is already in SQLite. The run is marked `stopped`
(not `failed`) and keeps its partial averages.

---

## API

Every screen is backed by a JSON endpoint, so LitmusLLM scripts as well as it
clicks.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/models` | Ollama models, live from `/api/tags` |
| `GET` | `/api/flagship-models` | Cloud model list + key availability |
| `GET` | `/api/metrics` | Metric catalogue with descriptions |
| `GET` | `/api/health` | App + Ollama status |
| `POST` | `/api/models/pull` · `/api/models/delete` | Manage local models |
| `GET` | `/api/datasets` · `/api/datasets/{id}` | List / inspect datasets |
| `POST` | `/api/datasets/upload` | Upload a CSV |
| `POST` | `/api/evals/start` | Start a run |
| `POST` | `/api/evals/{id}/stop` | Stop a run, keeping partial results |
| `GET` | `/api/evals` · `/api/evals/{id}` | List / inspect runs |
| `GET` | `/api/evals/{id}/export.{json,csv,md}` | Export a run |
| `POST` | `/api/compare` | Start a multi-model comparison |
| `GET` | `/api/compare/{id}` | Comparison results |
| `GET` | `/api/compare/{id}/export.{csv,md}` | Export a comparison |
| `GET` | `/api/reference` | Published benchmark figures + provenance |
| `POST` | `/api/reference` | Add a figure |
| `GET` | `/api/reference/export.csv` | Export the reference table |

```bash
curl -X POST http://localhost:8000/api/evals/start \
  -F "model=local:llama3.1:8b" \
  -F "judge=local:qwen2.5:14b" \
  -F "dataset_id=1" \
  -F "metrics=answer_relevancy" \
  -F "metrics=faithfulness"
```

Model ids are `local:<ollama-tag>` or `cloud:<litellm-model>`.

---

## Troubleshooting

**"Ollama is not reachable"**
LitmusLLM tries to start it automatically when it's installed locally. If that
fails, run `ollama serve` in a terminal. Under Docker, check
`docker compose ps` — the app waits for the `ollama` service to pass its health
check.

**"Judge model could not produce valid JSON after several attempts"**
The judge is too small. Switch to a 7B+ model (14B is comfortable) in the
**Judge model** selector, or use a cloud judge. This is the single most common
cause of a failed run.

**"OpenAI API key is not configured"**
Shouldn't happen — LitmusLLM passes an explicit judge to every metric precisely
to avoid DeepEval's silent GPT-4o fallback. If you see it, a custom metric was
added without passing `model=judge` in `build_metric()`.

**"Ollama returned 404"**
The model isn't pulled. Use the Pull button on `/models`, or
`ollama pull <model>`.

**A run is stuck at 0/N**
The first generation is warming the model into memory; a large model on CPU can
take a minute. The progress note tells you which phase it's in.

**Everything is very slow**
Local evals are LLM calls all the way down: one generation plus several judge
calls per test case per metric. Fewer metrics, a smaller dataset, or a smaller
judge all help. Concurrency is capped at 2 for local models on purpose —
raise `LITMUSLLM_LOCAL_CONCURRENCY` if you have the memory.

**A run says `failed` after a restart**
In-flight runs can't survive a process restart, so LitmusLLM marks them failed
on boot rather than leaving them permanently "running".

---

## Project layout

```
LitmusLLM/
├── main.py                 # FastAPI app: pages, JSON API, HTMX fragments
├── eval_runner.py          # generate → judge loop, cancellation, comparisons
├── llm_clients.py          # Ollama + LiteLLM transport, DeepEval judge adapter
├── model_registry.py       # Ollama discovery, flagship list, model ids
├── metrics_catalog.py      # metric definitions, thresholds, score direction
├── datasets.py             # built-in dataset, CSV parsing
├── benchmarks.py           # published third-party reference figures + provenance
├── database.py             # SQLite schema and queries
├── config.py               # env-driven configuration
├── templates/              # Jinja2 + HTMX views
├── static/                 # CSS, chart + page JS
└── data/                   # SQLite db + uploaded CSVs (gitignored)
```

## Configuration

All of `.env` is optional — see [`.env.example`](.env.example) for the full
list with defaults. The ones you're most likely to touch:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama native API |
| `LITMUSLLM_DEFAULT_JUDGE` | `llama3.1:latest` | Judge preselected in the UI |
| `LITMUSLLM_LOCAL_CONCURRENCY` | `2` | Concurrent requests to a local model |
| `LITMUSLLM_REQUEST_TIMEOUT` | `300` | Per-request timeout, seconds |
| `LITMUSLLM_DB_PATH` | `data/litmusllm.db` | SQLite location |
| `DEEPEVAL_TELEMETRY_OPT_OUT` | `YES` | DeepEval anonymous stats, off by default |
