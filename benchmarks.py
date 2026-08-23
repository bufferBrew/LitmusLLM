"""Published benchmark reference data.

## Why this exists, and what it is *not*

LitmusLLM measures models by running them. This module holds numbers that
somebody *else* measured and published, so you can see roughly where the
frontier sits without paying for a single cloud API call.

**These numbers are not comparable to LitmusLLM's own scores, and the UI never
plots them on the same axis.** A DeepEval metric like Answer Relevancy is an
LLM judge scoring *your* prompts on a 0-1 scale. The Artificial Analysis
Intelligence Index is an aggregate of ten fixed public benchmarks (GPQA
Diamond, Terminal-Bench, SciCode, GDPval-AA and others) expressed as a
percentage. Putting "0.85 Answer Relevancy" beside "63 Intelligence Index"
as adjacent bars would look authoritative and mean nothing. They live in a
separate panel, with a separate scale, and say so on screen.

## Source choice

A single third-party aggregator running one harness across every model beats
collecting vendor self-reported figures, which use different benchmarks, shot
counts and prompting between vendors and are therefore not comparable to each
other either. Anthropic's own model docs, for instance, publish no MMLU/GPQA
table at all.

## Coverage gap you should know about

The index tracks current models. The 2024-era small models many people run
locally -- llama3.1:8b, qwen2.5:14b, gemma2:2b, phi3.5, llama3.2:3b -- have
aged off it and have **no entry here**. That is deliberate: a blank is honest,
an invented number is not. The modern small open-weight rows (Qwen3.5 4B,
Ministral 3 8B, Phi-4 Mini) are the closest available anchors for that class
of model. Add your own rows on the /reference page if you have figures you
trust.

Every number carries its source and retrieval date. Re-check them before
quoting: leaderboards move.
"""
from __future__ import annotations

from typing import Any

import database

# --- Provenance shared by every seeded row --------------------------------
SEED_SOURCE = "Artificial Analysis Intelligence Index"
SEED_SOURCE_URL = "https://artificialanalysis.ai/leaderboards/models"
SEED_RETRIEVED = "2026-08-23"
SEED_INDEX_NOTE = (
    "Aggregate of ten public benchmarks (GPQA Diamond, Terminal-Bench, SciCode, "
    "GDPval-AA and others), expressed 0-100. Higher is better. Scores are shown "
    "as the leaderboard displayed them and are rounded."
)

# Reasoning-effort variants are kept separate where the leaderboard lists them
# separately -- collapsing them would hide that the same model scores
# differently depending on how hard it is allowed to think.
SEED_ROWS: tuple[dict[str, Any], ...] = (
    # --- Frontier / proprietary ---
    {"model_label": "Claude Opus 5", "variant": "max", "vendor": "Anthropic", "family": "Claude", "kind": "frontier", "score": 63.0},
    {"model_label": "Claude Opus 5", "variant": "xhigh", "vendor": "Anthropic", "family": "Claude", "kind": "frontier", "score": 63.0},
    {"model_label": "Claude Fable 5", "variant": "with fallback", "vendor": "Anthropic", "family": "Claude", "kind": "frontier", "score": 62.0},
    {"model_label": "Claude Opus 5", "variant": "high", "vendor": "Anthropic", "family": "Claude", "kind": "frontier", "score": 61.0},
    {"model_label": "GPT-5.6 Sol", "variant": "max", "vendor": "OpenAI", "family": "GPT", "kind": "frontier", "score": 61.0},
    {"model_label": "Grok 4.6", "variant": "high", "vendor": "xAI", "family": "Grok", "kind": "frontier", "score": 61.0},
    {"model_label": "Gemini 3.7 Flash", "variant": "", "vendor": "Google", "family": "Gemini", "kind": "frontier", "score": 56.0},
    {"model_label": "Gemini 3.6 Flash", "variant": "", "vendor": "Google", "family": "Gemini", "kind": "frontier", "score": 51.6},

    # --- Open weight ---
    {"model_label": "Qwen3.8 Max", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 58.0},
    {"model_label": "Qwen3.8 2.4T A95B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 58.0},
    {"model_label": "DeepSeek V4 Pro 0813", "variant": "max", "vendor": "DeepSeek", "family": "DeepSeek", "kind": "open_weight", "score": 53.2},
    {"model_label": "DeepSeek V4 Flash 0731", "variant": "max", "vendor": "DeepSeek", "family": "DeepSeek", "kind": "open_weight", "score": 51.8},
    {"model_label": "Qwen3.8 27B", "variant": "xhigh", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 52.0},
    {"model_label": "DeepSeek V4 Pro", "variant": "max", "vendor": "DeepSeek", "family": "DeepSeek", "kind": "open_weight", "score": 45.0},
    {"model_label": "DeepSeek V4 Pro", "variant": "high", "vendor": "DeepSeek", "family": "DeepSeek", "kind": "open_weight", "score": 44.0},
    {"model_label": "Qwen3.8 27B", "variant": "medium", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 44.0},
    {"model_label": "Qwen3.8 27B", "variant": "low", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 43.0},
    {"model_label": "Qwen3.7 Plus", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 39.0},
    {"model_label": "Qwen3.6 27B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 38.0},
    {"model_label": "Qwen3.5 397B A17B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 34.0},
    {"model_label": "Qwen3.5 122B A10B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 33.0},
    {"model_label": "Qwen3.6 35B A3B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 32.0},
    {"model_label": "DeepSeek V4 Pro", "variant": "non-reasoning", "vendor": "DeepSeek", "family": "DeepSeek", "kind": "open_weight", "score": 32.0},
    {"model_label": "Qwen3.6 27B", "variant": "non-reasoning", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 31.0},
    {"model_label": "Gemma 4 31B", "variant": "", "vendor": "Google", "family": "Gemma", "kind": "open_weight", "score": 30.0},
    {"model_label": "Mistral Medium 3.5", "variant": "", "vendor": "Mistral", "family": "Mistral", "kind": "open_weight", "score": 30.0},
    {"model_label": "Gemma 4 26B A4B", "variant": "", "vendor": "Google", "family": "Gemma", "kind": "open_weight", "score": 26.0},
    {"model_label": "Qwen3.5 35B A3B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 24.0},
    {"model_label": "gpt-oss-120b", "variant": "high", "vendor": "OpenAI", "family": "gpt-oss", "kind": "open_weight", "score": 24.0},
    {"model_label": "Qwen3.5 9B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 22.0},
    {"model_label": "Gemma 4 12B", "variant": "", "vendor": "Google", "family": "Gemma", "kind": "open_weight", "score": 22.0},
    {"model_label": "Qwen3 Coder Next", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 21.0},
    {"model_label": "Qwen3.5 4B", "variant": "", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 20.0},
    {"model_label": "Mistral Small 4", "variant": "", "vendor": "Mistral", "family": "Mistral", "kind": "open_weight", "score": 20.0},
    {"model_label": "Qwen3 Next 80B A3B", "variant": "reasoning", "vendor": "Alibaba", "family": "Qwen", "kind": "open_weight", "score": 17.0},
    {"model_label": "Mistral Large 3", "variant": "", "vendor": "Mistral", "family": "Mistral", "kind": "open_weight", "score": 16.0},
    {"model_label": "gpt-oss-20b", "variant": "high", "vendor": "OpenAI", "family": "gpt-oss", "kind": "open_weight", "score": 15.0},
    {"model_label": "gpt-oss-120b", "variant": "low", "vendor": "OpenAI", "family": "gpt-oss", "kind": "open_weight", "score": 15.0},
    {"model_label": "Llama 4 Maverick", "variant": "", "vendor": "Meta", "family": "Llama", "kind": "open_weight", "score": 14.0},
    {"model_label": "gpt-oss-20b", "variant": "low", "vendor": "OpenAI", "family": "gpt-oss", "kind": "open_weight", "score": 14.0},
    {"model_label": "Ministral 3 14B", "variant": "", "vendor": "Mistral", "family": "Mistral", "kind": "open_weight", "score": 11.0},
    {"model_label": "Llama 4 Scout", "variant": "", "vendor": "Meta", "family": "Llama", "kind": "open_weight", "score": 10.0},
    {"model_label": "Ministral 3 8B", "variant": "", "vendor": "Mistral", "family": "Mistral", "kind": "open_weight", "score": 9.0},
    {"model_label": "Ministral 3 3B", "variant": "", "vendor": "Mistral", "family": "Mistral", "kind": "open_weight", "score": 7.0},
    {"model_label": "Phi-4 Mini", "variant": "", "vendor": "Microsoft", "family": "Phi", "kind": "open_weight", "score": 6.0},
    {"model_label": "Phi-4", "variant": "", "vendor": "Microsoft", "family": "Phi", "kind": "open_weight", "score": 5.0},
)

# Ollama tag prefix -> reference family, so the panel can point out which rows
# are the nearest published relatives of a model you actually ran. It's a
# family hint only: your llama3.1:8b is NOT the Llama 4 Scout in the table.
FAMILY_HINTS: dict[str, str] = {
    "llama": "Llama",
    "qwen": "Qwen",
    "gemma": "Gemma",
    "phi": "Phi",
    "mistral": "Mistral",
    "mixtral": "Mistral",
    "deepseek": "DeepSeek",
    "gpt-oss": "gpt-oss",
}


def family_for_local_model(name: str) -> str | None:
    """Best-guess reference family for an Ollama tag such as 'qwen2.5:14b'."""
    stem = name.split(":")[0].lower()
    for prefix, family in FAMILY_HINTS.items():
        if stem.startswith(prefix):
            return family
    return None


def seed_reference_data() -> int:
    """Insert the seed rows once. Returns how many rows were added."""
    if database.count_benchmark_rows(seed_only=True):
        return 0
    rows = [
        {
            **row,
            "index_name": SEED_SOURCE,
            "source_name": SEED_SOURCE,
            "source_url": SEED_SOURCE_URL,
            "retrieved_at": SEED_RETRIEVED,
            "is_seed": True,
        }
        for row in SEED_ROWS
    ]
    database.insert_benchmark_rows(rows)
    return len(rows)
