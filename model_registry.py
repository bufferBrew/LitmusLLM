"""Model discovery and identity.

Two sources of models feed the app:
  * **local**  -- whatever `ollama` currently has pulled, discovered live via
                  the native API at GET /api/tags.
  * **cloud**  -- a hand-maintained list of flagship models reached through
                  LiteLLM, each mapped to the env var holding its API key.

Both are normalised into a single `ModelSpec` so the rest of the app never
has to branch on "is this local or cloud" except where it genuinely matters
(which HTTP client to use, and whether an API key is required).

Model id format: "local:<ollama tag>" or "cloud:<litellm model string>".
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from config import OLLAMA_HOST, REQUEST_TIMEOUT


class OllamaUnavailable(RuntimeError):
    """Raised when Ollama can't be reached and couldn't be auto-started."""


@dataclass(frozen=True)
class ModelSpec:
    """A model the app can send prompts to."""
    id: str          # 'local:llama3.1:latest' | 'cloud:anthropic/claude-opus-5'
    kind: str        # 'local' | 'cloud'
    name: str        # routable name: ollama tag, or litellm 'provider/model'
    label: str       # display name
    provider: str = "ollama"
    api_key_env: str | None = None

    @property
    def requires_key(self) -> bool:
        return self.api_key_env is not None

    @property
    def key_present(self) -> bool:
        return not self.requires_key or bool(os.getenv(self.api_key_env or ""))


# ---------------------------------------------------------------------------
# Flagship cloud models
# ---------------------------------------------------------------------------
# The 10 models named in the original brief, plus the two current-generation
# Anthropic models. The brief's Anthropic entries (claude-3-5-sonnet-20241022,
# claude-3-opus-20240229) are legacy ids that have since been retired from the
# API -- they're kept below and flagged `legacy` so the list matches the spec,
# but the Claude 5 entries at the top are what you actually want to compare
# against today. Delete or extend either group freely; this list is just data.
#
# `name` is the LiteLLM model string. Provider routing for the two open-weight
# giants (Llama 405B, Qwen2.5 72B) goes through Together AI, which is one of
# several possible hosts -- swap the prefix if you use Fireworks, Groq, etc.

FLAGSHIP_MODELS: tuple[dict[str, Any], ...] = (
    {
        "name": "anthropic/claude-opus-5",
        "label": "Claude Opus 5",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "blurb": "Anthropic's most capable model. The strongest judge in this list.",
    },
    {
        "name": "anthropic/claude-sonnet-5",
        "label": "Claude Sonnet 5",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "blurb": "Balanced Anthropic model -- near-Opus quality at lower cost.",
    },
    {
        "name": "anthropic/claude-3-5-sonnet-20241022",
        "label": "Claude 3.5 Sonnet",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "blurb": "Legacy 2024 Sonnet. Superseded by Claude Sonnet 5.",
        "legacy": True,
    },
    {
        "name": "anthropic/claude-3-opus-20240229",
        "label": "Claude 3 Opus",
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "blurb": "Legacy 2024 Opus. Superseded by Claude Opus 5.",
        "legacy": True,
    },
    {
        "name": "deepseek/deepseek-chat",
        "label": "DeepSeek V3",
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "blurb": "Strong general-purpose MoE model at a very low price point.",
    },
    {
        "name": "deepseek/deepseek-coder",
        "label": "DeepSeek Coder V2",
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "blurb": "DeepSeek's code-specialised variant.",
    },
    {
        "name": "openai/gpt-4o",
        "label": "GPT-4o",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "blurb": "OpenAI's multimodal flagship of the 4o generation.",
    },
    {
        "name": "openai/gpt-4-turbo",
        "label": "GPT-4 Turbo",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "blurb": "Previous-generation GPT-4, still a common baseline.",
    },
    {
        "name": "gemini/gemini-1.5-pro",
        "label": "Gemini 1.5 Pro",
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "blurb": "Google's long-context flagship (up to 2M tokens).",
    },
    {
        "name": "gemini/gemini-1.5-flash",
        "label": "Gemini 1.5 Flash",
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "blurb": "Fast, cheap Gemini. A sensible cloud judge for bulk runs.",
    },
    {
        "name": "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "label": "Llama 3.1 405B",
        "provider": "together_ai",
        "api_key_env": "TOGETHER_API_KEY",
        "blurb": "Meta's largest open-weight model, hosted. The ceiling for Llama-family locals.",
    },
    {
        "name": "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        "label": "Qwen2.5 72B",
        "provider": "together_ai",
        "api_key_env": "TOGETHER_API_KEY",
        "blurb": "Alibaba's 72B open-weight model, hosted. Natural big sibling to local qwen2.5.",
    },
)

# Short descriptions for local model families, keyed by the `family` field
# Ollama reports. This is a small built-in table rather than a scrape of
# ollama.com/library -- it keeps the app fully offline and never breaks when
# the website's markup changes.
FAMILY_BLURBS: dict[str, str] = {
    "llama": "Meta's Llama family. Strong general-purpose instruction following.",
    "qwen": "Alibaba's Qwen family. Notably strong at multilingual and coding tasks.",
    "qwen2": "Alibaba's Qwen2 family. Strong multilingual and reasoning performance.",
    "qwen3": "Alibaba's Qwen3 family. Latest generation, with reasoning modes.",
    "gemma": "Google's Gemma family. Small, efficient, permissively licensed.",
    "gemma2": "Google's Gemma 2 family. Efficient models tuned for quality per parameter.",
    "gemma3": "Google's Gemma 3 family. Efficient open models with vision variants.",
    "phi3": "Microsoft's Phi-3 family. Small models trained on textbook-quality data.",
    "mistral": "Mistral AI's family. Fast, capable models with strong European language support.",
    "mixtral": "Mistral's mixture-of-experts models. High quality per active parameter.",
    "deepseek2": "DeepSeek's MoE family. Strong reasoning and coding.",
    "starcoder2": "BigCode's code-specialised family.",
    "codellama": "Meta's code-specialised Llama variant.",
    "nomic-bert": "Embedding model -- not suitable for generation or judging.",
    "bert": "Embedding model -- not suitable for generation or judging.",
}

# Families that can't do chat completion, so they must not appear as
# evaluation targets or judges.
EMBEDDING_FAMILIES = {"nomic-bert", "bert"}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

async def ollama_is_up(timeout: float = 3.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def ensure_ollama_running() -> None:
    """Make sure Ollama is reachable, starting it locally if we're able to.

    Auto-start is only attempted when the host is loopback *and* the `ollama`
    binary is on PATH -- inside Docker, or against a remote host, we can only
    report the problem. Raises OllamaUnavailable with actionable instructions.
    """
    if await ollama_is_up():
        return

    is_local = any(h in OLLAMA_HOST for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    binary = shutil.which("ollama")

    if is_local and binary:
        try:
            # Detached so it outlives this request; output discarded because
            # `ollama serve` logs continuously and we don't consume them.
            subprocess.Popen(
                [binary, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise OllamaUnavailable(
                f"Tried to start Ollama automatically but failed: {exc}. "
                f"Start it yourself with `ollama serve`."
            ) from exc

        # Give the server a few seconds to bind its port.
        for _ in range(20):
            await asyncio.sleep(0.5)
            if await ollama_is_up():
                return

        raise OllamaUnavailable(
            "Started `ollama serve` but it did not become reachable within 10s at "
            f"{OLLAMA_HOST}. Run `ollama serve` in a terminal and check its output."
        )

    if is_local:
        raise OllamaUnavailable(
            f"Ollama is not running at {OLLAMA_HOST} and the `ollama` binary is not on "
            "PATH. Install it from https://ollama.com, then run `ollama serve`."
        )

    raise OllamaUnavailable(
        f"Ollama is not reachable at {OLLAMA_HOST}. If you're running under Docker "
        "Compose, check that the `ollama` service is healthy (`docker compose ps`) "
        "and that OLLAMA_HOST points at it (http://ollama:11434)."
    )


def _humanise_size(num_bytes: int) -> str:
    gb = num_bytes / 1_000_000_000
    return f"{gb:.2f} GB" if gb >= 1 else f"{num_bytes / 1_000_000:.0f} MB"


def _humanise_date(raw: str | None) -> str:
    """Format Ollama's RFC3339 timestamps for display.

    Ollama reports nanosecond precision (e.g. '...T21:50:16.07926522+02:00'),
    which `datetime.fromisoformat` rejects on Python < 3.11, so the fractional
    part is truncated to the six digits it accepts.
    """
    if not raw:
        return "unknown"
    cleaned = raw.replace("Z", "+00:00")
    cleaned = re.sub(r"\.(\d{6})\d+", r".\1", cleaned)
    try:
        return datetime.fromisoformat(cleaned).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return raw[:16]


def _family_of(details: dict[str, Any], name: str) -> str:
    family = (details.get("family") or "").lower()
    if family:
        return family
    # Fall back to the leading segment of the tag, e.g. 'llama3.1:8b' -> 'llama3'.
    return name.split(":")[0].split(".")[0].lower()


async def list_local_models() -> list[dict[str, Any]]:
    """Fetch pulled Ollama models as enriched model cards.

    Raises OllamaUnavailable if the daemon can't be reached (after one
    auto-start attempt), so callers can surface a clear instruction.
    """
    await ensure_ollama_running()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(f"{OLLAMA_HOST}/api/tags")
        resp.raise_for_status()
        payload = resp.json()

    cards: list[dict[str, Any]] = []
    for entry in payload.get("models", []):
        name = entry.get("name") or entry.get("model") or "unknown"
        details = entry.get("details") or {}
        family = _family_of(details, name)
        cards.append({
            "id": f"local:{name}",
            "kind": "local",
            "name": name,
            "label": name,
            "provider": "ollama",
            "size_bytes": entry.get("size", 0),
            "size": _humanise_size(entry.get("size", 0)),
            "modified_at": _humanise_date(entry.get("modified_at")),
            "family": family,
            "family_label": (details.get("family") or family).title(),
            "parameter_size": details.get("parameter_size") or "unknown",
            "quantization": details.get("quantization_level") or "unknown",
            "format": details.get("format") or "unknown",
            "blurb": FAMILY_BLURBS.get(family, "Locally hosted model served by Ollama."),
            "chat_capable": family not in EMBEDDING_FAMILIES,
            "requires_key": False,
            "key_present": True,
        })
    cards.sort(key=lambda c: c["name"])
    return cards


def flagship_model_cards() -> list[dict[str, Any]]:
    """The cloud model list, annotated with live API-key availability."""
    return [
        {
            "id": f"cloud:{m['name']}",
            "kind": "cloud",
            "name": m["name"],
            "label": m["label"],
            "provider": m["provider"],
            "blurb": m["blurb"],
            "api_key_env": m["api_key_env"],
            "requires_key": True,
            "key_present": bool(os.getenv(m["api_key_env"])),
            "legacy": bool(m.get("legacy")),
            "chat_capable": True,
        }
        for m in FLAGSHIP_MODELS
    ]


_FLAGSHIP_BY_NAME = {m["name"]: m for m in FLAGSHIP_MODELS}


def parse_model_id(model_id: str) -> ModelSpec:
    """Turn a 'local:...' / 'cloud:...' id into a ModelSpec.

    A bare string with no prefix is treated as a local Ollama tag, which keeps
    the `judge` form field forgiving.
    """
    if model_id.startswith("local:"):
        name = model_id[len("local:") :]
        return ModelSpec(id=model_id, kind="local", name=name, label=name)

    if model_id.startswith("cloud:"):
        name = model_id[len("cloud:") :]
        entry = _FLAGSHIP_BY_NAME.get(name)
        if entry:
            return ModelSpec(
                id=model_id,
                kind="cloud",
                name=name,
                label=entry["label"],
                provider=entry["provider"],
                api_key_env=entry["api_key_env"],
            )
        # Not in the curated list: still routable, infer the provider prefix.
        provider = name.split("/")[0] if "/" in name else "openai"
        return ModelSpec(
            id=model_id,
            kind="cloud",
            name=name,
            label=name,
            provider=provider,
            api_key_env=f"{provider.upper().replace('-', '_')}_API_KEY",
        )

    return ModelSpec(id=f"local:{model_id}", kind="local", name=model_id, label=model_id)


async def pull_model(name: str) -> dict[str, Any]:
    """Pull a model through Ollama's HTTP API.

    Uses the API rather than `subprocess.run(['ollama','pull'])` on purpose:
    the app container in docker-compose has no `ollama` binary, only network
    access to the daemon, so the HTTP path is the one that works everywhere.
    """
    await ensure_ollama_running()
    # A cold pull of a large model easily exceeds the normal request timeout.
    timeout = httpx.Timeout(connect=10.0, read=3600.0, write=60.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/pull", json={"model": name, "stream": False}
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama refused the pull: {resp.text[:300]}")
        return {"ok": True, "model": name}


async def delete_model(name: str) -> dict[str, Any]:
    """Remove a pulled model via Ollama's HTTP API (equivalent to `ollama rm`)."""
    await ensure_ollama_running()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.request(
            "DELETE", f"{OLLAMA_HOST}/api/delete", json={"model": name}
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama refused the delete: {resp.text[:300]}")
        return {"ok": True, "model": name}
