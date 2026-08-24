"""Model discovery and identity.

Two sources of models feed the app:
  * **local**  -- whatever the local inference runtimes currently hold. Ollama,
                  LM Studio and llama.cpp are each discovered live; see
                  runtimes.py, which owns probing, auto-start and the
                  per-runtime differences in how models are listed.
  * **cloud**  -- a hand-maintained list of flagship models reached through
                  LiteLLM, each mapped to the env var holding its API key.

Both are normalised into a single `ModelSpec` so the rest of the app never
has to branch on "is this local or cloud" except where it genuinely matters
(which HTTP client to use, and whether an API key is required).

Model id format:
    "local:<ollama tag>"       -- Ollama. The prefix predates multi-runtime
                                  support and is kept unchanged so historical
                                  runs still resolve.
    "lmstudio:<model key>"     -- LM Studio
    "llamacpp:<model id>"      -- llama.cpp
    "cloud:<litellm model string>"
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

import runtimes
from config import DEFAULT_RUNTIME, OLLAMA_HOST, REQUEST_TIMEOUT
from runtimes import (  # re-exported: existing callers import these from here
    EMBEDDING_FAMILIES,
    FAMILY_BLURBS,
    OllamaUnavailable,
    RuntimeUnavailable,
    ensure_ollama_running,
    ollama_is_up,
)


@dataclass(frozen=True)
class ModelSpec:
    """A model the app can send prompts to."""
    id: str          # 'local:llama3.1:latest' | 'cloud:anthropic/claude-opus-5'
    kind: str        # 'local' | 'cloud'
    name: str        # routable name: ollama tag, or litellm 'provider/model'
    label: str       # display name
    provider: str = "ollama"
    api_key_env: str | None = None
    runtime: str = DEFAULT_RUNTIME   # which local runtime serves it; unused when cloud

    @property
    def base_url(self) -> str | None:
        """OpenAI-compatible root for local models; None for cloud (LiteLLM routes)."""
        if self.kind != "local":
            return None
        return runtimes.get(self.runtime).api_base

    @property
    def runtime_label(self) -> str:
        return runtimes.get(self.runtime).label if self.kind == "local" else self.provider

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

# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------
# Discovery, health and auto-start all live in runtimes.py. What is left here
# is the one thing that spans runtimes: assembling their cards into a single
# list the pickers can render.

async def list_local_models(*, runtime: str | None = None) -> list[dict[str, Any]]:
    """Model cards from every local runtime, or just the one named.

    Only the default runtime is auto-started. Listing models is a page render,
    and a page render has no business spawning two extra inference servers --
    the others start on demand, when a model belonging to them is actually run.

    Raises RuntimeUnavailable only when a *specific* runtime was asked for and
    is unreachable. The unfiltered call never raises: with three backends, one
    being down is a normal state to render, not an error to 500 on.
    """
    if runtime is not None:
        return await runtimes.list_models(runtimes.get(runtime))

    models, statuses = await runtimes.all_local_models()
    if not models:
        # Nothing anywhere. Surface the default runtime's reason, which is the
        # one the user most likely wants to hear about.
        primary = next((st for st in statuses if st.key == DEFAULT_RUNTIME), None)
        if primary is not None and primary.error:
            raise RuntimeUnavailable(primary.error)
    return models


async def local_runtime_status() -> list[dict[str, Any]]:
    """Per-runtime health, for the models page and /api/health."""
    return [st.as_dict() for st in await runtimes.probe_all()]


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
    """Turn a prefixed model id into a ModelSpec.

    Local prefixes are matched against the runtime registry rather than a
    hardcoded list, so adding a runtime in runtimes.py is enough to make its
    ids resolve here. Splitting on the *first* colon only matters for local
    ids, where the remainder is itself colon-bearing -- 'local:llama3.1:8b'
    must yield the tag 'llama3.1:8b', not 'llama3.1'.

    A bare string with no recognised prefix is treated as a model on the
    default runtime, which keeps the `judge` form field forgiving.
    """
    prefix, _, remainder = model_id.partition(":")

    rt = runtimes.runtime_for_prefix(prefix)
    if rt is not None and remainder:
        return ModelSpec(
            id=model_id,
            kind="local",
            name=remainder,
            label=remainder,
            provider=rt.key,
            runtime=rt.key,
        )

    if prefix == "cloud" and remainder:
        name = remainder
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

    default = runtimes.get(DEFAULT_RUNTIME)
    return ModelSpec(
        id=f"{default.prefix}:{model_id}",
        kind="local",
        name=model_id,
        label=model_id,
        provider=default.key,
        runtime=default.key,
    )


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
