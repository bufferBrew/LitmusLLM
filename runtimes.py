"""Local inference runtimes: where they live, whether they're up, how to start them.

LitmusLLM originally spoke to exactly one local backend (Ollama) through one
hardcoded base URL. That was fine until the obvious comparison question came
up -- *is my model slower because it's a worse model, or because of how it's
being served?* -- which you cannot answer without running the same weights
under more than one runtime.

So the local side is now plural. Three runtimes are supported:

  * **Ollama**    -- model manager + server. Autoloads by tag.
  * **LM Studio** -- GUI-first, but ships a headless server and a `lms` CLI.
  * **llama.cpp** -- `llama-server`, plus the newer `llama serve` router build
                     which autoloads models the way Ollama does.

All three expose an OpenAI-compatible `/v1/chat/completions`, which is why one
transport can drive all of them. They differ in two places only, and those two
places are all this module exists to paper over:

  1. **Discovery.** Ollama has `/api/tags` (rich: quantization, parameter
     count, disk size). LM Studio has a native `/api/v0/models` that is nearly
     as rich. Plain llama.cpp has only OpenAI's `/v1/models`, which is a list
     of bare ids -- so metadata gets inferred from the filename instead.

  2. **Starting.** Each has a different command, and llama.cpp has *two*
     depending on which build you have.

## Why auto-start is guarded

Spawning processes on a user's machine deserves a narrow blast radius, so it
only happens when both conditions hold: the configured host is loopback, and
the runtime's binary is on PATH. Pointing at a remote server or running inside
Docker means we can only report the problem -- we have no business starting
something over there, and no ability to anyway.

## Why quantization is surfaced everywhere

It is the single most common way a local-vs-published comparison goes silently
wrong. Ollama serves Q4_K_M by default; leaderboards report bf16. Those are not
the same model, and the gap is real. Every card this module produces carries a
quantization field for exactly that reason, inferred from the filename when the
runtime won't tell us outright.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from config import (
    LLAMACPP_HOST,
    LLAMACPP_START_COMMAND,
    LMSTUDIO_HOST,
    LOCAL_MODEL_API_KEY,
    DEFAULT_RUNTIME,
    OLLAMA_HOST,
    REQUEST_TIMEOUT,
    RUNTIME_START_TIMEOUT,
)


class RuntimeUnavailable(RuntimeError):
    """A local runtime can't be reached and couldn't be auto-started."""


# Kept as an alias because the Ollama-only name is referenced throughout the
# app and in stored error text. Same exception, older spelling.
OllamaUnavailable = RuntimeUnavailable


# ---------------------------------------------------------------------------
# Runtime definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Runtime:
    """One local inference server LitmusLLM knows how to talk to.

    `prefix` is the model-id namespace. Ollama's is `local` rather than
    `ollama` purely for backwards compatibility: runs recorded before this
    module existed stored ids like `local:llama3.1:8b`, and renaming the
    prefix would orphan every one of them.
    """
    key: str                       # 'ollama' | 'lmstudio' | 'llamacpp'
    label: str
    prefix: str                    # model-id namespace
    host: str                      # scheme://host:port, no trailing slash
    api_base: str                  # OpenAI-compatible root, e.g. .../v1
    health_path: str               # cheap GET that proves the server is alive
    binaries: tuple[str, ...]      # candidate executables, in preference order
    start_args: dict[str, tuple[str, ...]]   # binary name -> argv tail
    install_url: str
    start_hint: str                # the command a human would type
    extra_bin_dirs: tuple[str, ...] = ()     # searched when PATH misses it
    manages_models: bool = False   # can it pull/delete on our behalf?

    @property
    def health_url(self) -> str:
        return f"{self.host}{self.health_path}"


def _home(*parts: str) -> str:
    return str(Path.home().joinpath(*parts))


RUNTIMES: dict[str, Runtime] = {
    "ollama": Runtime(
        key="ollama",
        label="Ollama",
        prefix="local",
        host=OLLAMA_HOST,
        api_base=f"{OLLAMA_HOST}/v1",
        health_path="/api/tags",
        binaries=("ollama",),
        start_args={"ollama": ("serve",)},
        install_url="https://ollama.com",
        start_hint="ollama serve",
        manages_models=True,
    ),
    "lmstudio": Runtime(
        key="lmstudio",
        label="LM Studio",
        prefix="lmstudio",
        host=LMSTUDIO_HOST,
        api_base=f"{LMSTUDIO_HOST}/v1",
        health_path="/v1/models",
        binaries=("lms",),
        start_args={"lms": ("server", "start")},
        install_url="https://lmstudio.ai",
        start_hint="lms server start",
        # LM Studio installs its CLI outside the default PATH; the app's own
        # bootstrap step (`npx lmstudio install-cli`) is what normally links it.
        extra_bin_dirs=(_home(".lmstudio", "bin"), _home(".cache", "lm-studio", "bin")),
    ),
    "llamacpp": Runtime(
        key="llamacpp",
        label="llama.cpp",
        prefix="llamacpp",
        host=LLAMACPP_HOST,
        api_base=f"{LLAMACPP_HOST}/v1",
        health_path="/health",
        # `llama serve` (the router build) is preferred over bare
        # `llama-server`: it autoloads models on demand, so it can be started
        # cold without being told which GGUF to hold in memory.
        binaries=("llama", "llama-server"),
        start_args={
            "llama": ("serve",),
            "llama-server": (),      # port/host appended in _start_argv
        },
        install_url="https://github.com/ggml-org/llama.cpp",
        start_hint="llama serve",
        extra_bin_dirs=(_home(".llama-app"), "/opt/homebrew/bin", "/usr/local/bin"),
    ),
}

#: Ordered for display. Ollama first because it is the default and the one
#: most likely to be present.
RUNTIME_ORDER: tuple[str, ...] = ("ollama", "lmstudio", "llamacpp")

_BY_PREFIX: dict[str, Runtime] = {rt.prefix: rt for rt in RUNTIMES.values()}


def runtime_for_prefix(prefix: str) -> Runtime | None:
    return _BY_PREFIX.get(prefix)


def get(key: str) -> Runtime:
    try:
        return RUNTIMES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown runtime '{key}'. Known: {', '.join(RUNTIME_ORDER)}") from exc


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
# Short descriptions for local model families. A small built-in table rather
# than a scrape of ollama.com/library -- it keeps the app fully offline and
# never breaks when someone else's markup changes.

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

#: Families that can't do chat completion, so they must not appear as
#: evaluation targets or judges.
EMBEDDING_FAMILIES = {"nomic-bert", "bert"}

# Quantization as it appears in GGUF filenames. Ordered alternation matters:
# IQ4_XS must be tried before Q4, or the match truncates.
_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    r"IQ\d(?:_[A-Z]+)*"          # IQ4_XS, IQ3_M
    r"|Q\d(?:_[0-9KMSXL]+)*"     # Q4_K_M, Q8_0, Q5_1
    r"|BF16|FP16|F16|FP32|F32|FP8|INT8|INT4"
    r"|\d+bit"                   # 4bit, 8bit (MLX naming)
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Parameter counts as they appear in model names: 8B, 3.8B, 70b, 1_5B.
_PARAM_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:[._]\d+)?)\s*[bB](?![A-Za-z0-9])")


def infer_quantization(*candidates: str | None) -> str:
    """Best-effort quantization label from a model name or file path.

    Only Ollama reports this outright. For the other two it has to come out of
    the filename, which is lossy but overwhelmingly better than showing
    'unknown' -- an unlabelled quantization is precisely how a Q4 local model
    ends up being compared against a bf16 published score.
    """
    for candidate in candidates:
        if not candidate:
            continue
        match = _QUANT_RE.search(candidate)
        if match:
            return match.group(1).upper().replace("BIT", "-bit")
    return "unknown"


def infer_parameter_size(*candidates: str | None) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        match = _PARAM_RE.search(candidate)
        if match:
            return f"{match.group(1).replace('_', '.')}B"
    return "unknown"


def humanise_size(num_bytes: int) -> str:
    if not num_bytes:
        return "unknown"
    gb = num_bytes / 1_000_000_000
    return f"{gb:.2f} GB" if gb >= 1 else f"{num_bytes / 1_000_000:.0f} MB"


def humanise_date(raw: str | None) -> str:
    """Format a runtime's RFC3339 timestamp for display.

    Ollama reports nanosecond precision (e.g. '...T21:50:16.07926522+02:00'),
    which `datetime.fromisoformat` rejects on Python < 3.11, so the fractional
    part is truncated to the six digits it accepts.
    """
    if not raw:
        return "unknown"
    cleaned = str(raw).replace("Z", "+00:00")
    cleaned = re.sub(r"\.(\d{6})\d+", r".\1", cleaned)
    try:
        return datetime.fromisoformat(cleaned).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(raw)[:16]


def family_of(details: dict[str, Any], name: str) -> str:
    family = (details.get("family") or "").lower()
    if family:
        return family
    # Fall back to the leading segment of the tag, e.g. 'llama3.1:8b' -> 'llama3'.
    stem = name.rsplit("/", 1)[-1]
    return stem.split(":")[0].split(".")[0].split("-")[0].lower()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def is_up(rt: Runtime, timeout: float = 3.0) -> bool:
    """Cheap liveness probe.

    A 404 still counts as up: `llama-server` builds without a `/health` route
    answer 404 there, and a server that refuses a route is unambiguously a
    server that is listening. Only a transport error means 'not running'.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(rt.health_url)
            return resp.status_code < 500
    except (httpx.HTTPError, OSError):
        return False


async def ollama_is_up(timeout: float = 3.0) -> bool:
    """Back-compat shim for the pre-multi-runtime health endpoint."""
    return await is_up(RUNTIMES["ollama"], timeout)


def _is_loopback(host: str) -> bool:
    return any(h in host for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def find_binary(rt: Runtime) -> str | None:
    """Locate the runtime's executable on PATH, then in its known install dirs."""
    for name in rt.binaries:
        found = shutil.which(name)
        if found:
            return found
    for directory in rt.extra_bin_dirs:
        for name in rt.binaries:
            candidate = Path(directory) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _start_argv(rt: Runtime, binary: str) -> list[str]:
    """Build the launch command for `binary`.

    `LITMUSLLM_LLAMACPP_START` exists because llama.cpp is the one runtime with
    a genuinely open-ended launch line -- context size, GPU layers, a specific
    GGUF path. Anyone running it in classic single-model mode needs to supply
    their own command, and no default we pick could be right for them.
    """
    if rt.key == "llamacpp" and LLAMACPP_START_COMMAND:
        import shlex
        return shlex.split(LLAMACPP_START_COMMAND)

    name = Path(binary).name
    argv = [binary, *rt.start_args.get(name, ())]

    if rt.key == "llamacpp" and name == "llama-server":
        # Bare `llama-server` binds :8080 by default, but say so explicitly so
        # a non-default LLAMACPP_HOST is actually honoured.
        parsed = httpx.URL(rt.host)
        argv += ["--host", parsed.host or "127.0.0.1", "--port", str(parsed.port or 8080)]
    return argv


async def ensure_running(rt: Runtime) -> None:
    """Make sure `rt` is reachable, starting it locally if we're able to.

    Raises RuntimeUnavailable with instructions the user can actually act on.
    """
    if await is_up(rt):
        return

    if not _is_loopback(rt.host):
        raise RuntimeUnavailable(
            f"{rt.label} is not reachable at {rt.host}. That is not a local address, so "
            f"LitmusLLM will not try to start it. If you're running under Docker Compose, "
            f"check the service is healthy (`docker compose ps`) and that the host is "
            f"pointed at it."
        )

    binary = find_binary(rt)
    if not binary:
        searched = ", ".join(rt.binaries)
        raise RuntimeUnavailable(
            f"{rt.label} is not running at {rt.host} and none of [{searched}] is on PATH. "
            f"Install it from {rt.install_url}, then run `{rt.start_hint}`."
        )

    argv = _start_argv(rt, binary)
    try:
        # Detached so it outlives this request; output discarded because these
        # servers log continuously and we don't consume them.
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeUnavailable(
            f"Tried to start {rt.label} automatically (`{' '.join(argv)}`) but failed: "
            f"{exc}. Start it yourself with `{rt.start_hint}`."
        ) from exc

    deadline = RUNTIME_START_TIMEOUT
    waited = 0.0
    while waited < deadline:
        await asyncio.sleep(0.5)
        waited += 0.5
        if await is_up(rt):
            return

    raise RuntimeUnavailable(
        f"Started {rt.label} (`{' '.join(argv)}`) but it did not become reachable "
        f"within {deadline:.0f}s at {rt.host}. Run `{rt.start_hint}` in a terminal "
        f"and check its output."
    )


async def ensure_ollama_running() -> None:
    """Back-compat shim: the Ollama-specific spelling of ensure_running."""
    await ensure_running(RUNTIMES["ollama"])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _card(rt: Runtime, name: str, **overrides: Any) -> dict[str, Any]:
    """Assemble one model card in the shape the templates expect."""
    details = overrides.pop("details", {}) or {}
    family = overrides.pop("family", None) or family_of(details, name)
    card: dict[str, Any] = {
        "id": f"{rt.prefix}:{name}",
        "kind": "local",
        "runtime": rt.key,
        "runtime_label": rt.label,
        "name": name,
        "label": name,
        "provider": rt.key,
        "size_bytes": 0,
        "size": "unknown",
        "modified_at": "unknown",
        "family": family,
        "family_label": (details.get("family") or family or "local").title(),
        "parameter_size": "unknown",
        "quantization": "unknown",
        "format": details.get("format") or "unknown",
        "blurb": FAMILY_BLURBS.get(family, f"Locally hosted model served by {rt.label}."),
        "chat_capable": family not in EMBEDDING_FAMILIES,
        "requires_key": False,
        "key_present": True,
        "deletable": rt.manages_models,
        "loaded": None,          # None = the runtime doesn't say
    }
    card.update(overrides)
    return card


async def _list_ollama(rt: Runtime) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(f"{rt.host}/api/tags")
        resp.raise_for_status()
        payload = resp.json()

    cards = []
    for entry in payload.get("models", []):
        name = entry.get("name") or entry.get("model") or "unknown"
        details = entry.get("details") or {}
        size = entry.get("size", 0)
        cards.append(_card(
            rt, name,
            details=details,
            size_bytes=size,
            size=humanise_size(size),
            modified_at=humanise_date(entry.get("modified_at")),
            parameter_size=details.get("parameter_size") or infer_parameter_size(name),
            quantization=details.get("quantization_level") or infer_quantization(name),
        ))
    return cards


async def _list_lmstudio(rt: Runtime) -> list[dict[str, Any]]:
    """LM Studio's native endpoint first, OpenAI's as the fallback.

    `/api/v0/models` reports architecture, quantization and whether the model
    is currently resident in memory -- none of which `/v1/models` carries.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        entries: list[dict[str, Any]] = []
        try:
            resp = await client.get(f"{rt.host}/api/v0/models")
            if resp.status_code < 400:
                entries = resp.json().get("data", [])
        except (httpx.HTTPError, ValueError):
            entries = []

        if not entries:
            resp = await client.get(f"{rt.host}/v1/models")
            resp.raise_for_status()
            entries = resp.json().get("data", [])

    cards = []
    for entry in entries:
        name = entry.get("id") or "unknown"
        arch = (entry.get("arch") or "").lower()
        kind = (entry.get("type") or "").lower()
        size = entry.get("size_bytes") or 0
        cards.append(_card(
            rt, name,
            family=arch or None,
            parameter_size=infer_parameter_size(name),
            quantization=entry.get("quantization") or infer_quantization(name),
            size_bytes=size,
            size=humanise_size(size),
            # 'embeddings' is LM Studio's own type tag -- trust it over the
            # family-name guess, which only knows the families we listed.
            chat_capable=kind not in ("embeddings", "embedding"),
            loaded=(entry.get("state") == "loaded") if entry.get("state") else None,
            context_length=entry.get("max_context_length"),
        ))
    return cards


async def _list_llamacpp(rt: Runtime) -> list[dict[str, Any]]:
    """llama.cpp exposes only OpenAI's `/v1/models`, so metadata is inferred.

    The router build (`llama serve`) lists every model in its preset and
    autoloads on demand. A classic `llama-server -m foo.gguf` lists exactly the
    one model it was launched with. Both come back through the same route.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(f"{rt.host}/v1/models")
        resp.raise_for_status()
        entries = resp.json().get("data", [])

        # /props names the loaded model, which in single-model mode is the only
        # place the GGUF path (and therefore the real quantization) appears.
        model_path = None
        try:
            props = await client.get(f"{rt.host}/props")
            if props.status_code < 400:
                raw = props.json().get("model_path")
                model_path = raw if raw and raw != "none" else None
        except (httpx.HTTPError, ValueError):
            pass

    cards = []
    for entry in entries:
        name = entry.get("id") or "unknown"
        meta = entry.get("meta") or {}
        size = meta.get("size") or 0
        cards.append(_card(
            rt, name,
            parameter_size=meta.get("n_params") and f"{meta['n_params'] / 1e9:.1f}B"
                           or infer_parameter_size(name, model_path),
            quantization=infer_quantization(name, model_path),
            size_bytes=size,
            size=humanise_size(size),
            modified_at=humanise_date(entry.get("created") and datetime.fromtimestamp(
                entry["created"]).isoformat()),
            context_length=meta.get("n_ctx_train"),
        ))
    return cards


_LISTERS = {
    "ollama": _list_ollama,
    "lmstudio": _list_lmstudio,
    "llamacpp": _list_llamacpp,
}


async def list_models(rt: Runtime, *, autostart: bool = True) -> list[dict[str, Any]]:
    """Model cards for one runtime, starting it first if it isn't up."""
    if autostart:
        await ensure_running(rt)
    elif not await is_up(rt):
        raise RuntimeUnavailable(f"{rt.label} is not running at {rt.host}.")

    cards = await _LISTERS[rt.key](rt)
    cards.sort(key=lambda c: c["name"])
    return cards


# ---------------------------------------------------------------------------
# Aggregate status
# ---------------------------------------------------------------------------

@dataclass
class RuntimeStatus:
    """What we know about one runtime right now, for the UI and /api/health."""
    runtime: Runtime
    up: bool
    installed: bool
    binary: str | None = None
    models: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    note: str | None = None

    @property
    def key(self) -> str:
        return self.runtime.key

    @property
    def label(self) -> str:
        return self.runtime.label

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def can_autostart(self) -> bool:
        return self.installed and _is_loopback(self.runtime.host)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "host": self.runtime.host,
            "up": self.up,
            "installed": self.installed,
            "binary": self.binary,
            "can_autostart": self.can_autostart,
            "model_count": self.model_count,
            "start_hint": self.runtime.start_hint,
            "install_url": self.runtime.install_url,
            "error": self.error,
            "note": self.note,
        }


async def probe(rt: Runtime, *, autostart: bool = False) -> RuntimeStatus:
    """Inspect one runtime without ever raising.

    A status page that 500s because one of three backends is down is worse
    than useless, so every failure lands in `error` and the caller decides how
    loudly to say it.
    """
    binary = find_binary(rt)
    status = RuntimeStatus(runtime=rt, up=False, installed=binary is not None, binary=binary)

    if autostart:
        try:
            await ensure_running(rt)
        except RuntimeUnavailable as exc:
            status.error = str(exc)
            return status

    status.up = await is_up(rt)
    if not status.up:
        if not status.error:
            status.error = (
                f"{rt.label} is not running at {rt.host}."
                + (f" LitmusLLM can start it (`{rt.start_hint}`)." if status.can_autostart
                   else f" Install it from {rt.install_url}.")
            )
        return status

    try:
        status.models = await list_models(rt, autostart=False)
    except RuntimeUnavailable as exc:
        status.error = str(exc)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        status.error = f"{rt.label} is up but its model list could not be read: {exc}"

    if status.up and not status.models and not status.error:
        status.note = _empty_note(rt)
    return status


def _empty_note(rt: Runtime) -> str:
    """Why a running runtime might legitimately have nothing to offer."""
    if rt.key == "ollama":
        return "Running, but no models are pulled. Use the Pull box above."
    if rt.key == "lmstudio":
        return (
            "Running, but no models are downloaded. Add one in the LM Studio app, "
            "or run `lms get <model>`."
        )
    return (
        "Running, but serving no models. The router build loads from a models preset "
        "-- add GGUF files to it, or start a single model with "
        "`llama-server -m /path/to/model.gguf`."
    )


async def probe_all(*, autostart: bool = False) -> list[RuntimeStatus]:
    """Probe every runtime concurrently. Three cold TCP timeouts in series is
    a visibly slow page; in parallel it's one timeout."""
    results = await asyncio.gather(
        *(probe(RUNTIMES[key], autostart=autostart) for key in RUNTIME_ORDER)
    )
    return list(results)


async def all_local_models(*, autostart_default: bool = True) -> tuple[list[dict[str, Any]], list[RuntimeStatus]]:
    """Every local model across every runtime, plus per-runtime status.

    Only the default runtime is auto-started. Opening the models page should
    not spawn two more inference servers the user never asked for -- those
    start on demand, when a model belonging to them is actually run.
    """
    statuses = await asyncio.gather(*(
        probe(RUNTIMES[key], autostart=(key == DEFAULT_RUNTIME and autostart_default))
        for key in RUNTIME_ORDER
    ))
    models: list[dict[str, Any]] = []
    for status in statuses:
        models.extend(status.models)
    return models, list(statuses)


async def describe_model(runtime_key: str, model_name: str) -> dict[str, Any] | None:
    """Find one model's card on its runtime, or None if it isn't listed.

    Never raises. This is called to *annotate* a run, and a run must not fail
    because a metadata lookup did.
    """
    try:
        rt = get(runtime_key)
    except KeyError:
        return None
    try:
        cards = await list_models(rt, autostart=False)
    except (RuntimeUnavailable, httpx.HTTPError, ValueError, KeyError):
        return None
    for card in cards:
        if card["name"] == model_name:
            return card
    return None


async def quantization_for(runtime_key: str, model_name: str) -> str | None:
    """The quantization a runtime is actually serving a model at.

    Asking the runtime beats guessing from the name, and for Ollama it is the
    only way: a tag like 'llama3.2:3b' says nothing about precision, while
    /api/tags reports Q4_K_M outright. Falls back to the filename for the
    runtimes that don't report it, and to None when neither knows -- an honest
    blank is better than a confident wrong label.
    """
    card = await describe_model(runtime_key, model_name)
    if card:
        quant = card.get("quantization")
        if quant and quant != "unknown":
            return quant
    inferred = infer_quantization(model_name)
    return inferred if inferred != "unknown" else None


async def resident_memory(runtime_key: str, model_name: str) -> int | None:
    """Bytes of accelerator memory a model is holding *right now*, or None.

    Only Ollama reports this, via /api/ps. LM Studio says whether a model is
    loaded but not how much it occupies; llama.cpp says neither. Rather than
    estimate from the file size -- which ignores the KV cache, and so
    understates real usage by a lot at long context -- the other two return
    None and the UI shows a blank.

    Must be read *while the model is loaded*, which is why the caller takes
    this measurement mid-run rather than at export time: by then Ollama has
    very likely evicted the model and the answer would be None for everything.
    """
    if runtime_key != "ollama":
        return None
    rt = RUNTIMES["ollama"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{rt.host}/api/ps")
            if resp.status_code >= 400:
                return None
            payload = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        return None

    for entry in payload.get("models", []):
        if entry.get("name") == model_name or entry.get("model") == model_name:
            vram = entry.get("size_vram")
            return int(vram) if vram else None
    return None
