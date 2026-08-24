"""LLM plumbing: one adapter that talks to every local runtime and every cloud provider.

Two distinct jobs happen here, and it's worth being explicit about them
because conflating them is the usual source of confusion in eval tooling:

  1. **Generation** -- prompting the *model under test* to produce the
     `actual_output` that will be graded. `TargetModel` does this.

  2. **Judging** -- DeepEval's LLM-as-judge metrics prompting a (possibly
     different) model to score that output. `JudgeModel` does this, and
     implements DeepEval's `DeepEvalBaseLLM` contract.

Local models go through their runtime's **OpenAI-compatible** endpoint
(`/v1/chat/completions`) rather than any native API, so the exact same request
shape works for Ollama, LM Studio, llama.cpp, and for every cloud provider
LiteLLM fronts. Which host that endpoint lives on comes from the ModelSpec
(`spec.base_url`), not from a global -- that is the whole reason a single
comparison run can put the same weights under two different runtimes.

The fiddly part is schema-constrained judging. DeepEval hands `a_generate` a
Pydantic model and expects an instance back. Small local models are casual
about JSON -- they wrap it in prose, in markdown fences, or emit trailing
commentary -- so `_coerce_to_schema` extracts the JSON payload and retries a
few times before giving up with a message that names the judge model, since
"switch to a bigger judge" is almost always the fix.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

import perf

from config import (
    CLOUD_CONCURRENCY,
    LOCAL_CONCURRENCY,
    LOCAL_MODEL_API_KEY,
    MAX_TOKENS,
    REQUEST_TIMEOUT,
    TEMPERATURE,
)
from model_registry import ModelSpec

# DeepEval is imported lazily: it pulls in a large dependency tree, and we want
# the web UI to boot (and give a useful error) even if it isn't installed yet.
_DEEPEVAL_BASE: Any = None


def _deepeval_base_llm() -> Any:
    global _DEEPEVAL_BASE
    if _DEEPEVAL_BASE is None:
        try:
            from deepeval.models.base_model import DeepEvalBaseLLM
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "DeepEval is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        _DEEPEVAL_BASE = DeepEvalBaseLLM
    return _DEEPEVAL_BASE


class ModelCallError(RuntimeError):
    """A prompt to a model failed in a way the user needs to know about."""


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """Pull the most likely JSON object/array out of a chatty model response."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    # 1. Fenced code block wins if present.
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate

    # 2. Otherwise take the outermost balanced {...} or [...] span. Scanning for
    #    balance (rather than a greedy regex) survives nested objects and any
    #    prose the model wrapped around them.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    # 3. Nothing structured found -- hand back the raw text so the caller's
    #    validation error mentions what the model actually said.
    return text


def _coerce_to_schema(text: str, schema: type[BaseModel]) -> BaseModel:
    """Validate a raw model response against a Pydantic schema."""
    payload = extract_json(text)
    try:
        return schema.model_validate_json(payload)
    except ValidationError:
        # Some models double-encode: a JSON string containing JSON.
        try:
            decoded = json.loads(payload)
            if isinstance(decoded, str):
                return schema.model_validate_json(decoded)
            return schema.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ValueError(f"response did not match schema: {exc}") from exc


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

class _Transport:
    """Sends chat-completion requests to whichever backend a ModelSpec names.

    A semaphore caps in-flight requests per instance. Local models are
    memory-bound -- firing eight concurrent requests at a 14B model on a
    laptop makes everything slower, not faster -- so the local default is
    deliberately small.
    """

    def __init__(self, spec: ModelSpec, concurrency: int | None = None):
        self.spec = spec
        default = LOCAL_CONCURRENCY if spec.kind == "local" else CLOUD_CONCURRENCY
        self._sem = asyncio.Semaphore(concurrency or default)

        # Resolved once, here, so every request path -- sync, async, judge --
        # is guaranteed to hit the same endpoint for a given spec.
        self._base_url = spec.base_url or ""
        self._runtime_label = spec.runtime_label
        # LM Studio and llama.cpp ignore the bearer token; Ollama's OpenAI
        # shim wants one present. Sending it unconditionally is harmless and
        # keeps one code path instead of three.
        self._headers = {"Authorization": f"Bearer {LOCAL_MODEL_API_KEY}"}

        if spec.kind == "cloud" and spec.api_key_env and not os.getenv(spec.api_key_env):
            raise ModelCallError(
                f"{spec.label} needs {spec.api_key_env}, which is not set. "
                f"Add it to your .env file and restart the app."
            )

    # -- async ------------------------------------------------------------
    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        async with self._sem:
            if self.spec.kind == "local":
                return await self._alocal(prompt, system, json_mode, max_tokens)
            return await self._alitellm(prompt, system, json_mode, max_tokens)

    async def _alocal(
        self, prompt: str, system: str | None, json_mode: bool, max_tokens: int | None
    ) -> str:
        body = self._openai_body(prompt, system, json_mode, max_tokens)
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions", json=body, headers=self._headers
                )
        except httpx.HTTPError as exc:
            raise ModelCallError(
                f"Could not reach {self._runtime_label} at {self._base_url}: {exc}"
            ) from exc
        return self._parse_openai(resp)

    # -- measured (streaming) ---------------------------------------------
    # Only the model under test goes down this path. Streaming is what makes
    # time-to-first-token observable at all: a non-streamed request tells you
    # when the whole answer arrived and nothing about when it started.

    async def acomplete_measured(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, perf.CallMetrics]:
        """Complete a prompt and report what it cost in time and tokens.

        Falls back to a plain non-streamed call if streaming fails, because a
        run losing its speed numbers is a far better outcome than a run losing
        its results. The returned metrics say `streamed=False` in that case, so
        a missing TTFT is distinguishable from a TTFT of zero.
        """
        async with self._sem:
            try:
                if self.spec.kind == "local":
                    return await self._alocal_measured(prompt, system, max_tokens)
                return await self._alitellm_measured(prompt, system, max_tokens)
            except ModelCallError:
                raise
            except Exception:  # noqa: BLE001 - instrumentation must never fail a run
                watch = perf.Stopwatch()
                if self.spec.kind == "local":
                    text = await self._alocal(prompt, system, False, max_tokens)
                else:
                    text = await self._alitellm(prompt, system, False, max_tokens)
                return text, watch.finish(streamed=False)

    async def _alocal_measured(
        self, prompt: str, system: str | None, max_tokens: int | None
    ) -> tuple[str, perf.CallMetrics]:
        body = self._openai_body(prompt, system, False, max_tokens)
        body["stream"] = True
        # All three local runtimes honour this and emit a final usage-only
        # chunk. Without it the stream ends with no token counts at all.
        body["stream_options"] = {"include_usage": True}

        chunks: list[str] = []
        usage: dict[str, Any] = {}
        watch = perf.Stopwatch()
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST", f"{self._base_url}/chat/completions",
                    json=body, headers=self._headers,
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        return self._parse_openai(resp), watch.finish(streamed=False)
                    async for line in resp.aiter_lines():
                        piece, chunk_usage, done = _parse_sse_line(line)
                        if chunk_usage:
                            usage = chunk_usage
                        if piece:
                            watch.mark_first_token()
                            chunks.append(piece)
                        if done:
                            break
        except httpx.HTTPError as exc:
            raise ModelCallError(
                f"Could not reach {self._runtime_label} at {self._base_url}: {exc}"
            ) from exc

        return "".join(chunks).strip(), watch.finish(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=0.0,   # local inference has no invoice -- see perf.py
            streamed=True,
        )

    async def _alitellm_measured(
        self, prompt: str, system: str | None, max_tokens: int | None
    ) -> tuple[str, perf.CallMetrics]:
        from litellm import acompletion

        kwargs = self._litellm_kwargs(prompt, system, False, max_tokens)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        chunks: list[str] = []
        usage: Any = None
        watch = perf.Stopwatch()
        try:
            stream = await acompletion(**kwargs)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                piece = getattr(choices[0].delta, "content", None)
                if piece:
                    watch.mark_first_token()
                    chunks.append(piece)
        except Exception as exc:
            raise ModelCallError(f"{self.spec.label}: {exc}") from exc

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        return "".join(chunks).strip(), watch.finish(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(self.spec.name, prompt_tokens, completion_tokens),
            streamed=True,
        )

    async def _alitellm(
        self, prompt: str, system: str | None, json_mode: bool, max_tokens: int | None
    ) -> str:
        from litellm import acompletion  # imported lazily; litellm is slow to import

        kwargs = self._litellm_kwargs(prompt, system, json_mode, max_tokens)
        try:
            response = await acompletion(**kwargs)
        except Exception as exc:
            # Not every provider supports response_format; retry once plainly
            # rather than failing the whole run over a formatting flag.
            if json_mode and "response_format" in kwargs:
                kwargs.pop("response_format")
                try:
                    response = await acompletion(**kwargs)
                except Exception as inner:
                    raise ModelCallError(f"{self.spec.label}: {inner}") from inner
            else:
                raise ModelCallError(f"{self.spec.label}: {exc}") from exc
        return (response.choices[0].message.content or "").strip()

    # -- sync -------------------------------------------------------------
    # DeepEval's BaseMetric.measure() calls the synchronous generate(). We use
    # a_measure() everywhere, but implementing both keeps the adapter usable
    # from any DeepEval code path (and from a plain script).
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        if self.spec.kind == "local":
            body = self._openai_body(prompt, system, json_mode, max_tokens)
            try:
                with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                    resp = client.post(
                        f"{self._base_url}/chat/completions", json=body, headers=self._headers
                    )
            except httpx.HTTPError as exc:
                raise ModelCallError(
                    f"Could not reach {self._runtime_label} at {self._base_url}: {exc}"
                ) from exc
            return self._parse_openai(resp)

        from litellm import completion

        kwargs = self._litellm_kwargs(prompt, system, json_mode, max_tokens)
        try:
            response = completion(**kwargs)
        except Exception as exc:
            raise ModelCallError(f"{self.spec.label}: {exc}") from exc
        return (response.choices[0].message.content or "").strip()

    # -- shared -----------------------------------------------------------
    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _openai_body(
        self, prompt: str, system: str | None, json_mode: bool, max_tokens: int | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.spec.name,
            "messages": self._messages(prompt, system),
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens or MAX_TOKENS,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _litellm_kwargs(
        self, prompt: str, system: str | None, json_mode: bool, max_tokens: int | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.spec.name,
            "messages": self._messages(prompt, system),
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens or MAX_TOKENS,
            "timeout": REQUEST_TIMEOUT,
        }
        if self.spec.api_key_env:
            key = os.getenv(self.spec.api_key_env)
            if key:
                kwargs["api_key"] = key
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _parse_openai(self, resp: httpx.Response) -> str:
        label = self._runtime_label
        if resp.status_code >= 400:
            detail = resp.text[:400]
            if resp.status_code == 404:
                raise ModelCallError(
                    f"{label} returned 404 for '{self.spec.name}': {detail}. "
                    f"{_missing_model_hint(self.spec)}"
                )
            raise ModelCallError(f"{label} returned HTTP {resp.status_code}: {detail}")
        try:
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ModelCallError(f"Unexpected response shape from {label}: {exc}") from exc


def _parse_sse_line(line: str) -> tuple[str | None, dict[str, Any] | None, bool]:
    """Decode one server-sent-events line into (text, usage, done).

    The usage-only chunk at the end of a stream has an empty `choices` array,
    so content and usage are pulled independently rather than assuming any
    chunk carries both.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None, None, False
    payload = line[5:].strip()
    if payload == "[DONE]":
        return None, None, True
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None, None, False

    usage = data.get("usage") or None
    text = None
    for choice in data.get("choices") or []:
        delta = choice.get("delta") or {}
        # A stream opens with a role-only delta and no content; returning None
        # for it keeps that non-event out of the time-to-first-token mark.
        content = delta.get("content")
        if content:
            text = (text or "") + content
    return text, usage, False


def _estimate_cost(
    model: str, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """Per-call cost from LiteLLM's own pricing tables.

    Deliberately not a pricing table of our own. Prices change, and a stale
    hardcoded rate reported to four decimal places is worse than admitting we
    don't know -- an unknown model simply returns None.
    """
    if prompt_tokens is None and completion_tokens is None:
        return None
    try:
        from litellm import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )
        return round(float(prompt_cost) + float(completion_cost), 8)
    except Exception:  # noqa: BLE001 - unpriced model, offline table, API drift
        return None


def _missing_model_hint(spec: ModelSpec) -> str:
    """What to actually do about a 404, which is runtime-specific.

    Ollama can fetch a missing model on request; the other two cannot, so
    telling a llama.cpp user to `ollama pull` would just waste their time.
    """
    if spec.runtime == "ollama":
        return (
            f"The model is probably not pulled -- try `ollama pull {spec.name}` "
            f"or use the Pull button on /models."
        )
    if spec.runtime == "lmstudio":
        return (
            "LM Studio only serves models it has downloaded. Check the model key on "
            "/models, or download it in the LM Studio app."
        )
    return (
        "llama.cpp serves only the models in its preset (or the single GGUF it was "
        "launched with). Check the id on /models."
    )


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

class TargetModel:
    """The model under evaluation -- prompted to produce `actual_output`."""

    #: Kept deliberately plain. A heavy system prompt would evaluate the prompt
    #: as much as the model, which defeats the point of a model comparison.
    SYSTEM = "You are a helpful assistant. Answer accurately and concisely."

    def __init__(self, spec: ModelSpec, concurrency: int | None = None):
        self.spec = spec
        self._transport = _Transport(spec, concurrency)

    async def generate(self, prompt: str) -> str:
        return await self._transport.acomplete(prompt, system=self.SYSTEM)

    async def generate_measured(self, prompt: str) -> tuple[str, perf.CallMetrics]:
        """Generate, and report what the generation cost in time and tokens."""
        return await self._transport.acomplete_measured(prompt, system=self.SYSTEM)


class JudgeModel:  # subclasses DeepEvalBaseLLM at construction time
    """Factory for a DeepEval-compatible judge bound to a ModelSpec.

    Implemented as a factory rather than a plain subclass so that importing
    this module doesn't require DeepEval to be installed -- the base class is
    only resolved when a judge is actually built.
    """

    def __new__(cls, spec: ModelSpec, retries: int = 3):  # type: ignore[misc]
        base = _deepeval_base_llm()

        class _Judge(base):  # type: ignore[misc, valid-type]
            def __init__(self, spec: ModelSpec, retries: int):
                self.spec = spec
                self.retries = retries
                self._transport = _Transport(spec)
                super().__init__(model_name=spec.label)

            # -- DeepEvalBaseLLM contract ---------------------------------
            def load_model(self) -> Any:
                # Nothing to load: the model lives behind an HTTP endpoint.
                return None

            def get_model_name(self) -> str:
                return self.spec.label

            def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
                last: Exception | None = None
                for _ in range(self.retries):
                    raw = self._transport.complete(
                        prompt,
                        json_mode=schema is not None,
                        max_tokens=1024,
                        system=_JUDGE_SYSTEM if schema is not None else None,
                    )
                    if schema is None:
                        return raw
                    try:
                        return _coerce_to_schema(raw, schema)
                    except ValueError as exc:
                        last = exc
                raise ModelCallError(_schema_failure_message(self.spec.label, last))

            async def a_generate(
                self, prompt: str, schema: type[BaseModel] | None = None
            ) -> Any:
                last: Exception | None = None
                for attempt in range(self.retries):
                    raw = await self._transport.acomplete(
                        prompt,
                        json_mode=schema is not None,
                        max_tokens=1024,
                        system=_JUDGE_SYSTEM if schema is not None else None,
                    )
                    if schema is None:
                        return raw
                    try:
                        return _coerce_to_schema(raw, schema)
                    except ValueError as exc:
                        last = exc
                        # Brief backoff: a retry at temperature 0 can repeat the
                        # same malformed output, so give the sampler a moment.
                        await asyncio.sleep(0.2 * (attempt + 1))
                raise ModelCallError(_schema_failure_message(self.spec.label, last))

        return _Judge(spec, retries)


_JUDGE_SYSTEM = (
    "You are a strict evaluation judge. Reply with a single valid JSON object "
    "and nothing else -- no prose, no markdown fences, no explanation outside "
    "the JSON."
)


def _schema_failure_message(label: str, last: Exception | None) -> str:
    return (
        f"Judge model '{label}' could not produce valid JSON after several attempts "
        f"({last}). Small models often struggle with the structured output DeepEval "
        f"requires -- pick a larger judge (7B+, ideally 14B+) or a cloud judge in the "
        f"'Judge model' selector."
    )


async def smoke_test(spec: ModelSpec) -> str:
    """One tiny prompt, used to fail fast before a long run starts."""
    model = TargetModel(spec, concurrency=1)
    return await model.generate("Reply with the single word: ready")
