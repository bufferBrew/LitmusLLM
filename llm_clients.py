"""LLM plumbing: one adapter that talks to both Ollama and cloud providers.

Two distinct jobs happen here, and it's worth being explicit about them
because conflating them is the usual source of confusion in eval tooling:

  1. **Generation** -- prompting the *model under test* to produce the
     `actual_output` that will be graded. `TargetModel` does this.

  2. **Judging** -- DeepEval's LLM-as-judge metrics prompting a (possibly
     different) model to score that output. `JudgeModel` does this, and
     implements DeepEval's `DeepEvalBaseLLM` contract.

Local models go through Ollama's **OpenAI-compatible** endpoint
(`/v1/chat/completions`) rather than its native `/api/generate`, so the exact
same request shape works for Ollama and for every cloud provider LiteLLM
fronts.

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

from config import (
    CLOUD_CONCURRENCY,
    LOCAL_CONCURRENCY,
    LOCAL_MODEL_API_KEY,
    LOCAL_MODEL_BASE_URL,
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
                return await self._aollama(prompt, system, json_mode, max_tokens)
            return await self._alitellm(prompt, system, json_mode, max_tokens)

    async def _aollama(
        self, prompt: str, system: str | None, json_mode: bool, max_tokens: int | None
    ) -> str:
        body = self._openai_body(prompt, system, json_mode, max_tokens)
        headers = {"Authorization": f"Bearer {LOCAL_MODEL_API_KEY}"}
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    f"{LOCAL_MODEL_BASE_URL}/chat/completions", json=body, headers=headers
                )
        except httpx.HTTPError as exc:
            raise ModelCallError(f"Could not reach Ollama at {LOCAL_MODEL_BASE_URL}: {exc}") from exc
        return self._parse_openai(resp)

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
            headers = {"Authorization": f"Bearer {LOCAL_MODEL_API_KEY}"}
            try:
                with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                    resp = client.post(
                        f"{LOCAL_MODEL_BASE_URL}/chat/completions", json=body, headers=headers
                    )
            except httpx.HTTPError as exc:
                raise ModelCallError(
                    f"Could not reach Ollama at {LOCAL_MODEL_BASE_URL}: {exc}"
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

    @staticmethod
    def _parse_openai(resp: httpx.Response) -> str:
        if resp.status_code >= 400:
            detail = resp.text[:400]
            if resp.status_code == 404:
                raise ModelCallError(
                    f"Ollama returned 404: {detail}. The model is probably not pulled "
                    f"-- try `ollama pull <model>` or use the Pull button on /models."
                )
            raise ModelCallError(f"Ollama returned HTTP {resp.status_code}: {detail}")
        try:
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ModelCallError(f"Unexpected response shape from Ollama: {exc}") from exc


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
