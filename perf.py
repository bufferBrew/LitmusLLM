"""Speed, token and cost measurement for individual model calls.

## Why this exists

LitmusLLM could always tell you whether a model was *good*. It could not tell
you what that quality cost you -- how long the first token took, how fast the
rest arrived, how many tokens went in and out, or what the bill was. Those are
the numbers every hosted-model leaderboard leads with, and without them a
local model and a frontier API model can't be compared on anything except
opinion.

## Why percentiles, not averages

A mean latency hides exactly the behaviour that makes a model annoying to use.
One 40-second stall inside fifty 2-second calls barely moves the mean and
completely changes how the thing feels. So every aggregate here carries p50,
p90 and p99 alongside the mean, and the UI leads with p50/p90. This matches
how OpenRouter publishes provider speed, which is the point: a number you
can't line up against the published one isn't much use for comparison.

## What gets measured, and what deliberately doesn't

Only the **model under test**. The judge is not measured, ever. Judge latency
is a property of your harness -- which judge you picked, running where -- not
of the model you are trying to characterise, and folding it in would make two
runs of the same model on different judges look like different models.

## The honest caveats, which the UI repeats

  * **Local throughput is a property of your machine**, not of the model. A
    7B at 45 tok/s on this laptop says nothing about that same 7B on a rented
    H100. Only cloud figures are comparable to published ones.
  * **Cost is zero for local models** because there is no invoice, not because
    inference is free. Electricity and the hardware are real; they just aren't
    per-token, so there is no honest per-token figure to report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class CallMetrics:
    """What one completion cost, in time and tokens.

    Every field is optional because runtimes differ in what they admit to.
    Ollama, LM Studio and llama.cpp all report usage when asked; some cloud
    providers omit it on streamed responses. A missing number is recorded as
    None and excluded from aggregates rather than silently counted as zero,
    which would drag every average toward a value nobody measured.
    """
    total_ms: float
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    streamed: bool = False

    @property
    def generation_ms(self) -> float | None:
        """Time spent producing tokens, excluding the wait for the first one."""
        if self.ttft_ms is None:
            return None
        return max(0.0, self.total_ms - self.ttft_ms)

    @property
    def output_tps(self) -> float | None:
        """Output tokens per second *after generation begins*.

        Measured over the post-TTFT window on purpose. Dividing tokens by wall
        time instead would blend queueing and prompt processing into what is
        supposed to be a decode-speed number, and would punish a long prompt
        for something that isn't decode at all.
        """
        gen = self.generation_ms
        if not self.completion_tokens or gen is None or gen <= 0:
            return None
        return self.completion_tokens * 1000.0 / gen

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms, 1),
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "output_tps": round(self.output_tps, 2) if self.output_tps is not None else None,
            "cost_usd": self.cost_usd,
            "streamed": self.streamed,
        }


class Stopwatch:
    """Wall-clock timer for one call, with a first-token mark.

    `perf_counter` rather than `time.time`: it is monotonic, so an NTP
    correction mid-generation can't produce a negative latency.
    """

    __slots__ = ("_start", "_first_token")

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._first_token: float | None = None

    def mark_first_token(self) -> None:
        """Record the first token's arrival. Only the first call counts.

        Guarded because an OpenAI-compatible stream opens with a role-only
        delta carrying no text; treating that as the first token would report
        a TTFT that no user ever experiences.
        """
        if self._first_token is None:
            self._first_token = time.perf_counter()

    @property
    def ttft_ms(self) -> float | None:
        if self._first_token is None:
            return None
        return (self._first_token - self._start) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def finish(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        streamed: bool = False,
    ) -> CallMetrics:
        return CallMetrics(
            total_ms=self.elapsed_ms,
            ttft_ms=self.ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            streamed=streamed,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile, matching numpy's default method.

    Written out rather than pulled from a library because it is eight lines and
    numpy is a heavy dependency to add for eight lines.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


@dataclass(frozen=True)
class Distribution:
    """One measured quantity summarised the way speed numbers should be."""
    label: str
    unit: str
    n: int
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p99: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    @property
    def measured(self) -> bool:
        return self.n > 0

    def as_dict(self) -> dict[str, Any]:
        rounder = (lambda v: round(v, 2) if v is not None else None)
        return {
            "label": self.label, "unit": self.unit, "n": self.n,
            "mean": rounder(self.mean), "p50": rounder(self.p50),
            "p90": rounder(self.p90), "p99": rounder(self.p99),
            "min": rounder(self.minimum), "max": rounder(self.maximum),
        }


def summarise(label: str, unit: str, values: Iterable[float | None]) -> Distribution:
    """Collapse a series into a Distribution, dropping unmeasured entries."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return Distribution(label=label, unit=unit, n=0)
    return Distribution(
        label=label,
        unit=unit,
        n=len(clean),
        mean=sum(clean) / len(clean),
        p50=percentile(clean, 50),
        p90=percentile(clean, 90),
        p99=percentile(clean, 99),
        minimum=min(clean),
        maximum=max(clean),
    )


@dataclass
class PerfRecorder:
    """Collects CallMetrics across a run and summarises them on demand.

    Appending is deliberately cheap and never raises: a run must not fail
    because its instrumentation did. A call that reported nothing measurable
    still counts toward `calls`, so `measured_calls` staying at zero is a
    visible signal that the runtime isn't reporting usage rather than a silent
    absence of data.
    """
    samples: list[CallMetrics] = field(default_factory=list)

    def add(self, metrics: CallMetrics | None) -> None:
        if metrics is not None:
            self.samples.append(metrics)

    @property
    def calls(self) -> int:
        return len(self.samples)

    @property
    def measured_calls(self) -> int:
        return sum(1 for s in self.samples if s.completion_tokens is not None)

    @property
    def total_cost_usd(self) -> float | None:
        costs = [s.cost_usd for s in self.samples if s.cost_usd is not None]
        return round(sum(costs), 6) if costs else None

    @property
    def total_tokens(self) -> dict[str, int]:
        return {
            "prompt": sum(s.prompt_tokens or 0 for s in self.samples),
            "completion": sum(s.completion_tokens or 0 for s in self.samples),
        }

    def distributions(self) -> dict[str, Distribution]:
        return {
            "ttft": summarise("Time to first token", "ms", (s.ttft_ms for s in self.samples)),
            "output_tps": summarise("Output speed", "tok/s", (s.output_tps for s in self.samples)),
            "total": summarise("Total request time", "ms", (s.total_ms for s in self.samples)),
        }

    def as_dict(self) -> dict[str, Any]:
        tokens = self.total_tokens
        return {
            "calls": self.calls,
            "measured_calls": self.measured_calls,
            "tokens": tokens,
            "total_cost_usd": self.total_cost_usd,
            "distributions": {k: d.as_dict() for k, d in self.distributions().items()},
        }


# ---------------------------------------------------------------------------
# Reading measurements back off a stored run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPerf:
    """A stored run's measurements, shaped for templates and exports.

    Deliberately tolerant: runs recorded before instrumentation existed have no
    `perf_json` at all, and a partial run may have timings but no token counts.
    Every accessor returns None rather than raising, so one un-instrumented run
    in a list cannot break the page rendering the others.
    """
    calls: int = 0
    measured_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float | None = None
    ttft: dict[str, Any] = field(default_factory=dict)
    output_tps: dict[str, Any] = field(default_factory=dict)
    total: dict[str, Any] = field(default_factory=dict)

    @property
    def measured(self) -> bool:
        return self.calls > 0

    @property
    def ttft_p50(self) -> float | None:
        return self.ttft.get("p50")

    @property
    def ttft_p90(self) -> float | None:
        return self.ttft.get("p90")

    @property
    def tps_p50(self) -> float | None:
        return self.output_tps.get("p50")

    @property
    def tps_p90(self) -> float | None:
        return self.output_tps.get("p90")

    @property
    def has_tokens(self) -> bool:
        return bool(self.prompt_tokens or self.completion_tokens)

    @staticmethod
    def format_ms(value: float | None) -> str:
        """Milliseconds in the unit a human reads without converting.

        Applied to every latency figure, not just p50: showing '1.97 s' beside
        '2216 ms' makes two numbers a few hundred milliseconds apart look like
        an order of magnitude, which is exactly the misreading percentiles are
        supposed to prevent.
        """
        if value is None:
            return "--"
        return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"

    @property
    def ttft_label(self) -> str:
        return self.format_ms(self.ttft_p50)

    @property
    def ttft_p90_label(self) -> str:
        return self.format_ms(self.ttft_p90)

    @property
    def tps_label(self) -> str:
        value = self.tps_p50
        return "--" if value is None else f"{value:.1f} tok/s"

    @property
    def cost_label(self) -> str:
        """Cost as text, distinguishing 'free' from 'not known'.

        Local runs record an explicit 0.0 -- there is genuinely no invoice --
        while an unpriced cloud model records None. Rendering both as '$0.00'
        would turn 'we could not price this' into 'this was free'.
        """
        if self.total_cost_usd is None:
            return "not priced"
        if self.total_cost_usd == 0:
            return "no API cost"
        if self.total_cost_usd < 0.01:
            return f"${self.total_cost_usd:.4f}"
        return f"${self.total_cost_usd:.2f}"


def from_run(run: Any) -> RunPerf | None:
    """Build a RunPerf from a stored eval-run row, or None if uninstrumented."""
    if run is None:
        return None
    raw = run.get("perf_json") if hasattr(run, "get") else getattr(run, "perf_json", None)
    if not raw:
        return None
    try:
        import json
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not data.get("calls"):
        return None

    dists = data.get("distributions") or {}
    tokens = data.get("tokens") or {}
    return RunPerf(
        calls=int(data.get("calls") or 0),
        measured_calls=int(data.get("measured_calls") or 0),
        prompt_tokens=int(tokens.get("prompt") or 0),
        completion_tokens=int(tokens.get("completion") or 0),
        total_cost_usd=data.get("total_cost_usd"),
        ttft=dists.get("ttft") or {},
        output_tps=dists.get("output_tps") or {},
        total=dists.get("total") or {},
    )
