"""The catalogue of evaluation metrics LitmusLLM exposes.

Each entry carries everything the UI and the runner need:
  * a plain-English explanation (shown as the info tooltip on the metric card),
  * the default pass threshold,
  * which *test-case fields* the metric requires, and
  * the score direction.

That last pair matters more than it looks. DeepEval metrics raise if a
required field is missing, and several of them are inverted -- for Toxicity,
Bias and Hallucination a score of 0.0 is the *good* outcome. Ranking models
without knowing the direction would silently crown the most toxic model the
winner, so direction is first-class data here rather than a special case
buried in the comparison code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    description: str            # 1-2 sentence explanation for the UI tooltip
    threshold: float
    higher_is_better: bool
    requires: tuple[str, ...] = ()   # LLMTestCase fields that must be non-empty
    category: str = "General"
    advanced: bool = False           # flagged in the UI as needing extra care
    notes: str = ""                  # extra guidance shown under the tooltip
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def requirement_label(self) -> str:
        """Human-readable summary of the data this metric needs."""
        pretty = {
            "expected_output": "expected output",
            "context": "context",
            "retrieval_context": "context",
            "tools_called": "tool-call trace",
            "expected_tools": "expected tools",
        }
        needs = [pretty.get(r, r) for r in self.requires]
        # 'context' can appear twice (context + retrieval_context); de-dupe.
        seen: list[str] = []
        for n in needs:
            if n not in seen:
                seen.append(n)
        return "Needs " + " + ".join(seen) if seen else "Works with input + output alone"


# Order here is the order shown in the UI.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="answer_relevancy",
        label="Answer Relevancy",
        description=(
            "Measures whether the response actually addresses the question that was "
            "asked, rather than drifting into related-but-unasked territory."
        ),
        threshold=0.7,
        higher_is_better=True,
        category="Core quality",
        notes="The best default metric -- it needs no extra dataset columns.",
    ),
    MetricSpec(
        key="faithfulness",
        label="Faithfulness",
        description=(
            "For RAG: checks that every claim in the answer is supported by the "
            "retrieved context, with no unsupported additions."
        ),
        threshold=0.7,
        higher_is_better=True,
        requires=("retrieval_context",),
        category="RAG",
    ),
    MetricSpec(
        key="hallucination",
        label="Hallucination",
        description=(
            "Detects information the model fabricated -- statements that contradict, "
            "or aren't grounded in, the supplied ground-truth context."
        ),
        threshold=0.5,
        higher_is_better=False,
        requires=("context",),
        category="Safety & truth",
        notes="Lower is better: 0.0 means nothing was fabricated.",
    ),
    MetricSpec(
        key="toxicity",
        label="Toxicity",
        description=(
            "Scores how harmful, offensive or abusive the response is. Useful as a "
            "guardrail check on any model you plan to expose to users."
        ),
        threshold=0.5,
        higher_is_better=False,
        category="Safety & truth",
        notes="Lower is better: 0.0 means no toxic content was detected.",
    ),
    MetricSpec(
        key="bias",
        label="Bias",
        description=(
            "Measures demographic, political or ideological slant in the response -- "
            "gendered assumptions, one-sided framing, stereotyping."
        ),
        threshold=0.5,
        higher_is_better=False,
        category="Safety & truth",
        notes="Lower is better: 0.0 means no bias was detected.",
    ),
    MetricSpec(
        key="contextual_recall",
        label="Contextual Recall",
        description=(
            "For RAG: how much of the information needed to produce the expected "
            "answer was actually present in the retrieved context."
        ),
        threshold=0.7,
        higher_is_better=True,
        requires=("expected_output", "retrieval_context"),
        category="RAG",
    ),
    MetricSpec(
        key="contextual_precision",
        label="Contextual Precision",
        description=(
            "For RAG: whether the retriever ranked the genuinely relevant chunks "
            "above the irrelevant ones."
        ),
        threshold=0.7,
        higher_is_better=True,
        requires=("expected_output", "retrieval_context"),
        category="RAG",
    ),
    MetricSpec(
        key="contextual_relevancy",
        label="Contextual Relevancy",
        description=(
            "For RAG: what proportion of the retrieved context is actually relevant "
            "to the question, i.e. how much noise the retriever pulled in."
        ),
        threshold=0.7,
        higher_is_better=True,
        requires=("retrieval_context",),
        category="RAG",
    ),
    MetricSpec(
        key="g_eval_correctness",
        label="G-Eval (Correctness)",
        description=(
            "A custom rubric graded by the judge model using chain-of-thought. This "
            "preset scores factual correctness against the expected output."
        ),
        threshold=0.7,
        higher_is_better=True,
        requires=("expected_output",),
        category="Custom rubric",
        advanced=True,
        notes=(
            "Advanced: G-Eval is only as good as its rubric and its judge. Results "
            "vary between judge models, so keep the judge fixed when comparing. "
            "Edit the criteria in metrics_catalog.py to grade for anything else."
        ),
        extra={
            "criteria": (
                "Determine whether the actual output is factually correct and "
                "consistent with the expected output. Penalise contradictions "
                "heavily, omissions of key facts moderately, and ignore differences "
                "in wording, formatting or verbosity."
            ),
            "params": ("input", "actual_output", "expected_output"),
        },
    ),
    MetricSpec(
        key="summarization",
        label="Summarization Score",
        description=(
            "For summarisation tasks: combines whether the summary stays factually "
            "aligned with the source text and whether it covers its key points."
        ),
        threshold=0.5,
        higher_is_better=True,
        category="Task-specific",
        notes=(
            "Treats the test case's input as the source document, so use it on a "
            "dataset whose 'input' column holds the text to summarise."
        ),
    ),
    MetricSpec(
        key="tool_correctness",
        label="Tool Call Accuracy",
        description=(
            "For agent workflows: compares the tools the model actually called "
            "against the tools it was expected to call."
        ),
        threshold=0.7,
        higher_is_better=True,
        requires=("tools_called", "expected_tools"),
        category="Agents",
        advanced=True,
        notes=(
            "Needs a dataset with 'tools_called' and 'expected_tools' columns "
            "(comma-separated tool names). Unlike every other metric here it is "
            "deterministic -- no judge model is involved."
        ),
    ),
)

METRICS_BY_KEY: dict[str, MetricSpec] = {m.key: m for m in METRICS}

# Default selection on first page load: cheap, needs no extra columns.
DEFAULT_METRIC_KEYS = ("answer_relevancy",)


def get_metric(key: str) -> MetricSpec:
    try:
        return METRICS_BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown metric '{key}'. Known: {sorted(METRICS_BY_KEY)}") from None


def resolve_metrics(keys: list[str]) -> list[MetricSpec]:
    """Validate and de-duplicate a list of metric keys, preserving catalog order."""
    unknown = [k for k in keys if k not in METRICS_BY_KEY]
    if unknown:
        raise KeyError(f"Unknown metric(s): {unknown}")
    chosen = set(keys)
    return [m for m in METRICS if m.key in chosen]


def categories() -> dict[str, list[MetricSpec]]:
    """Group the catalogue by category for the metric-picker grid."""
    grouped: dict[str, list[MetricSpec]] = {}
    for m in METRICS:
        grouped.setdefault(m.category, []).append(m)
    return grouped


def as_dicts() -> list[dict[str, Any]]:
    """Serialisable form for GET /api/metrics."""
    return [
        {
            "key": m.key,
            "label": m.label,
            "description": m.description,
            "threshold": m.threshold,
            "higher_is_better": m.higher_is_better,
            "requires": list(m.requires),
            "requirement_label": m.requirement_label,
            "category": m.category,
            "advanced": m.advanced,
            "notes": m.notes,
        }
        for m in METRICS
    ]


def missing_requirements(spec: MetricSpec, row: dict[str, Any]) -> list[str]:
    """Return the required fields a dataset row does not satisfy.

    Used to skip a metric for a specific case (with a recorded reason) instead
    of letting DeepEval raise and kill the whole run.
    """
    missing = []
    for req in spec.requires:
        if req == "retrieval_context" or req == "context":
            if not row.get("context"):
                missing.append(req)
        elif not row.get(req):
            missing.append(req)
    return missing
