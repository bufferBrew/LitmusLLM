"""The Litmus Score -- collapsing a run's metrics into one 0-100 number.

## Why a single number

Per-metric averages are the honest output of an eval, but they are hard to
hold in your head: "0.82 relevancy, 0.14 hallucination, 0.91 recall" against
"0.79 / 0.09 / 0.88" is a comparison most people give up on. Published
leaderboards solved this by aggregating -- MMLU-Pro, the Artificial Analysis
Intelligence Index, LMArena Elo are all one number per model, and that is why
people can actually rank models at a glance.

So LitmusLLM aggregates too. The Litmus Score is the mean of a run's metric
averages, each first flipped so that higher always means better, expressed
0-100.

## Why it is *not* an Intelligence Index score

This is the trap the aggregation invites, so it is worth being blunt: a
Litmus Score of 71 and an Intelligence Index of 71 have nothing to do with
each other. The index is ten fixed public benchmarks with known answers. A
Litmus Score is an LLM judge's opinion of *your* prompts, on *your* metrics,
graded by *your* judge model. Change the dataset, the metric set or the judge
and the number moves for reasons that have nothing to do with the model.

That is not a flaw -- it is the point of running your own evals -- but it does
mean the number is only meaningful *relative to other runs configured the same
way*. Hence `comparability_key` below, which the UI uses to show which runs
can honestly be ranked against each other and which cannot. A score without
that guard is exactly the kind of authoritative-looking meaningless figure
`benchmarks.py` refuses to plot.

## Why an unweighted mean

Weighting metrics would mean asserting that, say, faithfulness matters twice
as much as relevancy -- true for some applications, false for others, and
LitmusLLM does not know which one you are building. An unweighted mean over
the metrics *you chose to run* pushes that decision where it belongs: pick the
metrics you care about, and they count equally. Metric selection is the
weighting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import metrics_catalog

# Metrics land on 0-1; the score is shown 0-100 because a two-digit integer is
# what every leaderboard people already read uses, and decimals invite false
# precision an LLM judge cannot support.
SCALE = 100.0

# Below this many scored cases the mean is dominated by individual judge calls,
# which are noisy. The score is still shown -- suppressing it would be worse --
# but flagged, so nobody quotes a 5-case run as if it settled anything.
THIN_EVIDENCE_CASES = 10

# One metric is a measurement, not a profile. Composites of a single metric are
# flagged for the same reason.
THIN_EVIDENCE_METRICS = 2


@dataclass(frozen=True)
class Contribution:
    """One metric's part of the composite."""
    label: str
    raw: float               # the metric's own average, as measured
    good: float              # 0-1, flipped if the metric is lower-is-better
    inverted: bool           # True when lower raw is better
    cases: int


@dataclass(frozen=True)
class Composite:
    score: float                                  # 0-100
    contributions: list[Contribution] = field(default_factory=list)
    cases: int = 0                                # fewest cases any metric saw

    @property
    def metric_count(self) -> int:
        return len(self.contributions)

    @property
    def thin(self) -> bool:
        """True when the score rests on too little evidence to lean on."""
        return (self.cases < THIN_EVIDENCE_CASES
                or self.metric_count < THIN_EVIDENCE_METRICS)

    @property
    def caveat(self) -> str:
        """Plain-English reason the score is flagged, or '' if it is not."""
        reasons = []
        if self.metric_count < THIN_EVIDENCE_METRICS:
            reasons.append("only one metric")
        if self.cases < THIN_EVIDENCE_CASES:
            reasons.append(f"only {self.cases} test case{'' if self.cases == 1 else 's'}")
        return " and ".join(reasons)

    @property
    def band(self) -> str:
        """Coarse quality bucket, used for colour only."""
        if self.score >= 80:
            return "strong"
        if self.score >= 60:
            return "fair"
        if self.score >= 40:
            return "weak"
        return "poor"


def goodness(metric_label: str, average_score: float | None) -> float | None:
    """Normalise one metric average so that 1.0 is always the good end.

    Toxicity, Bias and Hallucination score 0.0 when the model behaved well.
    Averaging them raw alongside relevancy would rank the most toxic model
    highest, so they are flipped here rather than special-cased at each call
    site -- `metrics_catalog` already carries the direction as first-class data.
    """
    if average_score is None:
        return None
    spec = next((m for m in metrics_catalog.METRICS if m.label == metric_label), None)
    value = float(average_score)
    if spec is not None and not spec.higher_is_better:
        value = 1.0 - value
    # Judges occasionally return slightly out-of-range values; clamp so one bad
    # call cannot drag a composite outside 0-100.
    return min(1.0, max(0.0, value))


def composite(summary_rows: Iterable[dict[str, Any]]) -> Composite | None:
    """Build the Litmus Score from a run summary, or None if nothing scored.

    `summary_rows` is what `database.get_run_summary` returns: one row per
    metric with `metric_name`, `average_score` and `scored_cases`.
    """
    contributions: list[Contribution] = []
    for row in summary_rows or []:
        good = goodness(row["metric_name"], row.get("average_score"))
        if good is None or not row.get("scored_cases"):
            continue
        spec = next((m for m in metrics_catalog.METRICS
                     if m.label == row["metric_name"]), None)
        contributions.append(Contribution(
            label=row["metric_name"],
            raw=float(row["average_score"]),
            good=good,
            inverted=bool(spec is not None and not spec.higher_is_better),
            cases=int(row["scored_cases"]),
        ))

    if not contributions:
        return None

    mean = sum(c.good for c in contributions) / len(contributions)
    return Composite(
        score=round(mean * SCALE, 1),
        contributions=contributions,
        # The weakest link: a composite is only as well-evidenced as its
        # thinnest metric, so report that rather than a flattering maximum.
        cases=min(c.cases for c in contributions),
    )


# --------------------------------------------------------------------------
# Comparability
# --------------------------------------------------------------------------
# Two Litmus Scores can be ranked against each other only when they were
# produced the same way. Rather than leave that to the reader's memory, every
# run gets a short fingerprint of the three things that change the number
# without the model changing: the dataset, the metric set, and the judge.
# Matching tags mean the scores are directly comparable; differing tags mean
# they are not, however similar they look.

def comparability_key(dataset_id: Any, metrics: Any, judge_model: Any) -> str:
    """Stable 6-character tag for a run's evaluation setup."""
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except (TypeError, ValueError):
            metrics = [metrics]
    payload = json.dumps(
        {
            "dataset": dataset_id,
            # Metric *order* is a UI artefact, not part of the setup.
            "metrics": sorted(str(m) for m in (metrics or [])),
            "judge": judge_model or "",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:6]


def comparability_note(run: dict[str, Any]) -> str:
    """Human-readable version of the fingerprint, for tooltips."""
    metrics = run.get("metrics") or []
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except (TypeError, ValueError):
            metrics = [metrics]
    return (
        f"Comparable only with runs on '{run.get('dataset_name') or 'the same dataset'}' "
        f"using the same {len(metrics)} metric(s), judged by "
        f"{run.get('judge_model') or 'the same judge'}."
    )
