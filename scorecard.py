"""One table, every run: quality, speed, cost and how the model was served.

The per-run pages answer "how did this model do?". The scorecard answers the
question you actually have when choosing a model, which is comparative and
multi-dimensional: *what is the best quality I can get under 40 tok/s, or
inside 6 GB of VRAM, or for free?* No single column answers that. The point of
this table is to put the columns side by side and let you sort by whichever
constraint is real for you today.

## Why rows are runs, not models

Tempting to collapse to one row per model. Resisted, because the same model
served two ways is genuinely two different things to choose between: Ollama at
Q4_K_M and llama.cpp at Q8_0 differ in both quality and speed, and averaging
them would hide exactly the trade-off the table exists to show. When you do
want them merged, sort by model and read the adjacent rows.

## Why the comparability tag is a column and not a footnote

Litmus Scores are only comparable within an identical setup -- same dataset,
same metrics, same judge. In a table sorted by score, rows from incompatible
setups sit next to each other looking directly comparable when they are not.
The tag makes that checkable at a glance: matching tags compare, differing
tags do not, and no one has to remember which run used which judge.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

import database
import harness
import perf
import scoring


@dataclass
class Row:
    """One run, flattened into everything worth comparing."""
    run_id: int
    model: str
    model_type: str
    runtime: str | None
    quantization: str | None
    vram_bytes: int | None
    dataset: str | None
    judge: str | None
    status: str
    started_at: str | None
    setup_tag: str
    composite: scoring.Composite | None
    metrics: dict[str, float | None] = field(default_factory=dict)
    measured: perf.RunPerf | None = None
    #: task key -> the latest completed benchmark_runs row for this model.
    benchmarks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- display helpers ---------------------------------------------------
    @property
    def score(self) -> float | None:
        return self.composite.score if self.composite else None

    @property
    def thin(self) -> bool:
        return bool(self.composite and self.composite.thin)

    @property
    def vram_label(self) -> str:
        """Only Ollama reports residency, so a blank here means 'not reported'."""
        if not self.vram_bytes:
            return "--"
        return f"{self.vram_bytes / 1_000_000_000:.2f} GB"

    @property
    def tps_label(self) -> str:
        return self.measured.tps_label if self.measured else "--"

    @property
    def ttft_label(self) -> str:
        return self.measured.ttft_label if self.measured else "--"

    @property
    def cost_label(self) -> str:
        return self.measured.cost_label if self.measured else "--"

    def benchmark_label(self, task_key: str) -> str:
        """A benchmark score with its interval, or a dash.

        The interval travels with the number everywhere it is shown. A bare
        '62.5%' from eight questions reads as precision it does not have.
        """
        row = self.benchmarks.get(task_key)
        if not row or row.get("score") is None:
            return "--"
        percent = row["score"] * 100
        stderr = row.get("stderr")
        if stderr is None:
            return f"{percent:.1f}%"
        return f"{percent:.1f}% ± {stderr * harness.Z_95 * 100:.1f}"

    def benchmark_samples(self, task_key: str) -> int | None:
        row = self.benchmarks.get(task_key)
        return row.get("samples") if row else None

    @property
    def served_label(self) -> str:
        parts = [p for p in (self.runtime, self.quantization) if p]
        return " · ".join(parts) if parts else "--"


@dataclass
class Scorecard:
    """Rows plus the metric and benchmark columns they span."""
    rows: list[Row] = field(default_factory=list)
    metric_labels: list[str] = field(default_factory=list)
    #: Benchmark tasks any row actually has a score for -- an empty column for
    #: a task nobody has run is noise, not information.
    benchmark_tasks: list[harness.Task] = field(default_factory=list)

    @property
    def setup_groups(self) -> dict[str, list[Row]]:
        """Rows bucketed by comparability tag.

        A group with more than one row is a set that can legitimately be ranked
        against itself. Everything else is a single observation that happens to
        be in the same table.
        """
        groups: dict[str, list[Row]] = {}
        for row in self.rows:
            groups.setdefault(row.setup_tag, []).append(row)
        return groups

    @property
    def comparable_groups(self) -> int:
        return sum(1 for rows in self.setup_groups.values() if len(rows) > 1)

    @property
    def any_measured(self) -> bool:
        return any(r.measured and r.measured.measured for r in self.rows)


def build(runs: list[dict[str, Any]]) -> Scorecard:
    """Assemble a scorecard from stored eval-run rows.

    Runs with no scored metrics are dropped: a queued or instantly-failed run
    contributes nothing to a comparison and would only add a row of dashes.
    """
    rows: list[Row] = []
    labels: list[str] = []

    # Benchmarks attach per *model*, not per eval run: a GSM8K score is a
    # property of the weights, and re-running the judge-based eval doesn't
    # change it. Cached per model id so a table of thirty runs across four
    # models makes four queries rather than thirty.
    bench_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for run in runs:
        summary = run.get("summary") or []
        if not summary:
            continue

        per_metric: dict[str, float | None] = {}
        for entry in summary:
            label = entry.get("metric_name")
            if not label:
                continue
            if label not in labels:
                labels.append(label)
            per_metric[label] = entry.get("average_score")

        rows.append(Row(
            run_id=int(run["id"]),
            model=run.get("model_name") or "unknown",
            model_type=run.get("model_type") or "local",
            runtime=run.get("runtime"),
            quantization=run.get("quantization"),
            vram_bytes=run.get("vram_bytes"),
            dataset=run.get("dataset_name"),
            judge=run.get("judge_model"),
            status=run.get("status") or "unknown",
            started_at=run.get("started_at"),
            setup_tag=scoring.comparability_key(
                run.get("dataset_id"), run.get("metrics"), run.get("judge_model")
            ),
            composite=scoring.composite(summary),
            metrics=per_metric,
            measured=perf.from_run(run),
            benchmarks=_benchmarks_for(run.get("model_id") or "", bench_cache),
        ))

    # Best first, but only within what is measurable: unscored rows sink rather
    # than being treated as a zero they never earned.
    rows.sort(key=lambda r: (r.score is None, -(r.score or 0.0)))
    labels.sort()

    scored_tasks = {key for row in rows for key in row.benchmarks}
    tasks = [t for t in harness.TASKS if t.key in scored_tasks]
    return Scorecard(rows=rows, metric_labels=labels, benchmark_tasks=tasks)


def _benchmarks_for(
    model_id: str, cache: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    if not model_id:
        return {}
    if model_id not in cache:
        cache[model_id] = database.latest_benchmark_scores(model_id)
    return cache[model_id]


HEADERS = [
    "run_id", "model", "model_type", "runtime", "quantization",
    "dataset", "judge", "setup_tag", "status", "started_at",
    "litmus_score", "thin_evidence",
]
TAIL_HEADERS = [
    "ttft_p50_ms", "ttft_p90_ms", "output_tps_p50", "output_tps_p90",
    "prompt_tokens", "completion_tokens", "cost_usd", "vram_bytes",
]


def to_csv(card: Scorecard) -> str:
    """The whole table as CSV, one row per run.

    Numbers stay numeric here -- no '5.80 GB', no '$0.0012'. The display
    formatting belongs in the template; a CSV exists to be sorted and filtered
    by a spreadsheet, and a unit suffix turns every one of those columns into
    text that sorts alphabetically.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    bench_cols: list[str] = []
    for task in card.benchmark_tasks:
        bench_cols += [f"{task.key}_percent", f"{task.key}_margin95", f"{task.key}_samples"]
    writer.writerow(HEADERS + card.metric_labels + bench_cols + TAIL_HEADERS)

    for row in card.rows:
        measured = row.measured
        writer.writerow(
            [
                row.run_id, row.model, row.model_type, row.runtime or "",
                row.quantization or "", row.dataset or "", row.judge or "",
                row.setup_tag, row.status, row.started_at or "",
                "" if row.score is None else row.score,
                "yes" if row.thin else "",
            ]
            + [
                "" if row.metrics.get(label) is None else round(row.metrics[label], 4)
                for label in card.metric_labels
            ]
            + _benchmark_cells(row, card.benchmark_tasks)
            + [
                _num(measured.ttft_p50) if measured else "",
                _num(measured.ttft_p90) if measured else "",
                _num(measured.tps_p50) if measured else "",
                _num(measured.tps_p90) if measured else "",
                measured.prompt_tokens if measured else "",
                measured.completion_tokens if measured else "",
                "" if not measured or measured.total_cost_usd is None
                else measured.total_cost_usd,
                row.vram_bytes or "",
            ]
        )
    return buf.getvalue()


def _benchmark_cells(row: Row, tasks: list[harness.Task]) -> list[Any]:
    """Score, interval and sample count per task -- flat, numeric, sortable."""
    cells: list[Any] = []
    for task in tasks:
        entry = row.benchmarks.get(task.key)
        if not entry or entry.get("score") is None:
            cells += ["", "", ""]
            continue
        stderr = entry.get("stderr")
        cells += [
            round(entry["score"] * 100, 2),
            round(stderr * harness.Z_95 * 100, 2) if stderr is not None else "",
            entry.get("samples") or "",
        ]
    return cells


def _num(value: float | None) -> str | float:
    return "" if value is None else round(value, 2)
