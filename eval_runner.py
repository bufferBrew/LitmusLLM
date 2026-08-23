"""The evaluation engine: generate, judge, persist, and stay cancellable.

## How a run actually works

The original brief sketched `evaluate(model=..., test_cases=..., metrics=...)`,
but that isn't DeepEval's contract and it hides the step that matters most.
DeepEval never calls the model under test -- it grades text you hand it. So a
run here is explicitly two phases per test case:

    1. GENERATE  -- prompt the model under test, capture `actual_output`.
    2. JUDGE     -- build an LLMTestCase and let each metric score it.

We also drive metrics one at a time via `metric.a_measure(...)` rather than
calling DeepEval's bulk `evaluate()`. `evaluate()` is a blocking, all-or-
nothing call that prints to a console -- it gives no per-case progress and no
way to stop half-way. Looping ourselves is what makes the progress bar, the
Stop button and partial-result saving possible.

## Cancellation

Stopping is cooperative, not a kill. `asyncio.Event` is checked between test
cases and between metrics; when it's set the loop breaks, everything already
scored is already in SQLite, and the run is marked `stopped`. That's why a
stopped run still shows real numbers instead of vanishing.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import database
from database import utcnow
from llm_clients import JudgeModel, ModelCallError, TargetModel
from metrics_catalog import MetricSpec, get_metric, missing_requirements, resolve_metrics
import harness
import perf
import runtimes
from model_registry import ModelSpec, parse_model_id

log = logging.getLogger("litmusllm.runner")

# How many back-to-back generation failures before we conclude the model is
# simply unusable (not pulled, bad key) and abort rather than burn the dataset.
CONSECUTIVE_FAILURE_LIMIT = 3


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A live run: its cancellation flag and the task executing it."""
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


_eval_jobs: dict[int, Job] = {}
_comparison_jobs: dict[int, Job] = {}


def is_running(run_id: int) -> bool:
    return run_id in _eval_jobs


def request_stop(run_id: int) -> bool:
    """Ask an eval run to stop at its next checkpoint. False if not running."""
    job = _eval_jobs.get(run_id)
    if not job:
        return False
    job.cancel.set()
    return True


def request_comparison_stop(comparison_id: int) -> bool:
    """Stop a comparison and any child run currently executing."""
    job = _comparison_jobs.get(comparison_id)
    if not job:
        return False
    job.cancel.set()
    for child in database.get_comparison_child_runs(comparison_id):
        request_stop(int(child["id"]))
    return True


# ---------------------------------------------------------------------------
# DeepEval bridge
# ---------------------------------------------------------------------------

def _params_enum() -> Any:
    """Return the test-case params enum, tolerating the 3.9 rename.

    DeepEval 3.9 renamed `LLMTestCaseParams` to `SingleTurnParams` and emits a
    DeprecationWarning for the old name. Preferring the new one keeps us quiet
    on current versions without breaking on older ones.
    """
    try:
        from deepeval.test_case import SingleTurnParams
        return SingleTurnParams
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams
        return LLMTestCaseParams


def build_metric(spec: MetricSpec, judge: Any) -> Any:
    """Instantiate the DeepEval metric described by a catalogue entry.

    Every metric is given our judge explicitly -- DeepEval falls back to
    GPT-4o (and raises for a missing OPENAI_API_KEY) when `model` is None,
    which would quietly break the local-first promise.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BiasMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
        SummarizationMetric,
        ToolCorrectnessMetric,
        ToxicityMetric,
    )

    common = {
        "threshold": spec.threshold,
        "model": judge,
        # async_mode lets a single metric fan out its internal judge calls;
        # the per-model semaphore in llm_clients keeps that from swamping a
        # local GPU.
        "async_mode": True,
        "verbose_mode": False,
    }

    if spec.key == "answer_relevancy":
        return AnswerRelevancyMetric(**common)
    if spec.key == "faithfulness":
        return FaithfulnessMetric(**common)
    if spec.key == "hallucination":
        return HallucinationMetric(**common)
    if spec.key == "toxicity":
        return ToxicityMetric(**common)
    if spec.key == "bias":
        return BiasMetric(**common)
    if spec.key == "contextual_recall":
        return ContextualRecallMetric(**common)
    if spec.key == "contextual_precision":
        return ContextualPrecisionMetric(**common)
    if spec.key == "contextual_relevancy":
        return ContextualRelevancyMetric(**common)
    if spec.key == "summarization":
        return SummarizationMetric(**common)
    if spec.key == "tool_correctness":
        # Scoring here is deterministic (it diffs tool lists), but DeepEval 3.9
        # still resolves an `model` for its optional reason text -- and an
        # unset `model` silently defaults to GPT-4o, which then raises for a
        # missing OPENAI_API_KEY. Passing our judge keeps it local.
        return ToolCorrectnessMetric(
            threshold=spec.threshold, model=judge, verbose_mode=False
        )
    if spec.key == "g_eval_correctness":
        params_enum = _params_enum()
        wanted = spec.extra.get("params", ("input", "actual_output", "expected_output"))
        return GEval(
            name=spec.label,
            criteria=spec.extra["criteria"],
            evaluation_params=[getattr(params_enum, p.upper()) for p in wanted],
            **common,
        )

    raise KeyError(f"No DeepEval binding for metric '{spec.key}'")


_MEASURE_KWARGS: dict[str, Any] | None = None


def _measure_kwargs(metric: Any) -> dict[str, Any]:
    """Optional a_measure kwargs, filtered to those this version accepts.

    We want the console progress spinner off (we render our own progress) and
    Confident AI logging off (this app is local-first and must not phone home).
    Both flags are private and version-dependent, hence the signature check.
    """
    global _MEASURE_KWARGS
    if _MEASURE_KWARGS is None:
        try:
            params = inspect.signature(type(metric).a_measure).parameters
        except (TypeError, ValueError):
            params = {}
        desired = {"_show_indicator": False, "_log_metric_to_confident": False}
        _MEASURE_KWARGS = {k: v for k, v in desired.items() if k in params}
    return _MEASURE_KWARGS


def _to_tool_calls(names: list[str]) -> list[Any]:
    from deepeval.test_case import ToolCall
    return [ToolCall(name=n) for n in names]


def build_test_case(row: dict[str, Any], actual_output: str, needs_tools: bool) -> Any:
    """Assemble an LLMTestCase from a dataset row plus the generated output.

    `context` and `retrieval_context` are both populated from the row's single
    context column because they serve different metrics: DeepEval treats
    `context` as ground truth (Hallucination) and `retrieval_context` as what
    a retriever returned (Faithfulness, Contextual*). For a hand-written
    dataset the two are the same text.
    """
    from deepeval.test_case import LLMTestCase

    context = list(row.get("context") or [])
    kwargs: dict[str, Any] = {
        "input": row["input"],
        "actual_output": actual_output,
        "expected_output": row.get("expected_output") or None,
        "context": context or None,
        "retrieval_context": context or None,
    }
    if needs_tools:
        kwargs["tools_called"] = _to_tool_calls(row.get("tools_called") or [])
        kwargs["expected_tools"] = _to_tool_calls(row.get("expected_tools") or [])
    return LLMTestCase(**kwargs)


def build_prompt(row: dict[str, Any], grounded: bool) -> str:
    """Build the prompt sent to the model under test.

    When any selected metric grades the answer *against retrieved context*
    (Faithfulness, Contextual*), the model must actually see that context --
    otherwise we'd be scoring it on material it was never shown. Hallucination
    is deliberately excluded from `grounded`: the whole point there is to see
    whether the model invents facts when it *isn't* handed the answer.
    """
    context = row.get("context") or []
    if grounded and context:
        joined = "\n".join(f"- {chunk}" for chunk in context)
        return (
            "Answer the question using only the context below. If the context does "
            "not contain the answer, say so.\n\n"
            f"Context:\n{joined}\n\nQuestion: {row['input']}"
        )
    return row["input"]


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------

async def _execute_run(
    run_id: int,
    model_spec: ModelSpec,
    judge_spec: ModelSpec,
    metric_specs: list[MetricSpec],
    rows: list[dict[str, Any]],
    cancel: asyncio.Event,
) -> str:
    """Run every metric over every row. Returns the terminal status."""
    total = len(rows)
    database.update_eval_run(
        run_id, status="running", progress_total=total,
        progress_note=f"Warming up {model_spec.label}",
    )

    # Start whichever local runtimes this run needs -- and only those. A run
    # pitting an Ollama model against a cloud judge has no reason to boot
    # llama.cpp. Deduplicated because target and judge are usually the same
    # runtime, and starting it twice would just wait out the probe twice.
    for runtime_key in {s.runtime for s in (model_spec, judge_spec) if s.kind == "local"}:
        await runtimes.ensure_running(runtimes.get(runtime_key))

    # Now that the runtime is up it can be asked what it is actually serving,
    # which for Ollama is the only way to learn the quantization -- the tag
    # doesn't carry it.
    if model_spec.kind == "local":
        served_at = await runtimes.quantization_for(model_spec.runtime, model_spec.name)
        if served_at:
            database.update_eval_run(run_id, quantization=served_at)

    target = TargetModel(model_spec)
    needs_tools = any("tools_called" in m.requires for m in metric_specs)
    # Grounded prompting only when a metric grades against retrieval context.
    grounded = any("retrieval_context" in m.requires for m in metric_specs)

    # Fail fast: one throwaway prompt catches "model not pulled" and bad API
    # keys in two seconds instead of fifteen minutes into a run.
    try:
        await target.generate("Reply with the single word: ready")
    except ModelCallError as exc:
        raise ModelCallError(f"Model under test is not usable -- {exc}") from exc

    # Read memory residency straight after the warm-up, which is the one moment
    # the model is guaranteed to be loaded. Ollama evicts on a timer, so asking
    # after the run would usually return nothing.
    if model_spec.kind == "local":
        vram = await runtimes.resident_memory(model_spec.runtime, model_spec.name)
        if vram:
            database.update_eval_run(run_id, vram_bytes=vram)

    # Every metric gets an explicit judge. DeepEval falls back to GPT-4o when
    # `model` is None -- including for metrics that look judge-free -- so an
    # omission here would break the local-first guarantee.
    judge = JudgeModel(judge_spec)
    metrics = [(spec, build_metric(spec, judge)) for spec in metric_specs]

    # Speed and token accounting for the model under test only -- never the
    # judge, whose latency is a property of this harness rather than of the
    # model being characterised.
    recorder = perf.PerfRecorder()

    consecutive_failures = 0

    # try/finally rather than a save at each exit: a stopped or aborted run
    # has already paid for the tokens it generated, and its speed numbers are
    # exactly as valid as a completed run's. Losing them because the user hit
    # Stop would be throwing away measurements we already made.
    try:
        for index, row in enumerate(rows):
            if cancel.is_set():
                return "stopped"

            database.update_eval_run(
                run_id, progress_done=index,
                progress_note=f"Generating answer {index + 1}/{total} with {model_spec.label}",
            )

            # -- phase 1: generation ------------------------------------------
            try:
                actual_output, call_metrics = await target.generate_measured(
                    build_prompt(row, grounded)
                )
                recorder.add(call_metrics)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - recorded per case, not swallowed
                consecutive_failures += 1
                for spec, _ in metrics:
                    database.record_result(
                        eval_run_id=run_id, case_index=index, test_case_input=row["input"],
                        actual_output=None, metric_name=spec.label, score=None,
                        reason=f"Generation failed: {exc}", passed=None,
                        threshold=spec.threshold, status="error",
                    )
                if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                    raise ModelCallError(
                        f"{model_spec.label} failed to generate {consecutive_failures} times "
                        f"in a row -- aborting. Last error: {exc}"
                    ) from exc
                continue

            # -- phase 2: judging ---------------------------------------------
            for spec, metric in metrics:
                if cancel.is_set():
                    database.update_eval_run(run_id, progress_done=index)
                    return "stopped"

                missing = missing_requirements(spec, row)
                if missing:
                    database.record_result(
                        eval_run_id=run_id, case_index=index, test_case_input=row["input"],
                        actual_output=actual_output, metric_name=spec.label, score=None,
                        reason=(
                            f"Skipped: this test case has no "
                            f"{', '.join(m.replace('_', ' ') for m in missing)}. "
                            f"{spec.requirement_label}."
                        ),
                        passed=None, threshold=spec.threshold, status="skipped",
                    )
                    continue

                database.update_eval_run(
                    run_id,
                    progress_note=f"Scoring {spec.label} on case {index + 1}/{total}",
                )

                test_case = build_test_case(row, actual_output, needs_tools)
                try:
                    await metric.a_measure(test_case, **_measure_kwargs(metric))
                    score = float(metric.score) if metric.score is not None else None
                    passed = bool(metric.success) if metric.score is not None else None
                    database.record_result(
                        eval_run_id=run_id, case_index=index, test_case_input=row["input"],
                        actual_output=actual_output, metric_name=spec.label, score=score,
                        reason=getattr(metric, "reason", None), passed=passed,
                        threshold=spec.threshold, status="scored",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
                    log.warning("run %s: metric %s failed on case %s: %s",
                                run_id, spec.key, index, exc)
                    database.record_result(
                        eval_run_id=run_id, case_index=index, test_case_input=row["input"],
                        actual_output=actual_output, metric_name=spec.label, score=None,
                        reason=f"Metric error: {exc}", passed=None,
                        threshold=spec.threshold, status="error",
                    )

            database.update_eval_run(run_id, progress_done=index + 1)

    finally:
        database.update_eval_run(run_id, perf_json=json.dumps(recorder.as_dict()))

    return "stopped" if cancel.is_set() else "completed"


async def _run_wrapper(
    run_id: int,
    model_spec: ModelSpec,
    judge_spec: ModelSpec,
    metric_specs: list[MetricSpec],
    rows: list[dict[str, Any]],
    cancel: asyncio.Event,
    on_finish: Callable[[int, str], Any] | None = None,
) -> str:
    """Own the terminal state of a run: always writes a final status."""
    status = "failed"
    error: str | None = None
    already_finalised = False
    try:
        status = await _execute_run(run_id, model_spec, judge_spec, metric_specs, rows, cancel)
    except asyncio.CancelledError:
        # Hard cancel (process shutdown). Write the final state here, because
        # the re-raise below unwinds before the normal path would run.
        database.update_eval_run(
            run_id, status="stopped", error="Run cancelled.", completed_at=utcnow(),
            progress_note="Cancelled",
        )
        status, already_finalised = "stopped", True
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        log.exception("eval run %s failed", run_id)
        status, error = "failed", str(exc)
    finally:
        if not already_finalised:
            notes = {
                "completed": "Finished",
                "stopped": "Stopped early -- partial results saved",
                "failed": "Failed",
            }
            database.update_eval_run(
                run_id, status=status, error=error, completed_at=utcnow(),
                progress_note=notes.get(status, status),
            )
        _eval_jobs.pop(run_id, None)
        if on_finish:
            result = on_finish(run_id, status)
            if inspect.isawaitable(result):
                await result
    return status


def _quantization_of(spec: ModelSpec) -> str | None:
    """A first guess at the serving precision, from the model name alone.

    Only a guess: an Ollama tag like 'llama3.2:3b' carries no precision at all,
    whereas a llama.cpp GGUF filename usually does. `_execute_run` replaces
    this with the runtime's own answer as soon as the runtime is up -- this
    exists so that a run which never gets that far still records something.

    Cloud models return None: providers do not publish what precision they
    serve at, and inventing a label would be worse than an honest blank.
    """
    if spec.kind != "local":
        return None
    inferred = runtimes.infer_quantization(spec.name)
    return inferred if inferred != "unknown" else None


def start_eval_run(
    *,
    model_id: str,
    metric_keys: list[str],
    dataset_id: int,
    judge_model_id: str | None = None,
    comparison_run_id: int | None = None,
    on_finish: Callable[[int, str], Any] | None = None,
) -> int:
    """Create an eval run and start executing it in the background.

    Returns immediately with the new run id so the caller can redirect the
    user to a live progress view.
    """
    from config import DEFAULT_JUDGE

    metric_specs = resolve_metrics(metric_keys)
    if not metric_specs:
        raise ValueError("Select at least one metric.")

    rows = database.get_dataset_rows(dataset_id)
    if not rows:
        raise ValueError("That dataset has no rows.")

    model_spec = parse_model_id(model_id)
    judge_spec = parse_model_id(judge_model_id or f"local:{DEFAULT_JUDGE}")

    run_id = database.create_eval_run(
        model_name=model_spec.label,
        model_type=model_spec.kind,
        model_id=model_spec.id,
        metrics=[m.key for m in metric_specs],
        dataset_id=dataset_id,
        judge_model=judge_spec.label,
        total=len(rows),
        comparison_run_id=comparison_run_id,
        runtime=model_spec.runtime if model_spec.kind == "local" else model_spec.provider,
        quantization=_quantization_of(model_spec),
    )

    job = Job()
    _eval_jobs[run_id] = job
    job.task = asyncio.create_task(
        _run_wrapper(run_id, model_spec, judge_spec, metric_specs, rows, job.cancel, on_finish),
        name=f"litmusllm-run-{run_id}",
    )
    return run_id


# ---------------------------------------------------------------------------
# Comparison runs
# ---------------------------------------------------------------------------

def rank_models(
    rows: list[dict[str, Any]], metric_key_by_label: dict[str, str]
) -> list[dict[str, Any]]:
    """Assign a per-metric rank, respecting each metric's score direction.

    Toxicity, Bias and Hallucination are inverted -- 0.0 is the best possible
    score. Ranking everything descending would crown the most toxic model, so
    direction comes from the catalogue rather than being assumed.
    """
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_metric.setdefault(row["metric_name"], []).append(row)

    for metric_label, group in by_metric.items():
        key = metric_key_by_label.get(metric_label)
        higher_better = get_metric(key).higher_is_better if key else True

        scored = [g for g in group if g.get("average_score") is not None]
        unscored = [g for g in group if g.get("average_score") is None]
        scored.sort(key=lambda g: g["average_score"], reverse=higher_better)

        previous: float | None = None
        previous_rank = 0
        for position, entry in enumerate(scored, start=1):
            # Equal scores share a rank (standard competition ranking).
            if previous is not None and abs(entry["average_score"] - previous) < 1e-9:
                entry["rank"] = previous_rank
            else:
                entry["rank"] = position
                previous_rank = position
            previous = entry["average_score"]
        for entry in unscored:
            entry["rank"] = None

    return rows


def aggregate_comparison(comparison_id: int) -> list[dict[str, Any]]:
    """Recompute the comparison leaderboard from whatever child runs exist."""
    cmp_run = database.get_comparison_run(comparison_id)
    if not cmp_run:
        return []

    label_to_key = {get_metric(k).label: k for k in cmp_run["metrics"]}

    aggregated: list[dict[str, Any]] = []
    for child in database.get_comparison_child_runs(comparison_id):
        for summary in database.get_run_summary(int(child["id"])):
            aggregated.append({
                "eval_run_id": int(child["id"]),
                "model_name": child["model_name"],
                "metric_name": summary["metric_name"],
                "average_score": summary["average_score"],
                "pass_rate": summary["pass_rate"],
                "scored_cases": summary["scored_cases"],
            })

    ranked = rank_models(aggregated, label_to_key)
    database.replace_comparison_results(comparison_id, ranked)
    return ranked


async def _execute_comparison(
    comparison_id: int,
    model_ids: list[str],
    metric_keys: list[str],
    dataset_id: int,
    judge_model_id: str | None,
    mode: str,
    cancel: asyncio.Event,
) -> None:
    """Run the same eval across several models, then rank them."""
    database.update_comparison_run(comparison_id, status="running")

    async def run_one(model_id: str) -> None:
        if cancel.is_set():
            return
        run_id = start_eval_run(
            model_id=model_id,
            metric_keys=metric_keys,
            dataset_id=dataset_id,
            judge_model_id=judge_model_id,
            comparison_run_id=comparison_id,
        )
        job = _eval_jobs.get(run_id)
        if job and job.task:
            try:
                await job.task
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a failed model must not sink the comparison
                log.warning("comparison %s: child run %s failed", comparison_id, run_id)
        # Refresh standings after each model so partial results are visible.
        aggregate_comparison(comparison_id)

    if mode == "parallel":
        # Parallel is offered for cloud-heavy comparisons. Running several
        # local models at once will thrash memory on a laptop, which is why
        # sequential is the default in the UI.
        await asyncio.gather(*(run_one(m) for m in model_ids), return_exceptions=True)
    else:
        for model_id in model_ids:
            if cancel.is_set():
                break
            await run_one(model_id)

    aggregate_comparison(comparison_id)
    status = "stopped" if cancel.is_set() else "completed"
    database.update_comparison_run(comparison_id, status=status, completed_at=utcnow())


async def _comparison_wrapper(comparison_id: int, *args: Any) -> None:
    try:
        await _execute_comparison(comparison_id, *args)
    except asyncio.CancelledError:
        database.update_comparison_run(
            comparison_id, status="stopped", error="Cancelled.", completed_at=utcnow()
        )
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("comparison %s failed", comparison_id)
        database.update_comparison_run(
            comparison_id, status="failed", error=str(exc), completed_at=utcnow()
        )
    finally:
        _comparison_jobs.pop(comparison_id, None)


def start_comparison(
    *,
    model_ids: list[str],
    metric_keys: list[str],
    dataset_id: int,
    judge_model_id: str | None = None,
    mode: str = "sequential",
    name: str | None = None,
) -> int:
    """Kick off a multi-model comparison in the background."""
    from config import DEFAULT_JUDGE

    unique_models = list(dict.fromkeys(model_ids))  # de-dupe, keep order
    if len(unique_models) < 2:
        raise ValueError("Pick at least two models to compare.")

    metric_specs = resolve_metrics(metric_keys)
    if not metric_specs:
        raise ValueError("Select at least one metric.")
    if not database.get_dataset_rows(dataset_id):
        raise ValueError("That dataset has no rows.")

    judge_spec = parse_model_id(judge_model_id or f"local:{DEFAULT_JUDGE}")
    labels = [parse_model_id(m).label for m in unique_models]

    comparison_id = database.create_comparison_run(
        name=name or " vs ".join(labels[:3]) + (" ..." if len(labels) > 3 else ""),
        dataset_id=dataset_id,
        metrics=[m.key for m in metric_specs],
        model_ids=unique_models,
        judge_model=judge_spec.label,
        mode=mode,
    )

    job = Job()
    _comparison_jobs[comparison_id] = job
    job.task = asyncio.create_task(
        _comparison_wrapper(
            comparison_id, unique_models, [m.key for m in metric_specs],
            dataset_id, judge_model_id, mode, job.cancel,
        ),
        name=f"litmusllm-comparison-{comparison_id}",
    )
    return comparison_id


async def shutdown() -> None:
    """Cancel every in-flight task so uvicorn can exit promptly."""
    tasks = [j.task for j in (*_eval_jobs.values(), *_comparison_jobs.values()) if j.task]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Ground-truth benchmark runs
# ---------------------------------------------------------------------------
# Same job/cancel shape as an eval run, but the work happens in a subprocess,
# so stopping is a kill rather than a cooperative flag -- see harness.run_task.

_benchmark_jobs: dict[int, Job] = {}


def benchmark_is_running(run_id: int) -> bool:
    return run_id in _benchmark_jobs


def request_benchmark_stop(run_id: int) -> bool:
    job = _benchmark_jobs.get(run_id)
    if not job:
        return False
    job.cancel.set()
    return True


async def _execute_benchmark(
    run_id: int,
    task: harness.Task,
    model_spec: ModelSpec,
    *,
    limit: int | None,
    num_fewshot: int | None,
    cancel: asyncio.Event,
) -> str:
    database.update_benchmark_run(
        run_id, status="running", progress_note=f"Starting {task.label}"
    )

    await runtimes.ensure_running(runtimes.get(model_spec.runtime))

    # Recorded now, not at export: a benchmark score belongs to the weights
    # that produced it, and re-pulling a model at a different quantization
    # would otherwise silently relabel history.
    quantization = await runtimes.quantization_for(model_spec.runtime, model_spec.name)
    if quantization:
        database.update_benchmark_run(run_id, quantization=quantization)

    def note(text: str) -> None:
        database.update_benchmark_run(run_id, progress_note=text)

    outcome = await harness.run_task(
        task,
        base_url=model_spec.base_url or "",
        model_name=model_spec.name,
        output_dir=harness.RESULTS_DIR / f"run-{run_id}",
        limit=limit,
        num_fewshot=num_fewshot,
        on_progress=note,
        cancel=cancel,
    )

    database.update_benchmark_run(
        run_id,
        command=json.dumps(outcome.command),
        log_tail=outcome.log_tail or None,
    )

    if not outcome.ok or outcome.result is None:
        if cancel.is_set():
            database.update_benchmark_run(run_id, error=outcome.error)
            return "stopped"
        database.update_benchmark_run(run_id, error=outcome.error or "Unknown failure.")
        return "failed"

    result = outcome.result
    database.update_benchmark_run(
        run_id,
        metric_name=result.metric_key,
        score=result.score,
        stderr=result.stderr,
        samples=result.samples,
        num_fewshot=result.num_fewshot,
        raw_json=json.dumps(result.raw),
        progress_note=harness.calibration_hint(task, result),
    )
    return "completed"


async def _benchmark_wrapper(
    run_id: int,
    task: harness.Task,
    model_spec: ModelSpec,
    limit: int | None,
    num_fewshot: int | None,
    cancel: asyncio.Event,
) -> str:
    """Own the terminal state of a benchmark run: always writes a final status."""
    status = "failed"
    error: str | None = None
    try:
        status = await _execute_benchmark(
            run_id, task, model_spec,
            limit=limit, num_fewshot=num_fewshot, cancel=cancel,
        )
    except asyncio.CancelledError:
        database.update_benchmark_run(
            run_id, status="stopped", completed_at=utcnow(),
            progress_note="Stopped",
        )
        _benchmark_jobs.pop(run_id, None)
        raise
    except (harness.HarnessUnavailable, runtimes.RuntimeUnavailable) as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        log.exception("benchmark run %s failed", run_id)
        error = str(exc)
    finally:
        if not cancel.is_set() or status != "stopped":
            notes = {"completed": None, "stopped": "Stopped", "failed": "Failed"}
            fields: dict[str, Any] = {
                "status": status, "completed_at": utcnow(),
            }
            if error is not None:
                fields["error"] = error
            if notes.get(status):
                fields["progress_note"] = notes[status]
            database.update_benchmark_run(run_id, **fields)
        _benchmark_jobs.pop(run_id, None)
    return status


def start_benchmark_run(
    *,
    model_id: str,
    task_key: str,
    limit: int | None = None,
    num_fewshot: int | None = None,
    allow_code_execution: bool = False,
) -> int:
    """Create a benchmark run and execute it in the background.

    Refuses cloud models outright rather than failing deep inside lm-eval:
    benchmarking one would need a LiteLLM proxy for the harness to talk to,
    which does not exist yet, and a clear no beats a confusing stack trace.
    """
    task = harness.get_task(task_key)
    model_spec = parse_model_id(model_id)

    if model_spec.kind != "local":
        raise ValueError(
            "Benchmarks currently run against local runtimes only. Reaching a cloud "
            "model would need a LiteLLM proxy for lm-eval to talk to."
        )
    if task.needs_code_execution and not allow_code_execution:
        raise ValueError(
            f"{task.label} is scored by executing code the model wrote. Tick the "
            f"code-execution box to run it."
        )

    run_id = database.create_benchmark_run(
        model_name=model_spec.label,
        model_id=model_spec.id,
        task=task.key,
        task_label=task.label,
        runtime=model_spec.runtime,
        item_limit=limit,
        num_fewshot=num_fewshot if num_fewshot is not None else task.num_fewshot,
    )

    job = Job()
    _benchmark_jobs[run_id] = job
    job.task = asyncio.create_task(
        _benchmark_wrapper(run_id, task, model_spec, limit, num_fewshot, job.cancel),
        name=f"litmusllm-benchmark-{run_id}",
    )
    return run_id
