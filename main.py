"""LitmusLLM -- a local-first LLM evaluation dashboard.

FastAPI application: JSON API under /api, server-rendered pages everywhere
else, and small HTMX fragments under /ui for live-updating regions.

Run it with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import benchmarks
import database
import datasets as datasets_mod
import eval_runner
import harness
import metrics_catalog
import model_registry
import perf
import runtimes
import scorecard as scorecard_mod
import scoring
from config import DEFAULT_JUDGE, PROJECT_ROOT, UPLOAD_DIR
from database import utcnow
from runtimes import RuntimeUnavailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("litmusllm")


# Seeding the database is not a one-time startup chore: `database` re-runs
# these whenever it has to rebuild the schema, so a database that was deleted
# out from under a running app comes back with its built-in dataset and
# published benchmark rows intact rather than merely stopping the 500s.
# Both are idempotent, so registering them here and calling them in `lifespan`
# below costs nothing on a normal boot.
database.register_repair_hook(datasets_mod.ensure_builtin_dataset)
database.register_repair_hook(benchmarks.seed_reference_data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare storage on boot and cancel in-flight runs on shutdown."""
    config.ensure_dirs()
    database.init_db()
    reaped = database.reap_interrupted_runs()
    if reaped:
        log.warning("Marked %s interrupted run(s) as failed after restart.", reaped)
    dataset_id = datasets_mod.ensure_builtin_dataset()
    seeded = benchmarks.seed_reference_data()
    if seeded:
        log.info("Seeded %s published benchmark reference rows.", seeded)
    log.info("LitmusLLM ready -- built-in dataset id=%s, Ollama at %s",
             dataset_id, config.OLLAMA_HOST)
    yield
    await eval_runner.shutdown()


app = FastAPI(title="LitmusLLM", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def fmt_score(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def fmt_pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.0f}%"


templates.env.filters["score"] = fmt_score
templates.env.filters["pct"] = fmt_pct
templates.env.globals["metric_by_label"] = lambda label: next(
    (m for m in metrics_catalog.METRICS if m.label == label), None
)

# The Litmus Score and its comparability tag are derived, never stored: they
# fall straight out of the summary rows, so there is no second copy to drift
# out of step with the results table. See scoring.py for what the number means
# -- and, more importantly, what it does not.
templates.env.globals["litmus_score"] = scoring.composite
templates.env.globals["comparability_tag"] = lambda run: scoring.comparability_key(
    run.get("dataset_id"), run.get("metrics"), run.get("judge_model")
)
templates.env.globals["comparability_note"] = scoring.comparability_note
# Which local runtime is "the" one, so the compact warning knows whose absence
# is worth interrupting the user about.
templates.env.globals["default_runtime"] = config.DEFAULT_RUNTIME
# Speed measurements are read back the same way the Litmus Score is:
# derived from the stored row at render time, never a second copy.
templates.env.globals["run_perf"] = perf.from_run


def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    """Render a template with the globals every page needs."""
    ctx.setdefault("active", "")
    return templates.TemplateResponse(request, name, ctx)


async def model_options() -> dict[str, Any]:
    """Local + cloud models for the pickers, plus per-runtime health.

    Nothing here raises. With three local runtimes, one being down is the
    normal case, not a failure -- the picker still needs to render the other
    two, and the status strip is what explains the gap.
    """
    local: list[dict[str, Any]] = []
    statuses: list[runtimes.RuntimeStatus] = []
    runtime_error: str | None = None
    try:
        local, statuses = await runtimes.all_local_models()
    except Exception as exc:  # noqa: BLE001 - the page is useful without any runtime
        runtime_error = f"Could not list local models: {exc}"

    return {
        "local_models": local,
        "chat_models": [m for m in local if m["chat_capable"]],
        "flagship_models": model_registry.flagship_model_cards(),
        "runtime_statuses": [st.as_dict() for st in statuses],
        "runtime_error": runtime_error,
        # Retained so any template still checking it keeps working; it now
        # reports the default runtime specifically, not "local" in general.
        "ollama_error": next(
            (st.error for st in statuses if st.key == config.DEFAULT_RUNTIME and st.error),
            runtime_error,
        ),
    }


def _group_cases(run_id: int) -> list[dict[str, Any]]:
    """Collapse the flat results table into one entry per test case.

    Results are stored one row per (case, metric); the detail view wants one
    row per case with its metric scores nested underneath.
    """
    cases: dict[int, dict[str, Any]] = {}
    for r in database.get_run_results(run_id):
        case = cases.setdefault(int(r["case_index"]), {
            "index": int(r["case_index"]),
            "input": r["test_case_input"],
            "actual_output": r["actual_output"],
            "metrics": [],
        })
        # A skipped/errored row has no output; take it from whichever row has one.
        if r["actual_output"] and not case["actual_output"]:
            case["actual_output"] = r["actual_output"]
        case["metrics"].append(r)
    return [cases[k] for k in sorted(cases)]


def _perf_payload(run: dict[str, Any]) -> dict[str, Any] | None:
    """Measured speed/token/cost figures for a run, or None if uninstrumented.

    Kept beside `_score_payload` and used by every endpoint that reports a run,
    for the same reason: three callers formatting the same measurements three
    different ways is how an export ends up disagreeing with the page it was
    exported from.
    """
    measured = perf.from_run(run)
    if measured is None or not measured.measured:
        return None
    return {
        "calls": measured.calls,
        "measured_calls": measured.measured_calls,
        "ttft_ms": {"p50": measured.ttft_p50, "p90": measured.ttft_p90},
        "output_tokens_per_second": {"p50": measured.tps_p50, "p90": measured.tps_p90},
        "tokens": {
            "prompt": measured.prompt_tokens,
            "completion": measured.completion_tokens,
        },
        "cost_usd": measured.total_cost_usd,
        "runtime": run.get("runtime"),
        "quantization": run.get("quantization"),
        "note": (
            "Measured on the model under test only; the judge is not timed. "
            "Percentiles come from this run alone, not a rolling window. Local "
            "throughput is bound by this machine's hardware and is not comparable "
            "to a hosted provider's published figures."
        ),
    }


def _score_payload(run: dict[str, Any], summary: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The Litmus Score in JSON form, or None when nothing has scored yet.

    Kept in one place so the list endpoint, the detail endpoint and the export
    cannot drift into reporting the same run three slightly different ways.
    """
    comp = scoring.composite(summary)
    if comp is None:
        return None
    return {
        "score": comp.score,
        "scale": "0-100, higher is better",
        "metrics_used": comp.metric_count,
        "cases": comp.cases,
        "thin_evidence": comp.thin,
        "caveat": comp.caveat or None,
        # Two scores are rankable against each other only when these match.
        "comparability_key": scoring.comparability_key(
            run.get("dataset_id"), run.get("metrics"), run.get("judge_model")
        ),
        "note": (
            "Mean of this run's metric averages, each oriented so higher is "
            "better. Not comparable to published benchmark indices."
        ),
    }


def _form_list(raw: list[str] | None) -> list[str]:
    """Normalise a repeated form field into a clean list."""
    return [v for v in (raw or []) if v]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def page_index(request: Request):
    """Run configuration: pick a model, metrics and dataset, then start."""
    options = await model_options()
    dataset_list = database.list_datasets()
    return render(
        request, "index.html", active="run",
        categories=metrics_catalog.categories(),
        default_metrics=list(metrics_catalog.DEFAULT_METRIC_KEYS),
        datasets=dataset_list,
        dataset_caps={
            d["id"]: datasets_mod.dataset_capabilities(
                database.get_dataset_rows(int(d["id"]))
            )
            for d in dataset_list
        },
        default_judge=DEFAULT_JUDGE,
        recent_runs=database.list_eval_runs(limit=5, standalone_only=True),
        **options,
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return render(
        request, "dashboard.html", active="dashboard",
        runs=database.list_eval_runs(limit=200),
        comparisons=database.list_comparison_runs(limit=50),
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def page_run_detail(request: Request, run_id: int):
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    return render(
        request, "run_detail.html", active="dashboard",
        run=run, summary=database.get_run_summary(run_id),
        cases=_group_cases(run_id),
        is_running=eval_runner.is_running(run_id),
    )


@app.get("/models", response_class=HTMLResponse)
async def page_models(request: Request):
    options = await model_options()
    return render(request, "models.html", active="models", **options)


@app.get("/compare", response_class=HTMLResponse)
async def page_compare(request: Request):
    options = await model_options()
    return render(
        request, "compare.html", active="compare",
        categories=metrics_catalog.categories(),
        default_metrics=list(metrics_catalog.DEFAULT_METRIC_KEYS),
        datasets=database.list_datasets(),
        default_judge=DEFAULT_JUDGE,
        comparisons=database.list_comparison_runs(limit=20),
        **options,
    )


@app.get("/compare/{comparison_id}", response_class=HTMLResponse)
async def page_comparison_detail(request: Request, comparison_id: int):
    cmp_run = database.get_comparison_run(comparison_id)
    if not cmp_run:
        raise HTTPException(404, "No such comparison.")
    # Families of the models in this comparison, so the reference panel can
    # highlight their published relatives.
    families = []
    for child in database.get_comparison_child_runs(comparison_id):
        family = benchmarks.family_for_local_model(child["model_name"])
        if family and family not in families:
            families.append(family)
    return render(
        request, "compare_detail.html", active="compare",
        comparison=cmp_run, **_comparison_view(comparison_id),
        **_reference_view(families),
    )


def _comparison_view(comparison_id: int) -> dict[str, Any]:
    """Shape comparison results into the matrix the table and chart consume."""
    cmp_run = database.get_comparison_run(comparison_id)
    results = database.get_comparison_results(comparison_id)
    children = database.get_comparison_child_runs(comparison_id)

    metric_labels = [metrics_catalog.get_metric(k).label for k in (cmp_run["metrics"] if cmp_run else [])]
    model_labels: list[str] = []
    for child in children:
        if child["model_name"] not in model_labels:
            model_labels.append(child["model_name"])

    # cell[model][metric] -> the aggregate row, or None if not scored yet
    cells: dict[str, dict[str, Any]] = {m: {} for m in model_labels}
    for r in results:
        cells.setdefault(r["model_name"], {})[r["metric_name"]] = r

    # Best score per metric, so the table can highlight the leader.
    best: dict[str, float] = {}
    for label in metric_labels:
        spec = next((m for m in metrics_catalog.METRICS if m.label == label), None)
        scores = [
            c[label]["average_score"] for c in cells.values()
            if c.get(label) and c[label]["average_score"] is not None
        ]
        if scores:
            best[label] = max(scores) if (spec is None or spec.higher_is_better) else min(scores)

    status_by_model = {c["model_name"]: dict(c) for c in children}

    # Overall standings. Every model in a comparison ran the same dataset,
    # metrics and judge by construction, so their Litmus Scores are comparable
    # to each other without any further caveat -- this is the one place in the
    # app where ranking by the composite is unambiguously fair.
    composites: dict[str, Any] = {}
    for model, by_metric in cells.items():
        comp = scoring.composite(list(by_metric.values()))
        if comp is not None:
            composites[model] = comp
    overall_rank = {
        model: i + 1
        for i, model in enumerate(
            sorted(composites, key=lambda m: composites[m].score, reverse=True)
        )
    }

    return {
        "composites": composites,
        "overall_rank": overall_rank,
        "metric_labels": metric_labels,
        "model_labels": model_labels,
        "cells": cells,
        "best": best,
        "child_runs": children,
        "status_by_model": status_by_model,
        "chart_data": json.dumps({
            "labels": model_labels,
            "metrics": [
                {
                    "label": label,
                    "higher_is_better": bool(
                        next((m.higher_is_better for m in metrics_catalog.METRICS
                              if m.label == label), True)
                    ),
                    "scores": [
                        (cells.get(model, {}).get(label) or {}).get("average_score")
                        for model in model_labels
                    ],
                }
                for label in metric_labels
            ],
        }),
    }


# ---------------------------------------------------------------------------
# HTMX fragments
# ---------------------------------------------------------------------------

@app.get("/ui/run-progress/{run_id}", response_class=HTMLResponse)
async def ui_run_progress(request: Request, run_id: int):
    """Polled every 2s while a run is live."""
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    return render(
        request, "partials/run_progress.html",
        run=run, summary=database.get_run_summary(run_id),
        is_running=eval_runner.is_running(run_id),
    )


@app.get("/ui/runs-table", response_class=HTMLResponse)
async def ui_runs_table(request: Request):
    return render(request, "partials/runs_table.html", runs=database.list_eval_runs(limit=200))


@app.get("/ui/run-cases/{run_id}", response_class=HTMLResponse)
async def ui_run_cases(request: Request, run_id: int):
    """Polled while a run is live so per-case rows appear as they are scored."""
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    return render(request, "partials/run_cases.html", cases=_group_cases(run_id), run=run)


@app.get("/ui/comparison/{comparison_id}", response_class=HTMLResponse)
async def ui_comparison(request: Request, comparison_id: int):
    cmp_run = database.get_comparison_run(comparison_id)
    if not cmp_run:
        raise HTTPException(404, "No such comparison.")
    return render(request, "partials/comparison_results.html",
                  comparison=cmp_run, **_comparison_view(comparison_id))


@app.get("/ui/models-grid", response_class=HTMLResponse)
async def ui_models_grid(request: Request):
    options = await model_options()
    return render(request, "partials/models_grid.html", **options)


# ---------------------------------------------------------------------------
# JSON API -- models & metrics
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def api_models(runtime: str | None = None):
    """Local models across every runtime, or just the one named.

    `?runtime=llamacpp` narrows it, and in that form a runtime being down is a
    503 -- you asked about that backend specifically. Unfiltered, a single
    backend being down is not an error, so the other runtimes' models still
    come back and the per-runtime detail is in /api/runtimes.
    """
    if runtime is not None and runtime not in runtimes.RUNTIMES:
        raise HTTPException(
            404, f"Unknown runtime '{runtime}'. Known: {', '.join(runtimes.RUNTIME_ORDER)}"
        )
    try:
        return {"models": await model_registry.list_local_models(runtime=runtime)}
    except RuntimeUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/flagship-models")
async def api_flagship_models():
    """The curated cloud model list, annotated with API-key availability."""
    return {"models": model_registry.flagship_model_cards()}


@app.get("/api/metrics")
async def api_metrics():
    return {"metrics": metrics_catalog.as_dicts()}


@app.get("/api/health")
async def api_health():
    statuses = await runtimes.probe_all()
    return {
        "ok": True,
        "runtimes": [st.as_dict() for st in statuses],
        "default_runtime": config.DEFAULT_RUNTIME,
        "default_judge": DEFAULT_JUDGE,
        "db": str(config.DB_PATH),
        # Kept so existing health checks and scripts don't break.
        "ollama_host": config.OLLAMA_HOST,
        "ollama_up": next((st.up for st in statuses if st.key == "ollama"), False),
    }


@app.get("/api/runtimes")
async def api_runtimes():
    """Every local runtime: reachable, installed, how many models, how to fix."""
    return {"runtimes": [st.as_dict() for st in await runtimes.probe_all()]}


@app.post("/api/runtimes/start")
async def api_start_runtime(runtime: str = Form(...)):
    """Start a local runtime on demand.

    Runs are already self-starting, so this exists for the case where you want
    the runtime up *before* committing to a run -- to see what it is serving,
    or to confirm the auto-start works at all rather than discovering it two
    minutes into an evaluation.
    """
    key = runtime.strip()
    if key not in runtimes.RUNTIMES:
        raise HTTPException(
            404, f"Unknown runtime '{key}'. Known: {', '.join(runtimes.RUNTIME_ORDER)}"
        )
    rt = runtimes.get(key)
    try:
        await runtimes.ensure_running(rt)
    except RuntimeUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc
    return (await runtimes.probe(rt)).as_dict()


@app.post("/api/models/pull")
async def api_pull_model(model: str = Form(...)):
    """Pull a model. Blocks until the download finishes -- large models are slow."""
    try:
        return await model_registry.pull_model(model.strip())
    except (RuntimeUnavailable, RuntimeError) as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/models/delete")
async def api_delete_model(model: str = Form(...)):
    try:
        return await model_registry.delete_model(model.strip())
    except (RuntimeUnavailable, RuntimeError) as exc:
        raise HTTPException(502, str(exc)) from exc


# ---------------------------------------------------------------------------
# JSON API -- datasets
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
async def api_datasets():
    out = []
    for d in database.list_datasets():
        rows = database.get_dataset_rows(int(d["id"]))
        out.append({**d, "capabilities": datasets_mod.dataset_capabilities(rows)})
    return {"datasets": out}


@app.post("/api/datasets/upload")
async def api_upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(""),
    redirect: str = Form(""),
):
    """Accept a CSV of test cases (columns: input, expected_output, context, ...)."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        rows = datasets_mod.parse_csv(raw)
    except datasets_mod.CSVFormatError as exc:
        raise HTTPException(400, str(exc)) from exc

    label = (name or "").strip() or (file.filename or "dataset.csv").rsplit(".", 1)[0]
    stored = UPLOAD_DIR / f"{utcnow().replace(':', '-')}-{(file.filename or 'upload.csv')}"
    stored.write_bytes(raw)

    dataset_id = database.create_dataset(label, rows, csv_path=str(stored))
    if redirect:
        return RedirectResponse(f"{redirect}?dataset={dataset_id}", status_code=303)
    return {"id": dataset_id, "name": label, "rows": len(rows)}


@app.get("/api/datasets/{dataset_id}")
async def api_dataset_detail(dataset_id: int):
    dataset = database.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "No such dataset.")
    rows = database.get_dataset_rows(dataset_id)
    return {**dataset, "capabilities": datasets_mod.dataset_capabilities(rows), "rows": rows}


@app.get("/api/datasets/{dataset_id}/export.csv")
async def api_dataset_export(dataset_id: int):
    dataset = database.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "No such dataset.")
    body = datasets_mod.rows_to_csv(database.get_dataset_rows(dataset_id))
    return PlainTextResponse(
        body, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="dataset-{dataset_id}.csv"'},
    )


@app.post("/api/datasets/{dataset_id}/delete")
async def api_delete_dataset(dataset_id: int):
    dataset = database.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(404, "No such dataset.")
    if dataset["is_builtin"]:
        raise HTTPException(400, "The built-in dataset cannot be deleted.")
    database.delete_dataset(dataset_id)
    return RedirectResponse("/", status_code=303)


# ---------------------------------------------------------------------------
# JSON API -- eval runs
# ---------------------------------------------------------------------------

@app.post("/api/evals/start")
async def api_start_eval(
    request: Request,
    model: str = Form(...),
    dataset_id: int = Form(...),
    judge: str = Form(""),
):
    """Start an eval run. Returns JSON for API callers, redirects for the form."""
    form = await request.form()
    metric_keys = _form_list(form.getlist("metrics"))
    if not metric_keys:
        raise HTTPException(400, "Select at least one metric.")
    try:
        run_id = eval_runner.start_eval_run(
            model_id=model,
            metric_keys=metric_keys,
            dataset_id=dataset_id,
            judge_model_id=judge or None,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(f"/runs/{run_id}", status_code=303)
    return {"id": run_id, "status": "queued"}


@app.post("/api/evals/{run_id}/stop")
async def api_stop_eval(request: Request, run_id: int):
    """Cooperative stop -- partial results are kept."""
    if not database.get_eval_run(run_id):
        raise HTTPException(404, "No such eval run.")
    stopped = eval_runner.request_stop(run_id)
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(f"/runs/{run_id}", status_code=303)
    return {"id": run_id, "stopping": stopped}


@app.get("/api/evals")
async def api_list_evals(limit: int = 200):
    runs = database.list_eval_runs(limit=limit)
    for run in runs:
        run["litmus_score"] = _score_payload(run, run.get("summary") or [])
        run["performance"] = _perf_payload(run)
    return {"runs": runs}


@app.get("/api/evals/{run_id}")
async def api_eval_detail(run_id: int):
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    summary = database.get_run_summary(run_id)
    return {
        **run,
        "is_running": eval_runner.is_running(run_id),
        "litmus_score": _score_payload(run, summary),
        "performance": _perf_payload(run),
        "summary": summary,
        "results": database.get_run_results(run_id),
    }


@app.post("/api/evals/{run_id}/delete")
async def api_delete_eval(run_id: int):
    if not database.get_eval_run(run_id):
        raise HTTPException(404, "No such eval run.")
    eval_runner.request_stop(run_id)
    database.delete_eval_run(run_id)
    return RedirectResponse("/dashboard", status_code=303)


# --- exports ---------------------------------------------------------------

@app.get("/api/evals/{run_id}/export.json")
async def api_export_json(run_id: int):
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    summary = database.get_run_summary(run_id)
    payload = {
        "run": run,
        "litmus_score": _score_payload(run, summary),
        "performance": _perf_payload(run),
        "summary": summary,
        "results": database.get_run_results(run_id),
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": f'attachment; filename="litmusllm-run-{run_id}.json"'},
    )


@app.get("/api/evals/{run_id}/export.csv")
async def api_export_csv(run_id: int):
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "case_index", "model", "metric", "score", "threshold",
        "passed", "status", "input", "actual_output", "reason",
    ])
    for r in database.get_run_results(run_id):
        writer.writerow([
            r["case_index"], run["model_name"], r["metric_name"], r["score"],
            r["threshold"], r["passed"], r["status"],
            r["test_case_input"], r["actual_output"], r["reason"],
        ])
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="litmusllm-run-{run_id}.csv"'},
    )


@app.get("/api/evals/{run_id}/export.md")
async def api_export_markdown(run_id: int):
    """A shareable Markdown report -- paste straight into a PR or a doc."""
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    summary = database.get_run_summary(run_id)
    results = database.get_run_results(run_id)

    lines = [
        f"# LitmusLLM report -- run #{run_id}",
        "",
        f"- **Model:** `{run['model_name']}` ({run['model_type']})",
        f"- **Served by:** {run.get('runtime') or 'unrecorded'}"
        + (f" at `{run['quantization']}`" if run.get("quantization") else ""),
        f"- **Judge:** `{run['judge_model']}`",
        f"- **Dataset:** {run['dataset_name']} ({run['progress_total']} cases)",
        f"- **Status:** {run['status']}",
        f"- **Started:** {run['started_at']}",
        f"- **Completed:** {run['completed_at'] or '--'}",
        "",
    ]

    comp = scoring.composite(summary)
    if comp is not None:
        lines += [
            f"## Litmus Score: {comp.score:.1f} / 100",
            "",
            f"Mean of {comp.metric_count} metric average(s) over {comp.cases} scored "
            "case(s), each oriented so that higher is better.",
            "",
            f"Comparable only with runs tagged `{scoring.comparability_key(run.get('dataset_id'), run.get('metrics'), run.get('judge_model'))}` "
            "-- same dataset, same metrics, same judge. This is **not** an "
            "Artificial Analysis Intelligence Index score and cannot be read "
            "against one.",
            "",
        ]
        if comp.thin:
            lines += [f"> Thin evidence: {comp.caveat}.", ""]

    measured = perf.from_run(run)
    if measured is not None and measured.measured:
        lines += [
            "## Speed & cost",
            "",
            "| Measure | p50 | p90 |",
            "| --- | ---: | ---: |",
            f"| Time to first token | {measured.ttft_label} | {measured.ttft_p90_label} |",
            f"| Output speed | {measured.tps_label} | "
            + (f"{measured.tps_p90:.1f} tok/s" if measured.tps_p90 is not None else "--")
            + " |",
            "",
            f"{measured.completion_tokens:,} output tokens from {measured.prompt_tokens:,} "
            f"input tokens over {measured.calls} generation(s). Cost: {measured.cost_label}.",
            "",
            "> Measured on the model under test only -- the judge is not timed. "
            + (
                "Local throughput measures this machine, not the model, and does not "
                "compare to a hosted provider's published speeds."
                if run["model_type"] == "local"
                else "First-token latency includes the network path from this machine, "
                     "and these percentiles come from one run rather than a rolling window."
            ),
            "",
        ]

    lines += [
        "## Scores",
        "",
        "| Metric | Avg score | Pass rate | Cases scored |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s in summary:
        spec = next((m for m in metrics_catalog.METRICS if m.label == s["metric_name"]), None)
        arrow = "" if spec is None else (" (higher is better)" if spec.higher_is_better
                                         else " (lower is better)")
        lines.append(
            f"| {s['metric_name']}{arrow} | {fmt_score(s['average_score'])} | "
            f"{fmt_pct(s['pass_rate'])} | {s['scored_cases']} |"
        )

    lines += ["", "## Per test case", ""]
    by_case: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        by_case.setdefault(int(r["case_index"]), []).append(r)
    for index in sorted(by_case):
        rows = by_case[index]
        lines.append(f"### Case {index + 1}")
        lines.append("")
        lines.append(f"**Input:** {rows[0]['test_case_input']}")
        lines.append("")
        output = next((r["actual_output"] for r in rows if r["actual_output"]), None)
        if output:
            lines += ["**Output:**", "", "> " + output.replace("\n", "\n> "), ""]
        for r in rows:
            verdict = {"scored": "PASS" if r["passed"] else "FAIL"}.get(r["status"], r["status"].upper())
            lines.append(f"- **{r['metric_name']}** -- {fmt_score(r['score'])} [{verdict}]")
            if r["reason"]:
                lines.append(f"  - {r['reason']}")
        lines.append("")

    return PlainTextResponse(
        "\n".join(lines), media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="litmusllm-run-{run_id}.md"'},
    )


# ---------------------------------------------------------------------------
# JSON API -- comparisons
# ---------------------------------------------------------------------------

@app.post("/api/compare")
async def api_start_comparison(
    request: Request,
    dataset_id: int = Form(...),
    judge: str = Form(""),
    mode: str = Form("sequential"),
    name: str = Form(""),
):
    form = await request.form()
    model_ids = _form_list(form.getlist("models"))
    metric_keys = _form_list(form.getlist("metrics"))
    try:
        comparison_id = eval_runner.start_comparison(
            model_ids=model_ids,
            metric_keys=metric_keys,
            dataset_id=dataset_id,
            judge_model_id=judge or None,
            mode=mode if mode in ("sequential", "parallel") else "sequential",
            name=name.strip() or None,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(f"/compare/{comparison_id}", status_code=303)
    return {"id": comparison_id, "status": "queued"}


@app.get("/api/compare/{comparison_id}")
async def api_comparison_detail(comparison_id: int):
    cmp_run = database.get_comparison_run(comparison_id)
    if not cmp_run:
        raise HTTPException(404, "No such comparison.")
    return {
        **cmp_run,
        "results": database.get_comparison_results(comparison_id),
        "child_runs": database.get_comparison_child_runs(comparison_id),
    }


@app.post("/api/compare/{comparison_id}/stop")
async def api_stop_comparison(request: Request, comparison_id: int):
    if not database.get_comparison_run(comparison_id):
        raise HTTPException(404, "No such comparison.")
    stopped = eval_runner.request_comparison_stop(comparison_id)
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(f"/compare/{comparison_id}", status_code=303)
    return {"id": comparison_id, "stopping": stopped}


@app.post("/api/compare/{comparison_id}/delete")
async def api_delete_comparison(comparison_id: int):
    if not database.get_comparison_run(comparison_id):
        raise HTTPException(404, "No such comparison.")
    eval_runner.request_comparison_stop(comparison_id)
    database.delete_comparison_run(comparison_id)
    return RedirectResponse("/compare", status_code=303)


@app.get("/api/compare/{comparison_id}/export.csv")
async def api_comparison_export(comparison_id: int):
    cmp_run = database.get_comparison_run(comparison_id)
    if not cmp_run:
        raise HTTPException(404, "No such comparison.")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["model", "metric", "average_score", "pass_rate", "scored_cases", "rank"])
    for r in database.get_comparison_results(comparison_id):
        writer.writerow([
            r["model_name"], r["metric_name"], r["average_score"],
            r["pass_rate"], r["scored_cases"], r["rank"],
        ])
    # The composite goes in the same file rather than a second endpoint, using
    # the pseudo-metric name "Litmus Score" so a spreadsheet filter on the
    # metric column separates it from the raw per-metric rows.
    view = _comparison_view(comparison_id)
    for model, comp in view["composites"].items():
        writer.writerow([
            model, "Litmus Score", comp.score, "", comp.cases,
            view["overall_rank"].get(model),
        ])
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="litmusllm-comparison-{comparison_id}.csv"'
        },
    )


@app.get("/api/compare/{comparison_id}/export.md")
async def api_comparison_markdown(comparison_id: int):
    cmp_run = database.get_comparison_run(comparison_id)
    if not cmp_run:
        raise HTTPException(404, "No such comparison.")
    view = _comparison_view(comparison_id)
    lines = [
        f"# LitmusLLM comparison -- #{comparison_id}",
        "",
        f"- **Dataset:** {cmp_run['dataset_name']}",
        f"- **Judge:** `{cmp_run['judge_model']}`",
        f"- **Status:** {cmp_run['status']}",
        "",
        "| Model | Litmus Score | " + " | ".join(view["metric_labels"]) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in view["metric_labels"]) + " |",
    ]
    for model in view["model_labels"]:
        comp = view["composites"].get(model)
        rank = view["overall_rank"].get(model)
        overall = "--" if comp is None else f"**{comp.score:.1f}** (#{rank})"
        cells = [overall]
        for label in view["metric_labels"]:
            cell = view["cells"].get(model, {}).get(label)
            if not cell or cell["average_score"] is None:
                cells.append("--")
            else:
                marker = " **(best)**" if view["best"].get(label) == cell["average_score"] else ""
                cells.append(f"{fmt_score(cell['average_score'])} (#{cell['rank']}){marker}")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Higher is better except for Toxicity, Bias and Hallucination.",
        "",
        "The Litmus Score is the mean of each model's metric averages, oriented "
        "so higher is always better and scaled 0-100. Every model here ran the "
        "same dataset, metrics and judge, so the scores rank fairly against each "
        "other -- but not against published benchmark indices, or against runs "
        "configured differently.",
        "",
    ]
    return PlainTextResponse(
        "\n".join(lines), media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="litmusllm-comparison-{comparison_id}.md"'
        },
    )



# ---------------------------------------------------------------------------
# Published benchmark reference
# ---------------------------------------------------------------------------
# Third-party figures, deliberately kept on their own page and in their own
# panel. See benchmarks.py for why these must never share an axis with a
# measured eval score.

def _reference_view(highlight_families: list[str] | None = None) -> dict[str, Any]:
    rows = database.list_benchmark_rows()
    families = sorted({r["family"] for r in rows if r["family"]})
    highlight = set(highlight_families or [])
    return {
        "reference_rows": rows,
        "reference_families": families,
        "highlight_families": highlight,
        "reference_frontier": [r for r in rows if r["kind"] == "frontier"],
        "reference_open": [r for r in rows if r["kind"] == "open_weight"],
        "reference_source": benchmarks.SEED_SOURCE,
        "reference_source_url": benchmarks.SEED_SOURCE_URL,
        "reference_retrieved": benchmarks.SEED_RETRIEVED,
        "reference_note": benchmarks.SEED_INDEX_NOTE,
        "reference_chart": json.dumps({
            "labels": [
                r["model_label"] + (f" ({r['variant']})" if r["variant"] else "")
                for r in rows[:24]
            ],
            "scores": [r["score"] for r in rows[:24]],
            "kinds": [r["kind"] for r in rows[:24]],
            "highlighted": [bool(r["family"] in highlight) for r in rows[:24]],
        }),
    }


async def _benchmark_context(model_id: str | None = None) -> dict[str, Any]:
    """Everything the benchmarks page needs, including what it *can't* run."""
    options = await model_options()
    chosen = model_id or (options["chat_models"][0]["id"] if options["chat_models"] else None)

    available = None
    if chosen:
        spec = model_registry.parse_model_id(chosen)
        if spec.kind == "local":
            available = await harness.availability(spec.runtime, spec.name)

    runs = database.list_benchmark_runs(limit=100)
    return {
        **options,
        "tasks": harness.TASKS,
        "selected_model": chosen,
        "availability": available.as_dict() if available else None,
        "harness_installed": harness.harness_installed() is not None,
        "runs": runs,
        # Without this the results table renders without its poller, so loading
        # the page while a benchmark is live would show a frozen progress note
        # until the user refreshed by hand.
        "any_running": any(eval_runner.benchmark_is_running(r["id"]) for r in runs),
    }


@app.get("/benchmarks", response_class=HTMLResponse)
async def page_benchmarks(request: Request, model: str | None = None):
    """Ground-truth benchmarks: accuracy against known answers, not a judge."""
    return render(
        request, "benchmarks.html", active="benchmarks",
        **await _benchmark_context(model),
    )


@app.get("/ui/benchmark-runs", response_class=HTMLResponse)
async def ui_benchmark_runs(request: Request):
    """Polled while a benchmark is live so progress and results appear."""
    runs = database.list_benchmark_runs(limit=100)
    return render(
        request, "partials/benchmark_runs.html",
        runs=runs,
        any_running=any(eval_runner.benchmark_is_running(r["id"]) for r in runs),
    )


@app.get("/ui/benchmark-picker", response_class=HTMLResponse)
async def ui_benchmark_picker(request: Request, model: str | None = None):
    """Re-rendered when the model changes: availability is model-specific."""
    return render(
        request, "partials/benchmark_picker.html", **await _benchmark_context(model)
    )


@app.post("/api/benchmarks")
async def api_start_benchmark(
    request: Request,
    model: str = Form(...),
    task: str = Form(...),
    limit: str = Form(""),
    allow_code_execution: str = Form(""),
):
    """Start one benchmark run in the background."""
    raw_limit = (limit or "").strip()
    try:
        item_limit = int(raw_limit) if raw_limit else None
    except ValueError:
        raise HTTPException(400, f"'{raw_limit}' is not a whole number of items.") from None
    if item_limit is not None and item_limit < 1:
        raise HTTPException(400, "The item limit must be at least 1.")

    try:
        run_id = eval_runner.start_benchmark_run(
            model_id=model.strip(),
            task_key=task.strip(),
            limit=item_limit,
            allow_code_execution=bool(allow_code_execution),
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc

    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse("/benchmarks", status_code=303)
    return {"id": run_id, "status": "queued"}


@app.post("/api/benchmarks/{run_id}/stop")
async def api_stop_benchmark(request: Request, run_id: int):
    """Kill the harness subprocess. Nothing partial is kept -- lm-eval only
    reports a score once every item is done."""
    if not database.get_benchmark_run(run_id):
        raise HTTPException(404, "No such benchmark run.")
    stopped = eval_runner.request_benchmark_stop(run_id)
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse("/benchmarks", status_code=303)
    return {"id": run_id, "stopping": stopped}


@app.post("/api/benchmarks/{run_id}/delete")
async def api_delete_benchmark(run_id: int):
    if not database.get_benchmark_run(run_id):
        raise HTTPException(404, "No such benchmark run.")
    eval_runner.request_benchmark_stop(run_id)
    database.delete_benchmark_run(run_id)
    return RedirectResponse("/benchmarks", status_code=303)


@app.get("/api/benchmarks")
async def api_list_benchmarks(limit: int = 200, model: str | None = None):
    return {"runs": database.list_benchmark_runs(limit=limit, model_id=model)}


@app.get("/api/benchmarks/tasks")
async def api_benchmark_tasks(model: str | None = None):
    """The task catalogue, annotated with what this model can actually run."""
    payload: dict[str, Any] = {
        "harness_installed": harness.harness_installed() is not None,
        "tasks": [
            {
                "key": t.key, "label": t.label, "blurb": t.blurb, "kind": t.kind,
                "metric": t.metric_label, "items": t.items,
                "default_limit": t.default_limit, "num_fewshot": t.num_fewshot,
                "needs_code_execution": t.needs_code_execution,
                "contamination": t.contamination, "source_url": t.source_url,
            }
            for t in harness.TASKS
        ],
    }
    if model:
        spec = model_registry.parse_model_id(model)
        if spec.kind == "local":
            payload["availability"] = (
                await harness.availability(spec.runtime, spec.name)
            ).as_dict()
    return payload


@app.get("/scorecard", response_class=HTMLResponse)
async def page_scorecard(request: Request, limit: int = 200):
    """Every run in one sortable table: quality against speed, memory and cost."""
    return render(
        request, "scorecard.html", active="scorecard",
        card=scorecard_mod.build(database.list_eval_runs(limit=limit)),
    )


@app.get("/api/scorecard.csv")
async def api_scorecard_csv(limit: int = 200):
    """The scorecard as CSV -- the artefact you actually rank models in."""
    card = scorecard_mod.build(database.list_eval_runs(limit=limit))
    return PlainTextResponse(
        scorecard_mod.to_csv(card),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="litmusllm-scorecard.csv"'},
    )


@app.get("/reference", response_class=HTMLResponse)
async def page_reference(request: Request):
    """Published third-party benchmark figures, for context only."""
    # Highlight the families the user actually runs locally.
    local_families: list[str] = []
    try:
        for m in await model_registry.list_local_models():
            family = benchmarks.family_for_local_model(m["name"])
            if family and family not in local_families:
                local_families.append(family)
    except Exception:  # noqa: BLE001 - the page is useful without Ollama
        pass
    return render(
        request, "reference.html", active="reference",
        local_families=local_families, **_reference_view(local_families),
    )


@app.get("/api/reference")
async def api_reference(family: str = ""):
    return {
        "source": benchmarks.SEED_SOURCE,
        "source_url": benchmarks.SEED_SOURCE_URL,
        "retrieved_at": benchmarks.SEED_RETRIEVED,
        "note": benchmarks.SEED_INDEX_NOTE,
        "rows": database.list_benchmark_rows(family or None),
    }


@app.post("/api/reference")
async def api_add_reference(
    model_label: str = Form(...),
    score: float = Form(...),
    index_name: str = Form(""),
    vendor: str = Form(""),
    family: str = Form(""),
    variant: str = Form(""),
    kind: str = Form("frontier"),
    source_url: str = Form(""),
    source_name: str = Form(""),
    retrieved_at: str = Form(""),
    notes: str = Form(""),
):
    """Add your own published figure. Provenance fields are strongly encouraged."""
    if not model_label.strip():
        raise HTTPException(400, "A model label is required.")
    database.insert_benchmark_rows([{
        "model_label": model_label.strip(),
        "variant": variant.strip(),
        "vendor": vendor.strip() or None,
        "family": family.strip() or None,
        "kind": kind if kind in ("frontier", "open_weight") else "frontier",
        "index_name": index_name.strip() or benchmarks.SEED_SOURCE,
        "score": score,
        "source_name": source_name.strip() or None,
        "source_url": source_url.strip() or None,
        "retrieved_at": retrieved_at.strip() or utcnow()[:10],
        "notes": notes.strip() or None,
        "is_seed": False,
    }])
    return RedirectResponse("/reference", status_code=303)


@app.post("/api/reference/{row_id}/delete")
async def api_delete_reference(row_id: int):
    database.delete_benchmark_row(row_id)
    return RedirectResponse("/reference", status_code=303)


@app.get("/api/reference/export.csv")
async def api_reference_export():
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "model_label", "variant", "vendor", "family", "kind",
        "index_name", "score", "source_name", "source_url", "retrieved_at", "notes",
    ])
    for r in database.list_benchmark_rows():
        writer.writerow([
            r["model_label"], r["variant"], r["vendor"], r["family"], r["kind"],
            r["index_name"], r["score"], r["source_name"], r["source_url"],
            r["retrieved_at"], r["notes"],
        ])
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="litmusllm-reference.csv"'},
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Render HTML errors for browsers, JSON for API clients."""
    wants_html = "text/html" in request.headers.get("accept", "")
    if wants_html and not request.url.path.startswith("/api/"):
        return templates.TemplateResponse(
            request, "error.html",
            {"status": exc.status_code, "detail": exc.detail, "active": ""},
            status_code=exc.status_code,
        )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
