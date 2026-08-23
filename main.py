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
import metrics_catalog
import model_registry
from config import DEFAULT_JUDGE, PROJECT_ROOT, UPLOAD_DIR
from database import utcnow
from model_registry import OllamaUnavailable

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


def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    """Render a template with the globals every page needs."""
    ctx.setdefault("active", "")
    return templates.TemplateResponse(request, name, ctx)


async def model_options() -> dict[str, Any]:
    """Local + cloud models for the pickers, plus any Ollama error to show."""
    local: list[dict[str, Any]] = []
    ollama_error: str | None = None
    try:
        local = await model_registry.list_local_models()
    except OllamaUnavailable as exc:
        ollama_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        ollama_error = f"Could not list Ollama models: {exc}"
    return {
        "local_models": local,
        "chat_models": [m for m in local if m["chat_capable"]],
        "flagship_models": model_registry.flagship_model_cards(),
        "ollama_error": ollama_error,
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

    return {
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
async def api_models():
    """Ollama models currently pulled, via GET {OLLAMA_HOST}/api/tags."""
    try:
        return {"models": await model_registry.list_local_models()}
    except OllamaUnavailable as exc:
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
    return {
        "ok": True,
        "ollama_host": config.OLLAMA_HOST,
        "ollama_up": await model_registry.ollama_is_up(),
        "default_judge": DEFAULT_JUDGE,
        "db": str(config.DB_PATH),
    }


@app.post("/api/models/pull")
async def api_pull_model(model: str = Form(...)):
    """Pull a model. Blocks until the download finishes -- large models are slow."""
    try:
        return await model_registry.pull_model(model.strip())
    except (OllamaUnavailable, RuntimeError) as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/models/delete")
async def api_delete_model(model: str = Form(...)):
    try:
        return await model_registry.delete_model(model.strip())
    except (OllamaUnavailable, RuntimeError) as exc:
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
    return {"runs": database.list_eval_runs(limit=limit)}


@app.get("/api/evals/{run_id}")
async def api_eval_detail(run_id: int):
    run = database.get_eval_run(run_id)
    if not run:
        raise HTTPException(404, "No such eval run.")
    return {
        **run,
        "is_running": eval_runner.is_running(run_id),
        "summary": database.get_run_summary(run_id),
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
    payload = {
        "run": run,
        "summary": database.get_run_summary(run_id),
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
        f"- **Judge:** `{run['judge_model']}`",
        f"- **Dataset:** {run['dataset_name']} ({run['progress_total']} cases)",
        f"- **Status:** {run['status']}",
        f"- **Started:** {run['started_at']}",
        f"- **Completed:** {run['completed_at'] or '--'}",
        "",
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
        "| Model | " + " | ".join(view["metric_labels"]) + " |",
        "| --- | " + " | ".join("---:" for _ in view["metric_labels"]) + " |",
    ]
    for model in view["model_labels"]:
        cells = []
        for label in view["metric_labels"]:
            cell = view["cells"].get(model, {}).get(label)
            if not cell or cell["average_score"] is None:
                cells.append("--")
            else:
                marker = " **(best)**" if view["best"].get(label) == cell["average_score"] else ""
                cells.append(f"{fmt_score(cell['average_score'])} (#{cell['rank']}){marker}")
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines += ["", "Higher is better except for Toxicity, Bias and Hallucination.", ""]
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
