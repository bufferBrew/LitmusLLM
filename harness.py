"""Ground-truth benchmarks: the other kind of evaluation.

Everything else in LitmusLLM asks a judge model for an opinion about your
prompts. This asks a fixed question set with known answers and counts how many
the model got right. They are not two settings of one dial -- they are
different measurements, and the app keeps them apart on purpose:

    Litmus Score      a judge's opinion of your prompts       0-100, your scale
    Benchmark score   accuracy against published answers      %, everyone's scale

The second one is what a leaderboard publishes, which is why it exists here at
all: it is the only number in this app that can be lined up against a figure
someone else measured.

## Why lm-evaluation-harness, and why in a subprocess

The harness is the one whose numbers people actually quote, so using it is
what makes a score comparable rather than merely plausible. It is driven as a
subprocess rather than imported, for three reasons that all matter:

  * It pulls torch. Importing that into the web process would add gigabytes of
    resident memory to an app whose whole point is running comfortably beside
    a local model that wants all the RAM it can get.
  * Cancellation becomes a kill rather than a cooperative flag threaded
    through someone else's library.
  * A crash in the harness fails one run instead of taking down the server.

## The capability that decides which tasks you can run

Benchmarks come in two shapes, and the difference is not cosmetic:

  * **generate_until** -- the model writes an answer and it's checked. Needs
    only chat completions, so this works everywhere.
  * **loglikelihood** -- the model scores each candidate answer and the
    highest wins. Needs token logprobs from the API.

**Ollama does not return logprobs** on either of its OpenAI-compatible
endpoints, which rules out the multiple-choice family (HellaSwag, ARC) there
entirely. llama.cpp and LM Studio do expose them. That is not a limitation
worth hiding behind a confusing error two minutes into a run, so runtimes are
probed up front and unavailable tasks are reported as unavailable.

## What these numbers are not

They will not reproduce a published leaderboard figure to the decimal, and
anyone who tells you their homegrown harness does is not checking. Prompt
templates, answer extraction and shot counts all move a score by points. The
realistic goal is the same ballpark and a consistent ranking -- and the way to
confirm you got there is `calibration_hint`, below: run a model whose public
score you already know and see whether you land inside its confidence
interval.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import runtimes
from config import PROJECT_ROOT

#: Where lm-eval writes its result JSON. Under data/ so it inherits the
#: gitignore that already covers everything else the app generates.
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmarks"

#: 95% interval from a standard error, which is what lm-eval reports.
Z_95 = 1.96


class HarnessUnavailable(RuntimeError):
    """lm-evaluation-harness isn't installed, or the task can't run here."""


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One benchmark, and everything needed to run and read it.

    `metric_key` is lm-eval's own dotted name for the figure worth reporting,
    which is rarely the only one it emits. GSM8K reports both a lenient and a
    strict extraction; the strict one is what leaderboards quote, so taking
    whichever appears first would quietly report a different number than the
    one you're comparing against.
    """
    key: str                  # lm-eval task name
    label: str
    blurb: str
    kind: str                 # 'generate' | 'loglikelihood'
    metric_key: str           # e.g. 'exact_match,strict-match'
    metric_label: str
    items: int                # full size, so a limit can be shown as a fraction
    source_url: str
    default_limit: int = 50
    num_fewshot: int | None = None
    needs_code_execution: bool = False
    contamination: str = ""

    @property
    def needs_logprobs(self) -> bool:
        return self.kind == "loglikelihood"


TASKS: tuple[Task, ...] = (
    Task(
        key="gsm8k",
        label="GSM8K",
        blurb="Grade-school maths word problems needing several steps of arithmetic.",
        kind="generate",
        metric_key="exact_match,strict-match",
        metric_label="exact match",
        items=1319,
        source_url="https://huggingface.co/datasets/openai/gsm8k",
        num_fewshot=5,
        contamination="Widely used in training data. Treat a very high score with suspicion.",
    ),
    Task(
        key="ifeval",
        label="IFEval",
        blurb=(
            "Instructions with checkable constraints -- 'answer in exactly three "
            "bullets', 'use no commas'. Scored by a program, with no judge involved."
        ),
        kind="generate",
        metric_key="prompt_level_strict_acc,none",
        metric_label="prompt-level strict accuracy",
        items=541,
        source_url="https://huggingface.co/datasets/google/IFEval",
        num_fewshot=0,
        contamination="Less contaminated than the older sets, since constraints are generated.",
    ),
    Task(
        key="hellaswag",
        label="HellaSwag",
        blurb="Pick the plausible ending to an everyday scenario. Commonsense, four-way choice.",
        kind="loglikelihood",
        metric_key="acc_norm,none",
        metric_label="normalised accuracy",
        items=10042,
        source_url="https://huggingface.co/datasets/Rowan/hellaswag",
        num_fewshot=10,
        contamination="Heavily contaminated. Useful for ranking, weak as an absolute claim.",
    ),
    Task(
        key="arc_challenge",
        label="ARC Challenge",
        blurb="Grade-school science questions selected because retrieval alone fails them.",
        kind="loglikelihood",
        metric_key="acc_norm,none",
        metric_label="normalised accuracy",
        items=1172,
        source_url="https://huggingface.co/datasets/allenai/ai2_arc",
        num_fewshot=25,
        contamination="Long-standing public set; assume it is in most training corpora.",
    ),
    Task(
        key="humaneval",
        label="HumanEval",
        blurb="Complete a Python function so that its hidden unit tests pass. pass@1.",
        kind="generate",
        metric_key="pass@1,create_test",
        metric_label="pass@1",
        items=164,
        source_url="https://huggingface.co/datasets/openai/openai_humaneval",
        default_limit=164,
        num_fewshot=0,
        needs_code_execution=True,
        contamination="Public since 2021 and almost certainly memorised in part.",
    ),
)

_BY_KEY = {t.key: t for t in TASKS}


def get_task(key: str) -> Task:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown benchmark '{key}'. Known: {', '.join(t.key for t in TASKS)}"
        ) from exc


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def harness_installed() -> str | None:
    """Path to the `lm_eval` executable, or None if it isn't installed.

    Looked up beside the running interpreter first: the app usually lives in a
    virtualenv, and a stray system-wide lm_eval would be running against a
    different set of packages than the one we were installed into.
    """
    import sys

    candidate = Path(sys.executable).parent / "lm_eval"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("lm_eval")


#: Cached per runtime, not per model. Whether logprobs come back is a property
#: of the server and its OpenAI shim -- Ollama withholds them for everything it
#: serves -- so probing once per runtime is both correct and the difference
#: between opening this page instantly and loading a model to answer a question
#: whose answer never varies.
_LOGPROB_SUPPORT: dict[str, bool] = {}


def forget_logprob_support() -> None:
    """Drop the cache, e.g. after a runtime is restarted on a different build."""
    _LOGPROB_SUPPORT.clear()


async def supports_logprobs(runtime_key: str, model_name: str) -> bool:
    """Whether a runtime will return token logprobs for this model.

    Probed with a real one-token request rather than assumed from the runtime
    name, because it is a property of the server build and its OpenAI shim, not
    of the project's reputation. Ollama says no; llama.cpp and LM Studio
    generally say yes, but 'generally' is not something to stake a twenty-minute
    benchmark run on.
    """
    import httpx

    cached = _LOGPROB_SUPPORT.get(runtime_key)
    if cached is not None:
        return cached

    try:
        rt = runtimes.get(runtime_key)
    except KeyError:
        return False

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{rt.api_base}/completions",
                json={
                    "model": model_name,
                    "prompt": "1 + 1 =",
                    "max_tokens": 1,
                    "logprobs": 1,
                    "temperature": 0,
                },
            )
            if resp.status_code >= 400:
                return False
            choices = resp.json().get("choices") or []
    except Exception:  # noqa: BLE001 - any failure means "can't rely on it"
        return False

    supported = bool(choices and choices[0].get("logprobs"))
    _LOGPROB_SUPPORT[runtime_key] = supported
    return supported


@dataclass
class Availability:
    """Which benchmarks can actually run for one model, and why not otherwise."""
    harness_path: str | None = None
    logprobs: bool = False
    runnable: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)

    @property
    def installed(self) -> bool:
        return self.harness_path is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "logprobs": self.logprobs,
            "runnable": self.runnable,
            "blocked": self.blocked,
        }


async def availability(
    runtime_key: str | None, model_name: str, *, allow_code_execution: bool = False
) -> Availability:
    """Work out which tasks this model can be benchmarked on, before starting."""
    result = Availability(harness_path=harness_installed())

    if not result.installed:
        reason = (
            "lm-evaluation-harness is not installed. "
            "Run `pip install 'lm-eval[api]'` in the app's environment."
        )
        result.blocked = {t.key: reason for t in TASKS}
        return result

    # Cloud models reach an endpoint we don't control; assume the standard
    # OpenAI behaviour, which does expose logprobs on completions.
    result.logprobs = (
        await supports_logprobs(runtime_key, model_name) if runtime_key else True
    )

    for task in TASKS:
        if task.needs_logprobs and not result.logprobs:
            result.blocked[task.key] = (
                f"{task.label} scores candidate answers by log-probability, which this "
                f"endpoint does not return. Ollama never does; try the same weights "
                f"under llama.cpp or LM Studio."
            )
        elif task.needs_code_execution and not allow_code_execution:
            result.blocked[task.key] = (
                f"{task.label} scores by executing code the model wrote. That is off "
                f"unless you turn it on explicitly."
            )
        else:
            result.runnable.append(task.key)
    return result


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    """One benchmark score, with the uncertainty around it.

    The interval is not decoration. At the sample sizes anyone actually runs
    locally, two models a few points apart are usually tied, and a table that
    reports 61.2 against 58.9 without saying so invites a conclusion the data
    does not support.
    """
    task_key: str
    metric_key: str
    score: float                  # 0-1
    stderr: float | None
    samples: int
    model: str
    limit: int | None
    num_fewshot: int | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        return round(self.score * 100, 1)

    @property
    def margin(self) -> float | None:
        """Half-width of the 95% interval, in percentage points."""
        if self.stderr is None:
            return None
        return round(self.stderr * Z_95 * 100, 1)

    @property
    def interval_label(self) -> str:
        margin = self.margin
        if margin is None:
            return f"{self.percent}%"
        return f"{self.percent}% ± {margin}"

    def ties_with(self, other: "Result") -> bool:
        """True when the two intervals overlap, i.e. the gap isn't significant.

        Deliberately the conservative test. Overlapping intervals definitely
        means 'not distinguishable'; non-overlap is a slightly stronger claim
        than a formal test would require, which is the direction to err in.
        """
        a, b = self.margin, other.margin
        if a is None or b is None:
            return False
        return abs(self.percent - other.percent) <= (a + b)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_key,
            "metric": self.metric_key,
            "score_percent": self.percent,
            "stderr": self.stderr,
            "margin_95": self.margin,
            "samples": self.samples,
            "limit": self.limit,
            "num_fewshot": self.num_fewshot,
        }


def parse_results(payload: dict[str, Any], task: Task) -> Result:
    """Pull one task's headline figure out of lm-eval's results JSON."""
    results = payload.get("results") or {}
    block = results.get(task.key)
    if block is None and len(results) == 1:
        # Some task groups report under a different key than the one requested.
        block = next(iter(results.values()))
    if block is None:
        raise HarnessUnavailable(
            f"lm-eval produced no results for '{task.key}'. It reported: "
            f"{', '.join(results) or 'nothing'}."
        )

    score = block.get(task.metric_key)
    metric_key = task.metric_key
    if score is None:
        # Fall back to any metric that isn't a stderr, but say which one was
        # used -- silently reporting a different metric is how two runs of the
        # "same" benchmark end up not comparable.
        for name, value in block.items():
            if not name.endswith("_stderr,none") and isinstance(value, (int, float)):
                score, metric_key = value, name
                break
    if score is None:
        raise HarnessUnavailable(
            f"No numeric metric in lm-eval's output for '{task.key}': {sorted(block)}"
        )

    stderr_key = metric_key.replace(",", "_stderr,", 1) if "," in metric_key else f"{metric_key}_stderr"
    stderr = block.get(stderr_key)
    if not isinstance(stderr, (int, float)):
        stderr = next(
            (v for k, v in block.items() if "stderr" in k and isinstance(v, (int, float))),
            None,
        )

    counts = payload.get("n-samples") or {}
    entry = counts.get(task.key) or next(iter(counts.values()), {}) or {}
    samples = int(entry.get("effective") or entry.get("original") or 0)

    configs = payload.get("configs") or {}
    config = configs.get(task.key) or next(iter(configs.values()), {}) or {}

    return Result(
        task_key=task.key,
        metric_key=metric_key,
        score=float(score),
        stderr=float(stderr) if isinstance(stderr, (int, float)) else None,
        samples=samples,
        model=str((payload.get("config") or {}).get("model") or ""),
        limit=(payload.get("config") or {}).get("limit"),
        num_fewshot=config.get("num_fewshot"),
        raw=block,
    )


def calibration_hint(task: Task, result: Result) -> str:
    """What to check before believing a number.

    Every benchmark integration is wrong until proven otherwise, and the proof
    is cheap: score a model whose published figure you already know. Landing
    inside its interval says the harness is wired correctly. Landing eight
    points low says the extraction is broken, not that the model is worse --
    and those two look identical if you never run the check.
    """
    parts = [
        f"Scored {result.interval_label} on {result.samples} of {task.items} items."
    ]
    if result.limit and result.samples < task.items:
        parts.append(
            "That is a subset, so the interval is wider than a published figure "
            "measured on the full set."
        )
    if result.margin and result.margin >= 5:
        parts.append(
            f"±{result.margin} points is a wide interval -- only differences bigger "
            f"than that mean anything. Raise the item limit to narrow it."
        )
    if task.contamination:
        parts.append(task.contamination)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
# Local runtimes only, for now. Benchmarking a cloud model would mean either
# hitting each provider's endpoint directly or standing up a LiteLLM proxy for
# lm-eval to talk to -- separate plumbing, and not something to half-do.

def _model_args(task: Task, base_url: str, model_name: str, concurrency: int) -> str:
    """Build lm-eval's --model_args string.

    `tokenized_requests=False` because none of these servers expose the
    tokenizer lm-eval would otherwise ask for; without it every request fails
    on a length check it cannot perform.
    """
    endpoint = "completions" if task.needs_logprobs else "chat/completions"
    parts = [
        f"model={model_name}",
        f"base_url={base_url}/{endpoint}",
        f"num_concurrent={concurrency}",
        "tokenized_requests=False",
        "max_retries=3",
    ]
    return ",".join(parts)


def build_command(
    task: Task,
    *,
    executable: str,
    base_url: str,
    model_name: str,
    output_dir: Path,
    limit: int | None,
    num_fewshot: int | None = None,
    concurrency: int = 1,
    seed: int = 1234,
) -> list[str]:
    """The exact argv for one benchmark run.

    Separated from execution so it can be shown to the user and re-run by hand.
    A benchmark result nobody can reproduce outside the app is worth very
    little, and the command is the reproduction recipe.
    """
    argv = [
        executable,
        "--model", "local-completions" if task.needs_logprobs else "local-chat-completions",
        "--model_args", _model_args(task, base_url, model_name, concurrency),
        "--tasks", task.key,
        "--output_path", str(output_dir),
        "--seed", str(seed),
    ]
    if limit:
        argv += ["--limit", str(limit)]

    shots = task.num_fewshot if num_fewshot is None else num_fewshot
    if shots is not None:
        argv += ["--num_fewshot", str(shots)]

    if not task.needs_logprobs:
        # Chat endpoints need the template applied or lm-eval hands them a raw
        # string and the request is rejected outright. Worth knowing that this
        # also makes the prompt differ from the raw-completion format most
        # published figures use -- one of several reasons scores drift a little.
        argv.append("--apply_chat_template")
    if task.needs_code_execution:
        argv.append("--confirm_run_unsafe_code")
    return argv


def _environment(task: Task) -> dict[str, str]:
    env = dict(os.environ)
    env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    # This project has its own datasets.py, and lm-eval imports HuggingFace's
    # `datasets`. If the project root is on PYTHONPATH -- which it is whenever
    # the app is started with `PYTHONPATH=.`, a normal thing to do -- ours wins
    # the import and lm-eval dies with a baffling
    # "cannot import name 'load_dataset'". Strip our own directory out of the
    # child's path; it has no reason to import anything of ours.
    existing = env.get("PYTHONPATH", "")
    if existing:
        root = str(PROJECT_ROOT)
        kept = [
            entry for entry in existing.split(os.pathsep)
            if entry and os.path.abspath(entry) != root
        ]
        if kept:
            env["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            env.pop("PYTHONPATH", None)
    if task.needs_code_execution:
        # HumanEval is scored by running code the model wrote. This flag is the
        # harness's own deliberate speed bump, and it is only ever set on a task
        # the caller explicitly opted into -- never inferred, never defaulted.
        env["HF_ALLOW_CODE_EVAL"] = "1"
    return env


def _newest_results_file(output_dir: Path) -> Path | None:
    """lm-eval nests results under a sanitised model name it chooses itself."""
    candidates = sorted(
        output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


@dataclass
class RunOutcome:
    """Everything one benchmark invocation produced, success or not."""
    ok: bool
    result: Result | None = None
    error: str | None = None
    command: list[str] = field(default_factory=list)
    log_tail: str = ""


async def run_task(
    task: Task,
    *,
    base_url: str,
    model_name: str,
    output_dir: Path,
    limit: int | None = None,
    num_fewshot: int | None = None,
    concurrency: int = 1,
    on_progress: Any = None,
    cancel: asyncio.Event | None = None,
) -> RunOutcome:
    """Run one benchmark as a subprocess and parse what it produced.

    Cancellation kills the process rather than asking it to stop: lm-eval has
    no cooperative interrupt, and a benchmark the user has abandoned should
    stop consuming their GPU immediately.
    """
    executable = harness_installed()
    if executable is None:
        raise HarnessUnavailable(
            "lm-evaluation-harness is not installed. Run `pip install 'lm-eval[api]'` "
            "in the app's environment, then reload this page."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_command(
        task,
        executable=executable,
        base_url=base_url,
        model_name=model_name,
        output_dir=output_dir,
        limit=limit,
        num_fewshot=num_fewshot,
        concurrency=concurrency,
    )

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=_environment(task),
        cwd=str(PROJECT_ROOT),
    )

    # lm-eval writes its progress bar to stderr, which is merged into stdout
    # here. The tail is kept for the error message: when a run fails, the last
    # few lines are almost always the reason, and re-running to find out is
    # expensive.
    tail: list[str] = []
    buffer = ""
    last_note: str | None = None
    assert process.stdout is not None
    try:
        while True:
            if cancel is not None and cancel.is_set():
                process.kill()
                await process.wait()
                return RunOutcome(
                    ok=False, error="Stopped before finishing.", command=argv,
                    log_tail="\n".join(tail[-20:]),
                )
            try:
                chunk = await asyncio.wait_for(process.stdout.read(2048), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                break

            # Read raw chunks rather than lines: lm-eval draws its progress bar
            # with carriage returns and no newline, so readline() would block
            # until the run finished and the progress display would sit at 0%
            # for the entire run before jumping to done.
            buffer += chunk.decode(errors="replace")
            pieces = re.split(r"[\r\n]", buffer)
            buffer = pieces.pop()          # keep the incomplete tail for next time
            for line in pieces:
                line = line.strip()
                if not line:
                    continue
                tail.append(line)
                del tail[:-40]
                if on_progress is not None:
                    note = _progress_note(line)
                    if note and note != last_note:
                        last_note = note
                        result = on_progress(note)
                        if asyncio.iscoroutine(result):
                            await result
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()

    log_tail = "\n".join(tail[-20:])
    if process.returncode != 0:
        return RunOutcome(
            ok=False,
            error=f"lm-eval exited with code {process.returncode}.",
            command=argv,
            log_tail=log_tail,
        )

    results_file = _newest_results_file(output_dir)
    if results_file is None:
        return RunOutcome(
            ok=False,
            error="lm-eval finished but wrote no results file.",
            command=argv,
            log_tail=log_tail,
        )

    try:
        payload = json.loads(results_file.read_text())
        parsed = parse_results(payload, task)
    except (ValueError, HarnessUnavailable) as exc:
        return RunOutcome(ok=False, error=str(exc), command=argv, log_tail=log_tail)

    return RunOutcome(ok=True, result=parsed, command=argv, log_tail=log_tail)


def _progress_note(line: str) -> str | None:
    """Turn one line of lm-eval chatter into something worth showing.

    Most of its output is library logging that means nothing to someone
    watching a progress bar; only the request counter and the dataset download
    actually tell you where the run is.
    """
    if "Requesting API:" in line or "Running loglikelihood" in line:
        match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
        if match:
            _, done, total = match.groups()
            return f"Answering questions: {done}/{total}"
        return None
    if "Building contexts" in line:
        return "Building prompts"
    if "Downloading" in line or "Generating" in line and "split" in line:
        return "Fetching the dataset"
    if "Running loglikelihood requests" in line:
        return "Scoring candidate answers"
    return None
