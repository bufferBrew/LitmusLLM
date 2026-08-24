"""Central configuration, loaded once from the environment / .env file.

Every tunable in LitmusLLM funnels through this module so that a single
`.env` edit is enough to retarget the app (different Ollama host, different
judge, different concurrency) without touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# load_dotenv is a no-op when .env is absent, so a fresh checkout still runs.
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    """Read an env var, treating an empty string the same as 'unset'."""
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _path(key: str, default: str) -> Path:
    """Resolve a path-valued env var against the project root when relative."""
    raw = Path(_env(key, default))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


# --- Local inference runtimes ---------------------------------------------
# Three local backends are supported, each on its own port. All three speak
# OpenAI-compatible /v1/chat/completions, which is what lets one transport
# drive all of them -- see runtimes.py for the differences that remain.
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LMSTUDIO_HOST = _env("LMSTUDIO_HOST", "http://localhost:1234").rstrip("/")
LLAMACPP_HOST = _env("LLAMACPP_HOST", "http://localhost:8080").rstrip("/")

# Which runtime an unprefixed model name belongs to, and the only one the app
# will auto-start just to populate a page.
DEFAULT_RUNTIME = _env("LITMUSLLM_DEFAULT_RUNTIME", "ollama")

# llama.cpp is the one runtime with an open-ended launch line (GGUF path,
# context size, GPU layers). Set this to the exact command you use and
# LitmusLLM will run that instead of guessing.
LLAMACPP_START_COMMAND = _env("LITMUSLLM_LLAMACPP_START", "")

# How long to wait for an auto-started runtime to bind its port. llama.cpp
# loading a large GGUF from a cold page cache needs more than Ollama does.
RUNTIME_START_TIMEOUT = float(_env("LITMUSLLM_RUNTIME_START_TIMEOUT", "30"))

# DeepEval's own docs use LOCAL_MODEL_BASE_URL, so we honour the same name.
# It remains the Ollama default; per-runtime base URLs are derived in runtimes.py.
LOCAL_MODEL_BASE_URL = _env("LOCAL_MODEL_BASE_URL", f"{OLLAMA_HOST}/v1").rstrip("/")
LOCAL_MODEL_API_KEY = _env("LOCAL_MODEL_API_KEY", "ollama")

# --- Evaluation defaults --------------------------------------------------
DEFAULT_JUDGE = _env("LITMUSLLM_DEFAULT_JUDGE", "llama3.1:latest")
LOCAL_CONCURRENCY = int(_env("LITMUSLLM_LOCAL_CONCURRENCY", "2"))
CLOUD_CONCURRENCY = int(_env("LITMUSLLM_CLOUD_CONCURRENCY", "4"))
MAX_TOKENS = int(_env("LITMUSLLM_MAX_TOKENS", "512"))
TEMPERATURE = float(_env("LITMUSLLM_TEMPERATURE", "0.0"))
REQUEST_TIMEOUT = float(_env("LITMUSLLM_REQUEST_TIMEOUT", "300"))

# --- Storage --------------------------------------------------------------
DB_PATH = _path("LITMUSLLM_DB_PATH", "data/litmusllm.db")
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

# DeepEval reads this at import time, so set it before deepeval is imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", _env("DEEPEVAL_TELEMETRY_OPT_OUT", "YES"))

# Keep DeepEval's own result cache/artifacts inside our data dir rather than
# scattering .deepeval* files across the working directory.
os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", str(PROJECT_ROOT / "data" / "deepeval"))


def ensure_dirs() -> None:
    """Create the writable directories the app expects. Safe to call repeatedly."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "deepeval").mkdir(parents=True, exist_ok=True)
