# LitmusLLM application image.
#
# Slim rather than alpine: DeepEval and LiteLLM pull in packages (tokenizers,
# grpcio, pydantic-core) that ship manylinux wheels but not musl ones, so
# alpine would force a long source build for no size win.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so edits to app code don't invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root, and it must own /app/data because SQLite needs to create the
# database plus its -wal/-shm sidecar files at runtime.
RUN useradd --create-home --uid 1000 litmus \
 && mkdir -p /app/data/uploads /app/data/deepeval \
 && chown -R litmus:litmus /app
USER litmus

EXPOSE 8000

# Container-internal defaults. docker-compose overrides OLLAMA_HOST to reach
# the ollama service; these values make `docker run` alone work too.
ENV OLLAMA_HOST=http://ollama:11434 \
    LOCAL_MODEL_BASE_URL=http://ollama:11434/v1 \
    LITMUSLLM_DB_PATH=/app/data/litmusllm.db

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health',timeout=4)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
