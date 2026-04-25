FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV HF_TOKEN=
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml /app/
COPY uv.lock /app/
COPY __init__.py constants.py engine.py graders.py models.py planner.py tasks.py /app/
COPY server /app/server
COPY scripts /app/scripts
COPY multi_agent /app/multi_agent
COPY domains /app/domains
COPY training /app/training

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
