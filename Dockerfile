# DELPHI published API image (CLAUDE.md §7: deliberately minimal).
#
# Serves `api.wsgi:application` under gunicorn. The app is fail-closed on auth:
# it requires DELPHI_SECRET_API_TOKEN to serve forecasts (health/readiness stay
# open for load-balancer probes). Provide at runtime:
#   DELPHI_PG_DSN, DELPHI_SECRET_ANTHROPIC_API_KEY, DELPHI_SECRET_TAVILY_API_KEY,
#   DELPHI_SECRET_API_TOKEN  (+ optional DELPHI_LLM_PROVIDER / DELPHI_MODEL_* / etc.)
#
#   docker build -t delphi-api .
#   docker run --rm -p 8080:8080 --env-file .env -e DELPHI_SECRET_API_TOKEN=... delphi-api

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Layer 1: dependencies only (cached until the lockfile changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra serve

# Layer 2: the project source + install it.
COPY . .
RUN uv sync --frozen --extra serve


FROM python:3.12-slim-bookworm

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 delphi

WORKDIR /app
COPY --from=build --chown=delphi:delphi /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

USER delphi
EXPOSE 8080

# gunicorn workers are I/O-bound (they wait on LLM/search calls), so a longer
# timeout and a modest worker count are appropriate. --threads > 1 selects the
# gthread worker so long-polls on /v1/forecast/jobs/{id}?wait=N cannot starve
# the worker pool. Tune via WEB_CONCURRENCY / GUNICORN_THREADS /
# GUNICORN_TIMEOUT / PORT at deploy time.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers ${WEB_CONCURRENCY:-2} --threads ${GUNICORN_THREADS:-8} --timeout ${GUNICORN_TIMEOUT:-300} --access-logfile - --error-logfile - api.wsgi:application"]
