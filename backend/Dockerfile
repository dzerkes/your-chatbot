# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build-base
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY chatbot/pyproject.toml chatbot/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
COPY frontend/pyproject.toml frontend/pyproject.toml

FROM build-base AS build-backend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package chatbot-backend --no-install-workspace
COPY chatbot/src chatbot/src
COPY backend/app backend/app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package chatbot-backend

FROM build-base AS build-migrate
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group migrations --no-editable --no-install-workspace
COPY chatbot/src chatbot/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group migrations --no-editable

FROM ${PYTHON_IMAGE} AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/home/app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
WORKDIR /app
USER app

FROM runtime AS migrate
COPY --from=build-migrate /opt/venv /opt/venv
COPY db/alembic.ini db/bootstrap.py db/
COPY db/migrations db/migrations
COPY db/seed db/seed
CMD ["python", "db/bootstrap.py"]

FROM runtime AS backend
COPY --from=build-backend /opt/venv /opt/venv
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
