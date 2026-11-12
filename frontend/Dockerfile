# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY chatbot/pyproject.toml chatbot/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
COPY frontend/pyproject.toml frontend/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package chatbot-frontend --no-install-workspace
COPY frontend/*.py frontend/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package chatbot-frontend

FROM ${PYTHON_IMAGE} AS frontend
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/home/app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
USER app
EXPOSE 8501
CMD ["python", "-m", "frontend", "--host", "0.0.0.0"]
