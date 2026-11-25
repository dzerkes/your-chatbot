# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY chatbot/pyproject.toml chatbot/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
COPY frontend/pyproject.toml frontend/pyproject.toml
COPY mcp_servers/pyproject.toml mcp_servers/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package chatbot-mcp-servers --no-install-workspace
COPY chatbot/src chatbot/src
COPY mcp_servers/*.py mcp_servers/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --package chatbot-mcp-servers

FROM ${PYTHON_IMAGE} AS mcp
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/home/app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /home/app/.cache/fastembed && chown -R app:app /home/app/.cache
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
USER app
EXPOSE 8001
CMD ["python", "-m", "mcp_servers", "--host", "0.0.0.0"]
