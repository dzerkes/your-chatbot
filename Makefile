COMPOSE ?= docker compose
UV ?= uv

.PHONY: up down psql test test-integration lint

up:
	$(COMPOSE) up -d --wait postgres

down:
	$(COMPOSE) down

psql:
	$(COMPOSE) exec postgres sh -c 'PGPASSWORD="$$OWNER_DB_PASSWORD" exec psql -X -h 127.0.0.1 -U "$$OWNER_DB_USER" -d "$$POSTGRES_DB"'

test:
	$(UV) run pytest -m 'not integration and not llm'

test-integration: up
	$(UV) run pytest -m integration

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
