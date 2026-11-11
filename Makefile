COMPOSE ?= docker compose
UV ?= uv

.PHONY: up down psql migrate seed test test-integration lint

up:
	$(COMPOSE) up -d --wait postgres

down:
	$(COMPOSE) down

psql:
	$(COMPOSE) exec postgres sh -c 'PGPASSWORD="$$OWNER_DB_PASSWORD" exec psql -X -h 127.0.0.1 -U "$$OWNER_DB_USER" -d "$$POSTGRES_DB"'

migrate: up
	$(COMPOSE) run --rm migrate

seed: migrate
	$(COMPOSE) exec -T postgres sh -c 'PGPASSWORD="$$OWNER_DB_PASSWORD" exec psql -X -h 127.0.0.1 -U "$$OWNER_DB_USER" -d "$$POSTGRES_DB" -v ON_ERROR_STOP=1' < db/seed/01_demo.sql

test:
	$(UV) run pytest -m 'not integration and not llm'

test-integration: migrate
	$(UV) run pytest -m integration

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
