.PHONY: up down migrate seed backend-test

up:
	docker compose up -d --build

down:
	docker compose down

migrate:
	cd backend && alembic upgrade head

seed:
	cd backend && python seed.py

backend-test:
	cd backend && pytest -q
