# Quick Start (Windows + Docker)

## 1. Prepare environment

1. Copy `.env.example` to `.env`.
2. Adjust ports/credentials if needed.

## 2. Start services

```powershell
docker compose up -d --build
```

## 3. Apply migrations and seed demo data

```powershell
docker compose exec backend alembic upgrade head
docker compose exec backend python backend/seed.py
```

If you run migrations from host (outside Docker), ensure `.env` has localhost DB URL:

```powershell
DATABASE_URL=postgresql+psycopg://bees:bees@localhost:5432/bees_db
```

If you run tests locally (outside Docker), install requirements first:

```powershell
cd backend
pip install -r requirements.txt
pytest -q
```

## 4. Open apps

- Frontend: `http://localhost:5173`
- Backend OpenAPI: `http://localhost:8000/docs`

## 5. Run tests

```powershell
cd backend
pytest -q
```
