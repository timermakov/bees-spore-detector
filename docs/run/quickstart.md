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
cd backend
alembic upgrade head
python seed.py
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
