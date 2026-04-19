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
docker compose up -d --build db
docker compose run --rm backend sh -lc "cd /app/backend && alembic -c alembic.ini upgrade head"
docker compose run --rm backend python /app/backend/seed.py
docker compose up -d backend frontend
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

# DB Schema
Entity relations diagram:
![er_diagram](../schemas/er_diagram.png)

# Frontend Screenshots

![frontend_1](../images/frontend_1.jpg)

![frontend_2](../images/frontend_2.jpg)

![frontend_3](../images/frontend_3.jpg)

![frontend_4](../images/frontend_4.jpg)

# Backend Screenshots

![backend_1](../images/backend_1.png)

![backend_2](../images/backend_2.png)

![backend_3](../images/backend_3.png)

