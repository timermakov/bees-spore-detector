Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
docker compose up -d --build
docker compose exec backend alembic upgrade head
docker compose exec backend python backend/seed.py
Write-Host "Backend: http://localhost:$env:APP_PORT"
Write-Host "Frontend: http://localhost:$env:VITE_PORT"
