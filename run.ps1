Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
docker compose up -d --build
Set-Location backend
python -m pip install -r requirements.txt
alembic upgrade head
python seed.py
Write-Host "Backend: http://localhost:$env:APP_PORT"
Write-Host "Frontend: http://localhost:$env:VITE_PORT"
