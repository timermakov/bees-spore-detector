Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
docker compose up -d --build db
docker compose run --rm backend sh -lc "cd /app/backend && alembic -c alembic.ini upgrade head || alembic -c alembic.ini stamp head"
docker compose run --rm backend python /app/backend/seed.py
docker compose up -d backend frontend

$backendPort = "8000"
$frontendPort = "5173"

if (Test-Path ".env") {
    $envLines = Get-Content ".env"
    foreach ($line in $envLines) {
        if ($line -match "^\s*APP_PORT\s*=\s*(.+)\s*$") {
            $backendPort = $matches[1].Trim()
        }
        if ($line -match "^\s*VITE_PORT\s*=\s*(.+)\s*$") {
            $frontendPort = $matches[1].Trim()
        }
    }
}

Write-Host "Backend: http://localhost:$backendPort"
Write-Host "Frontend: http://localhost:$frontendPort"
