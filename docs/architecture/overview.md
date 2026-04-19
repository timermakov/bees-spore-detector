# Architecture Overview

## Components

- **Frontend (`frontend`)**: React + TypeScript SPA for CRUD, upload, analytics triggers, and dashboard charts.
- **Backend (`backend/app`)**: FastAPI service exposing CRUD and analysis endpoints.
- **Database**: PostgreSQL with SQLAlchemy ORM and Alembic migrations.
- **Storage**: local `uploads/` directory for microscopy images.
- **Analytics Core**: existing `bees` package reused by backend service layer.

## Runtime flow

1. User creates `Project -> Probe -> Sample`.
2. User uploads JPG/PNG images into sample.
3. Frontend calls `POST /probes/{id}/analyze`.
4. Backend executes analysis service using existing `bees` pipeline.
5. Aggregated metrics are upserted into `probe_results`.
6. Frontend dashboard renders table and plot from results.

## Why this design

- Keeps biological analysis logic in one reusable place.
- Preserves clean separation between UI, API, DB, and compute layer.
- Supports easy scaling: analysis can later move to worker queue without API contract changes.
