# API Endpoints

Base URL: `http://localhost:8000`

## Core CRUD

- `GET /users`, `POST /users`
- `GET /species`, `POST /species`, `PUT /species/{id}`, `DELETE /species/{id}`
- `GET /projects`, `POST /projects`, `PUT /projects/{id}`, `DELETE /projects/{id}`
- `GET /probes`, `POST /probes`, `PUT /probes/{id}`, `DELETE /probes/{id}`
- `GET /samples`, `POST /samples`, `PUT /samples/{id}`, `DELETE /samples/{id}`
- `GET /images`, `DELETE /images/{id}`
- `POST /samples/{sample_id}/images` (multipart upload of JPG/PNG)
- `GET /probe-results`, `POST /probe-results` (manual upsert)
- `GET /model-weights`, `POST /model-weights`, `DELETE /model-weights/{id}`

## Analytics

- `POST /probes/{probe_id}/analyze?mode=yolo|opencv`
  - reads uploaded sample images for selected probe
  - runs analyzer and stores aggregate in `probe_results`
  - returns `mean_titer`, `std_titer`, `n_measurements`, `p_value`, `method`

## Health

- `GET /health`
