FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app
ENV PYTHONPATH=/app/backend:/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
