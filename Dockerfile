FROM python:3.11-slim

WORKDIR /app

# psycopg2-binary ships its own libpq, so no build toolchain is needed here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads

# Render and most PaaS providers inject PORT. Fall back to 8080 locally.
ENV PORT=8080
EXPOSE 8080

# One worker: the free tier has little memory, and this workload is IO bound
# (API calls out to Gemini and Postgres) so threads carry the concurrency.
CMD ["sh", "-c", "gunicorn app.main:app --bind 0.0.0.0:${PORT} --worker-class uvicorn.workers.UvicornWorker --workers 1 --threads 4 --timeout 120"]
