FROM node:22-bookworm-slim AS css

WORKDIR /build

COPY theme/static_src/ ./

RUN npm install && npm run build


FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Retain the stylesheet in the theme's static directory.
COPY --from=css /build/output.css \
    /app/theme/static/css/dist/styles.css

# Put it directly in STATIC_ROOT without initializing Django at build time.
COPY --from=css /build/output.css \
    /app/staticfiles/css/dist/styles.css

# Database operations happen only at runtime, after Compose's
# database health check has passed.
CMD ["sh", "-c", "python manage.py migrate --run-syncdb --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 240"]
