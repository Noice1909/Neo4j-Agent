# ── Builder stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install project dependencies from pyproject.toml (single source of truth)
COPY pyproject.toml .
COPY app/ ./app/
RUN pip install --no-cache-dir ".[dev]" 2>/dev/null || pip install --no-cache-dir . \
    && pip install --no-cache-dir slowapi structlog python-json-logger \
       prometheus-fastapi-instrumentator

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/ ./app/

# Ensure the non-root user owns the files
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Worker count configurable via environment variable
ENV WEB_CONCURRENCY=2

# Health check
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

# Start Uvicorn — worker count from WEB_CONCURRENCY, graceful shutdown enabled
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --timeout-graceful-shutdown 30 --log-level info"]
