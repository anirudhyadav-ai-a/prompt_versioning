# prompt_versioning — multi-stage production Dockerfile
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/wheels -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
COPY --from=builder /wheels /usr/local/lib/python3.12/site-packages
COPY . .
RUN chown -R appuser:appgroup /app
USER appuser
CMD ["python", "-m", "prompt_versioning.examples.prompt_registry.main"]
