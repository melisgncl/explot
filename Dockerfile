FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY explot/ ./explot/
COPY simulator/ ./simulator/
COPY config/ ./config/

RUN pip install --no-cache-dir -e ".[ml,survival,parquet,excel]"

WORKDIR /workspace

ENTRYPOINT ["explot"]
CMD ["--help"]
