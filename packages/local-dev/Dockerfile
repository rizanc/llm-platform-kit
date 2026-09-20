FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv \
    && uv pip install --system \
        "fastapi>=0.115" "uvicorn[standard]>=0.32" \
        "lancedb>=0.8" "pyarrow>=18" \
        "ollama>=0.3" "pypdf>=5.0" \
        "pydantic>=2" "pydantic-settings>=2"

COPY localrag ./localrag
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "localrag.main:app", "--host", "0.0.0.0", "--port", "8000"]