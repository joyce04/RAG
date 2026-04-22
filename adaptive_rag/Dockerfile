FROM python:3.13-slim

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies globally inside the container
RUN uv sync --no-dev

# Copy project files
COPY . .

# Expose the API port
EXPOSE 8000

# Run FastAPI server
# Since uv sync creates a virtual env by default if not strictly forced, we use it directly or 
# relies on the system install. Assuming `uv sync` installed in `/app/.venv` or system.
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
