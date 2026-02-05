# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src ./src

# Install dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install . && \
    uv pip install git+https://github.com/openai/CLIP.git

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser assets ./assets

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3001/api/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3001"]
