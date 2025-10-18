# NeuroFM-X Production Deployment Dockerfile
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package files
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with all dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[all]"

# Stage 2: Runtime
FROM python:3.10-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NEUROFM_MODEL_DIR=/app/models
ENV NEUROFM_DATA_DIR=/app/data
ENV NEUROFM_LOG_DIR=/app/logs

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default command: run API server
CMD ["python", "-m", "neuros_neurofm.api.server", "--host", "0.0.0.0", "--port", "8000"]
