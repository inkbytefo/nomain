## Developer: inkbytefo
## Modified: 2025-11-09

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY cloud_deployment/requirements-cloud.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 chimera && chown -R chimera:chimera /app
USER chimera

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "run_chimera.py", "--host", "0.0.0.0", "--port", "8000"]
