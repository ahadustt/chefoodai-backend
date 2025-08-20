# Multi-stage build for ChefoodAI Backend Service
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Make startup script executable
RUN chmod +x startup.py

# Switch to non-root user
USER appuser

# Set Python to unbuffered mode for Cloud Run logging
ENV PYTHONUNBUFFERED=1

# Set default port (Cloud Run will override this)
ENV PORT=8000

# Expose port (documentation only)
EXPOSE 8000

# Health check - removed as Cloud Run handles this
# HEALTHCHECK is not needed for Cloud Run

# Run the application using the startup script
CMD ["python3", "startup.py"]

