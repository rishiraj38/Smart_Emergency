# ─────────────────────────────────────────────────────────────────────────────
# Emergency Dispatch — OpenEnv Environment (Hugging Face Spaces)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy and install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Fix ownership for non-root user
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Environment variables
ENV PORT=7860
ENV HOST=0.0.0.0
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]