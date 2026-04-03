FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ────────────────────────────────────────────────────
COPY . .

# ── Environment variables (overridden at runtime) ──────────────────
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ── Hugging Face Spaces: must listen on port 7860 ──────────────────
EXPOSE 7860

# ── Health-check (HF pings / to confirm Space is live) ────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start the FastAPI server ───────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]