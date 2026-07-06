FROM python:3.13-slim

# Install uv (Python package/workspace manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the full repo into the image
COPY . .

# Install all workspace dependencies
RUN uv sync --all-packages

# Pre-cache the cross-encoder model at build time (~800MB) so it is not
# downloaded on cold-start at runtime.
RUN uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 7860

# --workers 1 is mandatory: the rate limiter is in-process.
# HF Docker Spaces always serve on port 7860 (hardcoded, not $PORT).
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "apps/api/src", "--workers", "1"]
