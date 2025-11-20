FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/app/.venv/bin:$PATH"

# Copy the project configuration files into the container
# pyproject.toml     → project metadata and dependencies
# uv.lock            → locked dependency versions (for reproducibility)
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies in a reproducible isolated environment
RUN uv sync --frozen --no-dev

# Copy application code and model data into the container
COPY "app.py" "preprocessing_utils.py" ./
COPY "models/" "./models/"

# Expose TCP port 8000 so it can be accessed from outside the container
EXPOSE 8000

# Run the application using uvicorn (ASGI server)
# app:app → refers to 'app' object inside app.py
# --host 0.0.0.0 → listen on all interfaces
# --port 8000    → listen on port 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]