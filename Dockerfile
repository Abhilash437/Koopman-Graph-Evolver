FROM python:3.10-slim

# Install system dependencies (needed for some scientific packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code and GUI script
COPY koopman_evolver /app/koopman_evolver
COPY app.py /app/app.py

# Set entrypoint to the CLI
ENTRYPOINT ["python", "-m", "koopman_evolver.cli"]
CMD ["-h"]
