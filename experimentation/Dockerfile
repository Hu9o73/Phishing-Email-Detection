# Stage 1: grab the Ollama CLI from the official image (no external download URLs needed)
FROM ollama/ollama:latest AS ollama

# Stage 2: final runtime image
FROM python:3.12-slim

WORKDIR /app/experimentation

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bring in the Ollama CLI binary from the official image
COPY --from=ollama /bin/ollama /usr/local/bin/ollama

# Python deps for experimentation code
COPY experimentation/requirements.txt /app/experimentation/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/experimentation/requirements.txt

# Copy the evaluation script
COPY experimentation/ollama_llm_eval.py /app/experimentation/ollama_llm_eval.py

# Default command; override in compose or docker run as needed
CMD ["python", "ollama_llm_eval.py", "--csv-path", "/app/data/enron_data_fraud_labeled.csv"]
