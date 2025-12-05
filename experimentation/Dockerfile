FROM python:3.12-slim

WORKDIR /app/experimentation

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps and Ollama CLI (used to talk to the Ollama server from this container)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama \
    && chmod +x /usr/local/bin/ollama \
    && rm -rf /var/lib/apt/lists/*

# Python deps for experimentation code
COPY experimentation/requirements.txt /app/experimentation/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/experimentation/requirements.txt

# Copy the evaluation script
COPY experimentation/ollama_llm_eval.py /app/experimentation/ollama_llm_eval.py

# Default command; override in compose or docker run as needed
CMD ["python", "ollama_llm_eval.py", "--csv-path", "/app/data/enron_data_fraud_labeled.csv"]
