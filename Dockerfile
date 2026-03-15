FROM python:3.13 AS builder

ENV POETRY_VERSION=2.0.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 -
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN /opt/poetry/bin/poetry install --no-root --only main

# === Runtime Stage ===
FROM python:3.13-slim AS runtime
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copy the venv and the application code
COPY --from=builder /app/.venv /app/.venv
COPY . .

# IMPORTANT: Put the venv at the front of the PATH
# This ensures that 'python' refers to the venv's python
ENV PATH="/app/.venv/bin:$PATH"

# Use only ONE entrypoint
ENTRYPOINT ["python", "cirrhosis_patient_survival_prediction/model.py"]