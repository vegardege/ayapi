FROM python:3.13-slim

WORKDIR /app

# Setup the base system with `poetry`
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && mv /root/.local/bin/poetry /usr/local/bin/poetry

# Tell Poetry to install into the container (not a venv)
RUN poetry config virtualenvs.create false --local

# Install dependencies
COPY pyproject.toml poetry.lock README.md ./
COPY src/ src/
RUN poetry install --no-interaction --no-ansi --without dev

# Run app
CMD ["poetry", "run", "fastapi", "run", "src/ayapi/main.py"]