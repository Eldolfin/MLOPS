FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

COPY data/ ./data/
COPY src/ ./src/
COPY pyproject.toml uv.lock .
RUN uv sync

EXPOSE 8080

CMD ["uv", "run", "fastapi", "run", "--port", "8080", "src/model_api.py"]
