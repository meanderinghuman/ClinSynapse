FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy CS_DEVICE=cpu
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group obs --group ui --no-install-project
COPY src ./src
COPY evals ./evals
COPY ui ./ui
RUN uv sync --frozen --no-dev --group obs --group ui
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "clinsynapse.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
