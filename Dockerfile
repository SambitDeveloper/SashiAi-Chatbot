# ---------- 1. Build the React frontend ----------
FROM node:22-alpine AS frontend
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src
RUN npm run build

# ---------- 2. MongoDB MCP Server (Node.js), pinned version ----------
FROM node:22-slim AS mcp
RUN npm install --prefix /opt/mcp --omit=dev mongodb-mcp-server@3.3.0

# ---------- 3. Python runtime (API + static frontend + MCP server) ----------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/app ./app
# Bake the topic-check embedding model into the image (no download at startup, works offline)
RUN .venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"
COPY --from=frontend /web/dist /app/dist
# Node.js runtime + MongoDB MCP Server, started by the backend in read-only mode when DATA_SOURCE=mcp
COPY --from=mcp /usr/local/bin/node /usr/local/bin/node
COPY --from=mcp /opt/mcp /opt/mcp

RUN useradd --system --uid 10001 appuser && chown -R appuser /app
USER appuser

ENV ENVIRONMENT=production STATIC_DIR=/app/dist PORT=8000 TOPIC_GATE_CACHE_DIR=/app/models \
    MCP_SERVER_COMMAND="node /opt/mcp/node_modules/mongodb-mcp-server/dist/index.js"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
