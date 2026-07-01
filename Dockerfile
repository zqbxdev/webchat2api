ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETARCH

FROM --platform=$BUILDPLATFORM node:22-alpine AS web-build

WORKDIR /app/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY VERSION /app/VERSION
COPY web ./
RUN NEXT_PUBLIC_APP_VERSION="$(cat /app/VERSION)" npm run build


FROM --platform=$TARGETPLATFORM python:3.13-slim AS app

ARG TARGETPLATFORM
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    CHROMIUM_PATH=/usr/bin/chromium \
    BRIDGE_PORT=3080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libpq-dev \
    gcc \
    openssl \
    curl \
    chromium \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libpango-1.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# mihomo 代理内核（订阅代理池用，固定 v1.19.27 与开发环境一致）
RUN case "$TARGETARCH" in \
      amd64) _ARCH="linux-amd64-compatible" ;; \
      arm64) _ARCH="linux-arm64" ;; \
      *) echo "unsupported arch: $TARGETARCH" && exit 1 ;; \
    esac && \
    _VER="v1.19.27" && \
    mkdir -p /app/scripts/.bin && \
    curl -fsSL "https://github.com/MetaCubeX/mihomo/releases/download/${_VER}/mihomo-${_ARCH}-${_VER}.gz" -o /tmp/mihomo.gz && \
    python3 -c "import gzip,shutil; shutil.copyfileobj(gzip.open('/tmp/mihomo.gz','rb'), open('/app/scripts/.bin/mihomo','wb'))" && \
    chmod +x /app/scripts/.bin/mihomo && \
    rm -f /tmp/mihomo.gz

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/browser_bridge/package.json services/browser_bridge/package-lock.json /app/services/browser_bridge/
RUN cd /app/services/browser_bridge && npm ci --omit=dev && cd /app

COPY main.py ./
COPY config.example.json ./config.json
COPY VERSION ./
COPY api ./api
COPY services ./services
COPY utils ./utils
COPY scripts ./scripts
COPY --from=web-build /app/web/out ./web_dist

RUN chmod +x /app/scripts/entrypoint.sh

EXPOSE 83

CMD ["/app/scripts/entrypoint.sh"]
