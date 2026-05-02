# ====================== 1. ベースイメージ ======================
ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

# プロキシ設定を.envファイルから読み込む
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG no_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV no_proxy=${no_proxy}

# ====================== 2. 環境変数 & 基本ツール =================
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    PATH="/root/.local/bin:/root/.npm-global/bin:$PATH" \
    NPM_CONFIG_PREFIX=/root/.npm-global \
    # Claude Code設定
    CLAUDE_CONFIG_DIR=/root/.claude \
    NODE_OPTIONS="--max-old-space-size=4096"

# --------- 会社プロキシ証明書がある場合のみ有効化 ----------
COPY cert_TrustCA_pa.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

# ====================== 3. システムパッケージ ===================
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        # 基本ツール
        tzdata ca-certificates curl git wget \
        # Python関連
        python3 python3-venv python3-distutils \
        # Claude Code / MCP用追加ツール
        ripgrep \
        fd-find \
        jq \
        tree \
        # Playwright依存（ヘッドレスブラウザ用）
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxkbcommon0 \
        libatspi2.0-0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
 && rm -rf /var/lib/apt/lists/*

# ====================== 4. Node.js 20 LTS インストール ==========
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /root/.npm-global \
 && npm config set prefix '/root/.npm-global'

# Node.jsバージョン確認
RUN node --version && npm --version

# ====================== 5. uv インストール ======================
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
 && ~/.local/bin/uv --version

# ====================== 6. Claude Code インストール =============
RUN npm install -g @anthropic-ai/claude-code \
 && claude --version || echo "Claude Code installed (version check may require auth)"

# ====================== 6.5. Codex CLI インストール ==============
RUN npm install -g @openai/codex \
 && codex --version || echo "Codex CLI installed"

# MCP用Node.jsパッケージ（事前インストールでnpx高速化）
RUN npm install -g \
    @anthropic-ai/claude-code \
    @upstash/context7-mcp \
    @modelcontextprotocol/server-github \
    mcp-sequentialthinking-tools

# ====================== 7. Playwright ブラウザ ==================
# Chromiumのみインストール（サイズ削減）
RUN npx playwright install chromium --with-deps \
 && npx playwright install-deps chromium

# ====================== 8. アプリセットアップ ===================
WORKDIR /work
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# textlint関連パッケージ（プロジェクトで使用する場合）
RUN npm install -g \
    textlint \
    textlint-rule-preset-ja-technical-writing \
    textlint-rule-preset-jtf-style \
    textlint-rule-prh

COPY . .

# ====================== 9. GPU直列化セットアップ =================
RUN chmod +x /work/scripts/run_gpu.sh \
 && ln -sf /work/scripts/run_gpu.sh /usr/local/bin/run_gpu \
 && chmod +x /work/scripts/codex-review.sh \
 && chmod +x /work/.claude/hooks/gpu_serialize.sh \
 && mkdir -p /work/.gpu_logs

# ====================== 9.5. Git hooks セットアップ ================
RUN chmod +x /work/scripts/setup-git-hooks.sh \
 && chmod +x /work/scripts/git-hooks/* \
 && bash /work/scripts/setup-git-hooks.sh

# ====================== 10. Claude設定ディレクトリ ===============
RUN mkdir -p /root/.claude

# ====================== 11. ヘルスチェック用 ====================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node --version && python3 --version || exit 1

# デフォルトコマンド（docker-compose.ymlで上書き）
# CMD ["bash"]