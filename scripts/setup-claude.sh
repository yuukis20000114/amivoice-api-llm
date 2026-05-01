#!/usr/bin/env bash
# ==============================================================================
# Claude Code 初回セットアップスクリプト
# ==============================================================================
# 使用方法:
#   docker compose exec app bash scripts/setup-claude.sh
# ==============================================================================

set -euo pipefail

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ヘッダー表示
echo ""
echo "=============================================="
echo " Claude Code + MCP 初回セットアップ"
echo "=============================================="
echo ""

# 1. 環境確認
log_info "環境を確認しています..."

# Node.js確認
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    log_success "Node.js: $NODE_VERSION"
else
    log_error "Node.jsがインストールされていません"
    exit 1
fi

# Python確認
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_success "Python: $PYTHON_VERSION"
else
    log_error "Python3がインストールされていません"
    exit 1
fi

# uv確認
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    log_success "uv: $UV_VERSION"
else
    log_error "uvがインストールされていません"
    exit 1
fi

# Claude Code確認
if command -v claude &> /dev/null; then
    log_success "Claude Code: インストール済み"
else
    log_warn "Claude Codeがインストールされていません。インストールを試みます..."
    npm install -g @anthropic-ai/claude-code
fi

echo ""

# 2. GPU確認
log_info "GPU環境を確認しています..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null && \
        log_success "NVIDIA GPU: 検出" || \
        log_warn "nvidia-smiは利用可能ですが、GPUが検出されません"
else
    log_warn "nvidia-smiが利用できません（GPU非対応環境の可能性）"
fi

echo ""

# 3. MCP設定確認
log_info "MCP設定を確認しています..."

if [ -f "/work/.claude.json" ]; then
    log_success ".claude.json: 存在"
    # MCP数をカウント
    MCP_COUNT=$(jq '.mcpServers | length' /work/.claude.json 2>/dev/null || echo "0")
    log_info "設定済みMCPサーバー数: $MCP_COUNT"
else
    log_warn ".claude.jsonが見つかりません"
fi

echo ""

# 4. textlint設定確認
log_info "textlint設定を確認しています..."

if [ -f "/work/.textlintrc.json" ]; then
    log_success ".textlintrc.json: 存在"
else
    log_warn ".textlintrc.jsonが見つかりません"
fi

# textlintルール確認
if npm list -g textlint &> /dev/null; then
    log_success "textlint: グローバルインストール済み"
else
    log_warn "textlintがインストールされていません。インストールを試みます..."
    npm install -g textlint textlint-rule-preset-ja-technical-writing textlint-rule-preset-jtf-style textlint-rule-prh
fi

echo ""

# 5. Playwright確認
log_info "Playwrightを確認しています..."

# npx --no でプロンプトを抑制し、タイムアウトを設定
if timeout 5 npx --no playwright --version &> /dev/null; then
    PLAYWRIGHT_VERSION=$(npx --no playwright --version 2>/dev/null || echo "unknown")
    log_success "Playwright: $PLAYWRIGHT_VERSION"
else
    log_warn "Playwrightがインストールされていない可能性があります"
    log_info "インストールする場合: npm install -g playwright && npx playwright install chromium"
fi

echo ""

# 6. Claude Code認証状態確認
log_info "Claude Code認証状態を確認しています..."

if [ -d "/root/.claude" ] && [ -f "/root/.claude/credentials.json" ] 2>/dev/null; then
    log_success "Claude Code: 認証済み"
else
    log_warn "Claude Code: 未認証"
    echo ""
    echo "=============================================="
    echo " Claude Code 認証手順"
    echo "=============================================="
    echo ""
    echo "1. 以下のコマンドを実行:"
    echo "   ${GREEN}claude${NC}"
    echo ""
    echo "2. 表示されるURLをブラウザで開く"
    echo ""
    echo "3. Anthropicアカウントでログイン"
    echo ""
    echo "4. 認証コードをターミナルに貼り付け"
    echo ""
    echo "=============================================="
fi

echo ""

# 7. GPU直列化セットアップ確認
log_info "GPU直列化設定を確認しています..."

if command -v flock &> /dev/null; then
    log_success "flock: 利用可能"
else
    log_error "flockが利用できません（GPU直列化に必要）"
fi

if [ -x "/work/scripts/run_gpu.sh" ]; then
    log_success "run_gpu: 利用可能"
else
    log_warn "run_gpu.shが見つかりません"
fi

if [ -f "/work/.claude/settings.json" ]; then
    HOOK_COUNT=$(jq '.hooks.PreToolUse | length' /work/.claude/settings.json 2>/dev/null || echo "0")
    log_success "PreToolUseフック: ${HOOK_COUNT}個設定済み"
else
    log_warn ".claude/settings.jsonが見つかりません（GPU自動直列化が無効）"
fi

echo ""

# 8. サマリー
echo "=============================================="
echo " セットアップ完了"
echo "=============================================="
echo ""
echo "次のステップ:"
echo ""
echo "  1. Claude Code認証（未認証の場合）:"
echo "     ${GREEN}claude${NC}"
echo ""
echo "  2. MCP確認:"
echo "     ${GREEN}claude mcp list${NC}"
echo ""
echo "  3. GPU確認:"
echo "     ${GREEN}python src/check_gpu.py${NC}"
echo ""
echo "  4. Jupyter Lab起動:"
echo "     ${GREEN}jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root${NC}"
echo ""
echo "=============================================="