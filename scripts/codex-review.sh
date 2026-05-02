#!/usr/bin/env bash
# ==============================================================================
# Codex コードレビュースクリプト
# ==============================================================================
# 使用方法:
#   bash scripts/codex-review.sh                    # main との diff をレビュー
#   bash scripts/codex-review.sh develop             # develop との diff をレビュー
#   bash scripts/codex-review.sh --file src/train.py # 特定ファイルをレビュー
# ==============================================================================

set -euo pipefail

if ! command -v codex &> /dev/null; then
    echo "Error: Codex CLI がインストールされていません"
    echo "インストール: npm install -g @openai/codex"
    exit 1
fi

if [[ "${1:-}" == "--file" ]]; then
    FILE="${2:?ファイルパスを指定してください}"
    if [[ ! -f "$FILE" ]]; then
        echo "Error: ファイルが見つかりません: $FILE"
        exit 1
    fi
    INPUT=$(cat "$FILE")
    PROMPT="以下のコードをレビューしてください。バグ、セキュリティ問題、改善点を日本語で指摘してください。"
else
    BASE_BRANCH="${1:-main}"
    INPUT=$(git diff "$BASE_BRANCH" -- . ':(exclude)*.lock' ':(exclude)uv.lock' 2>/dev/null || git diff HEAD -- . ':(exclude)*.lock' ':(exclude)uv.lock')
    if [[ -z "$INPUT" ]]; then
        echo "差分がありません（ベース: $BASE_BRANCH）"
        exit 0
    fi
    PROMPT="以下のdiffをコードレビューしてください。バグ、セキュリティ問題、パフォーマンス、コードスタイルの観点から日本語でフィードバックしてください。"
fi

echo "$INPUT" | codex exec \
    --sandbox read-only \
    --output-last-message \
    "$PROMPT"
