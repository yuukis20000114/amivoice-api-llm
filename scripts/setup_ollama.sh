#!/usr/bin/env bash
set -euo pipefail

echo "=== Ollama セットアップ ==="

if command -v ollama &>/dev/null; then
    echo "Ollama は既にインストール済み"
else
    echo "Ollama をインストール中..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

if pgrep -x ollama &>/dev/null; then
    echo "Ollama サーバーは既に起動中"
else
    echo "Ollama サーバーを起動中..."
    ollama serve &
    sleep 3
fi

MODEL="qwen3.5:9b"
echo "モデル ${MODEL} を取得中..."
ollama pull "${MODEL}"

echo "=== インストール済みモデル ==="
ollama list

echo "=== セットアップ完了 ==="
