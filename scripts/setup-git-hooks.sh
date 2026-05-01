#!/usr/bin/env bash
# ==============================================================================
# Git hooks セットアップ
# ==============================================================================
# scripts/git-hooks/ 配下のフックを .git/hooks/ にシンボリックリンクする。
# Dockerfile または手動で実行:
#   bash scripts/setup-git-hooks.sh
# ==============================================================================

set -euo pipefail

HOOKS_SRC="/work/scripts/git-hooks"
HOOKS_DST="/work/.git/hooks"

if [ ! -d "$HOOKS_SRC" ]; then
    echo "[WARN] $HOOKS_SRC が見つかりません。スキップします。"
    exit 0
fi

if [ ! -d "$HOOKS_DST" ]; then
    echo "[WARN] $HOOKS_DST が見つかりません（.gitが未初期化）。スキップします。"
    exit 0
fi

for hook in "$HOOKS_SRC"/*; do
    [ -f "$hook" ] || continue
    name=$(basename "$hook")
    ln -sf "$hook" "$HOOKS_DST/$name"
    echo "[OK] $name -> $hook"
done
