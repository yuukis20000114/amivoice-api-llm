#!/usr/bin/env bash
# ==============================================================================
# PreToolUse フック: GPU コマンドの自動直列化
# ==============================================================================
# Claude Code の Bash ツール呼び出しを監視し、Python スクリプト実行等の
# GPU コマンドを flock で自動ラップして直列化する。
#
# 直列化するパターン:
#   - uv run python(3) *.py / python(3) *.py
#   - torchrun / accelerate launch
#   - make gpu-check
#
# 直列化しないパターン:
#   - python --version, python -m pip/ruff/..., python -c "..."
#   - uv add/sync/pip/lock/venv
#   - ruff, pip, make lint/format/clean 等
#   - すでに flock/run_gpu で包まれているコマンド
# ==============================================================================

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[ -z "$COMMAND" ] && exit 0

# すでにflock/run_gpuで包まれている場合はスキップ
echo "$COMMAND" | grep -q 'flock.*gpu\.lock\|run_gpu' && exit 0

# GPU不使用パターン（除外）
EXCLUDE='(uv\s+run\s+)?python[3]?\s+(--version|-m\s+\S|-(c|V)\s)|^uv\s+(add|sync|pip|lock|venv)\b|^(ruff|pip|make\s+(lint|format|clean|help))\b'
echo "$COMMAND" | grep -qP "$EXCLUDE" && exit 0

# 直列化パターン
GPU_PATTERN='(uv\s+run\s+)?python[3]?\s+\S+\.py|torchrun\b|accelerate\s+launch|make\s+gpu-check'
if echo "$COMMAND" | grep -qP "$GPU_PATTERN"; then
    WRAPPED="flock -x /var/lock/gpu.lock bash -c $(printf '%q' "$COMMAND")"
    jq -n --arg cmd "$WRAPPED" '{
        hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "allow",
            permissionDecisionReason: "GPU command auto-serialized via flock",
            updatedInput: { command: $cmd }
        }
    }'
else
    exit 0
fi
