#!/usr/bin/env bash
# ==============================================================================
# run_gpu: GPU実行の直列化ラッパー
# ==============================================================================
# flock による排他ロックで GPU コマンドを直列化する。
# 同時に複数の run_gpu が呼ばれても、ロック待ちで自動的にキュー化される。
#
# 使用方法:
#   run_gpu uv run python src/train.py --epochs 10
#   run_gpu make gpu-check
# ==============================================================================

set -euo pipefail

LOCK_FILE="/var/lock/gpu.lock"
LOG_DIR="/work/.gpu_logs"
mkdir -p "$LOG_DIR"

TS="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$TS] QUEUED: $*" >> "$LOG_DIR/gpu_queue.log"

exec flock -x "$LOCK_FILE" bash -c '
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] STARTED: $*" >> /work/.gpu_logs/gpu_queue.log
    "$@"
    rc=$?
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] FINISHED (exit $rc): $*" >> /work/.gpu_logs/gpu_queue.log
    exit $rc
' _ "$@"
