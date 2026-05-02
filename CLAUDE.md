# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated Python development environment template using Docker, NVIDIA CUDA 12.4, and uv package manager. Pre-configured with PyTorch, TensorFlow, CatBoost, and XGBoost for ML/DL projects.

## Common Commands

```bash
# Package management (uv)
uv add <package>              # Add a dependency
uv add --dev <package>        # Add dev dependency
uv sync                       # Install dependencies from pyproject.toml

# Code quality (ruff)
uv run ruff check .           # Run linter
uv run ruff format .          # Format code

# GPU verification
uv run python src/check_gpu.py # Run GPU availability check

# Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Docker operations (run from host)
docker compose up -d --build  # Build and start container
docker compose exec app bash  # Enter container shell
docker compose down -v        # Stop and remove volumes

# Cleanup (remove Python cache files)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -type f -name "*.pyc" -delete 2>/dev/null
```

## Project Structure

- `src/` - Python source code
- `notebooks/` - Jupyter notebooks
- `inputs/` - Input data files
- `outputs/` - Output data files
- `scripts/` - Setup and utility scripts

## Configuration Files

- `pyproject.toml` - Python dependencies and ruff configuration
- `.claude.json` - MCP server configurations
- `.textlintrc.json` - Japanese text linting rules
- `AGENTS.md` - OpenAI Codex レビュワー設定

## MCP Servers Available

| Server | Purpose |
|--------|---------|
| textlint | Japanese text proofreading |
| serena | Semantic code analysis (LSP) |
| context7 | Latest library documentation lookup |
| playwright | Browser automation |
| sequential-thinking | Complex task decomposition |

## Code Style

- Python 3.10-3.11 compatible
- Line length: 88 characters
- Uses ruff for linting and formatting
- Quote style: double quotes
- Indent: 4 spaces

## GPU Execution Policy

Python script execution is automatically serialized via `flock` by a PreToolUse hook.

**Automatic serialization (hook):**
- `uv run python *.py` and `python *.py` are automatically wrapped with flock
- GPU launchers like `torchrun`, `accelerate launch` are also detected
- CPU-only tools (ruff, pip, uv sync, pytest, etc.) are NOT serialized (parallel OK)

**Explicit serialization (for cases not caught by the hook):**
- Use `run_gpu <command>` to run any command with GPU lock
- Example: `run_gpu uv run python src/my_script.py`

**Rules:**
- Always execute Python scripts via `uv run python`
- Check GPU lock status: `flock -n /var/lock/gpu.lock echo "GPU: 空き" || echo "GPU: 使用中"`
- Logs are recorded in `.gpu_logs/gpu_queue.log`

## Codex レビュワー

OpenAI Codex CLI がレビュー専用ツールとしてインストール済み。ユーザーからレビュー依頼があった場合に使用する。

**レビュー実行（推奨）:**
```bash
bash scripts/codex-review.sh              # main との差分をレビュー
bash scripts/codex-review.sh develop       # 任意のブランチとの差分をレビュー
bash scripts/codex-review.sh --file <path> # 特定ファイルをレビュー
```

**直接実行:**
```bash
# diff を渡してレビュー
git diff main | codex exec --sandbox read-only --output-last-message "このdiffをレビューして"

# ファイルを渡してレビュー
cat src/train.py | codex exec --sandbox read-only --output-last-message "このコードをレビューして"

# 対話モードで調査
codex
```

**ルール:**
- Codex はレビュー専用。ファイル変更は Claude Code が行う
- `--sandbox read-only` を必ず使用すること
- レビュー結果は日本語で要約してユーザーに報告すること

## Permission Policy

Development commands are auto-approved via `Bash(*)`. The following are explicitly DENIED:

| Category | Denied Patterns |
|----------|----------------|
| Filesystem destruction | `rm -rf /`, `dd`, `mkfs` |
| System control | `shutdown`, `reboot`, `halt`, `poweroff` |
| Privilege escalation | `sudo` |
| Destructive git | `push --force`, `push -f`, `reset --hard` |
| Docker destruction | `docker rm`, `docker system prune`, `docker rmi` |
| Network commands | `curl`, `wget`（requires confirmation each time） |

MCP tools (textlint, serena, context7, playwright, sequential-thinking) are also auto-approved.
