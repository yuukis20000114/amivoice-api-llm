# AGENTS.md

Codex がこのリポジトリでコードレビューを行う際のガイダンス。

## プロジェクト概要
GPU対応Python開発環境テンプレート（Docker, NVIDIA CUDA 12.4, uv パッケージマネージャ）。
PyTorch, TensorFlow, CatBoost, XGBoost を事前構成。

## コードスタイル
- Python 3.10-3.11 互換
- 行長: 88文字（ruff設定）
- フォーマッタ/リンター: ruff
- クォートスタイル: ダブルクォート
- インデント: スペース4つ

## レビュー重点項目
- GPU メモリ管理: torch.cuda.empty_cache()、with torch.no_grad() の適切な使用
- GPU 直列化: Python スクリプトは flock による排他制御下で実行される設計
- パッケージ管理: pip ではなく uv を使用（uv add, uv sync）
- スクリプト実行: `uv run python` 経由で実行すること
- セキュリティ: ハードコードされた認証情報、unsafe な eval/exec の検出

## 出力形式
- 日本語で回答すること
- 重大度レベル付き（🔴 致命的 / 🟡 警告 / 🔵 情報）
- 修正提案はコードブロックで提示
