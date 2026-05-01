# Docker UV Template for GPU-accelerated Python Projects

NVIDIA GPU対応のPython開発環境を、[uv](https://github.com/astral-sh/uv)パッケージマネージャと[Claude Code](https://docs.anthropic.com/claude-code) + MCPサーバーを使って素早く構築するためのDockerテンプレートです。機械学習や深層学習のプロジェクトに最適化されています。

## 特徴

- **GPU対応**: NVIDIA CUDA 12.4 & cuDNN ベース
- **高速パッケージ管理**: [uv](https://github.com/astral-sh/uv) パッケージマネージャ採用
- **AI開発支援**: Claude Code CLI + MCPサーバー統合
- **主要MLライブラリ**: PyTorch, TensorFlow, CatBoost, XGBoostをGPU対応で事前設定
- **開発環境**: Jupyter Lab、ruff（Linter/Formatter）
- **日本語対応**: textlint MCPで技術文書の品質チェック

## 前提条件

- [Docker](https://www.docker.com/) と [Docker Compose](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- Anthropic APIアクセス（Claude Code用）: [Claude Max](https://claude.ai/settings/billing) または [API Key](https://console.anthropic.com/)

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/docker-uv-template.git
cd docker-uv-template
```

### 2. 環境変数の設定

```bash
cp .env.example .env
# 必要に応じて.envを編集（プロキシ設定など）
```

### 3. Dockerイメージのビルドと起動

```bash
# イメージをビルドしてコンテナを起動（バックグラウンド実行）
docker compose up -d --build

# コンテナ内のシェルにアクセス
docker compose exec app bash
```

### 4. Claude Code初回セットアップ

```bash
# コンテナ内で実行
bash scripts/setup-claude.sh

# Claude Code認証（ブラウザでOAuth認証）
claude
```

### 5. 動作確認

```bash
# GPU確認
python src/check_gpu.py

# MCP確認
claude mcp list

# Jupyter Lab起動（オプション）
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## ディレクトリ構造

```
.
├── Dockerfile            # Docker設定ファイル
├── docker-compose.yml    # Docker Compose設定
├── pyproject.toml        # Pythonプロジェクト設定
├── uv.lock               # uv依存関係ロックファイル
├── Makefile              # 便利なコマンド集
├── .claude.json          # MCP設定
├── .textlintrc.json      # textlint設定
├── CLAUDE.md             # Claude Code用プロジェクト説明
├── scripts/
│   └── setup-claude.sh   # 初回セットアップスクリプト
├── src/                  # ソースコード
│   ├── check_gpu.py      # GPU動作確認用スクリプト
│   └── README.md
├── notebooks/            # Jupyter Notebookファイル用
│   ├── check_gpu.ipynb   # GPU動作確認用Notebook
│   └── README.md
├── inputs/               # 入力データ用
│   └── README.md
└── outputs/              # 出力データ用
    └── README.md
```

## Claude Code + MCP

### 設定済みMCPサーバー

| MCPサーバー | 用途 | API Key |
|------------|------|---------|
| textlint | 日本語テキスト校正 | 不要 |
| serena | セマンティックコード解析（LSP統合） | 不要 |
| context7 | 最新ライブラリドキュメント取得 | 不要 |
| playwright | ブラウザ自動化・テスト | 不要 |
| sequential-thinking | 複雑タスクの分解・推論支援 | 不要 |

### MCP管理コマンド

```bash
# MCP一覧表示
claude mcp list

# Claude Code内でMCPステータス確認
/mcp
```

### MCP使用例

```bash
# Claude Codeを起動
claude

# 以下のようなプロンプトでMCPが自動的に使用される：

# textlint: 日本語校正
> このREADMEをtextlintでチェックして

# context7: 最新ドキュメント
> PyTorchの最新のDataLoaderの使い方を教えて use context7

# serena: コード解析
> このプロジェクトのcheck_gpu.pyの依存関係を分析して

# playwright: ブラウザ操作
> https://pytorch.org にアクセスして最新バージョンを確認して
```

## GPU確認方法

### 1. Pythonスクリプトによる確認

```bash
# コンテナ内で実行
python src/check_gpu.py
```

このスクリプトは以下のライブラリのGPU対応状況を確認します：
- PyTorch
- TensorFlow
- CatBoost
- XGBoost

### 2. Jupyter Notebookによる確認

Jupyter Lab環境で`notebooks/check_gpu.ipynb`を開き、実行することでより詳細な情報を視覚的に確認できます。

## パッケージの追加方法

コンテナ内で`uv`コマンドを使用してパッケージを追加します：

```bash
# パッケージ追加
uv add package_name

# 特定のバージョンを指定して追加
uv add package_name==1.0.0

# 開発依存関係として追加
uv add --dev package_name

# PyTorchのようにインデックスURLが必要な場合
uv add torch --index-url https://download.pytorch.org/whl/cu124
```

依存関係をインストール（pyproject.tomlから）：
```bash
uv sync
```

## コード品質管理

このテンプレートには[ruff](https://github.com/charliermarsh/ruff)が導入されています。

```bash
# Makefileを使用
make lint      # Lintチェック
make format    # フォーマット
make clean     # キャッシュ削除
make gpu-check # GPU確認

# または直接ruffを実行
ruff check .
ruff format .
```

## 日本語テキスト校正

textlint MCPを使用して、日本語技術文書の品質をチェックできます。

```bash
# 直接textlintを実行
npx textlint README.md

# 自動修正
npx textlint --fix README.md

# Claude Code経由（推奨）
claude
> このドキュメントをtextlintでチェックして問題点を教えて
```

## カスタマイズ

### Python依存関係
`pyproject.toml`を編集してパッケージを追加・削除

### システムパッケージ
`Dockerfile`を編集して必要なシステムパッケージを追加

### MCP設定
`.claude.json`を編集してMCPサーバーを追加・削除

### ボリュームマウント
`docker-compose.yml`を編集してボリュームマウントやポート設定を調整

## トラブルシューティング

### Claude Code認証エラー

```bash
# 認証情報をクリアして再認証
rm -rf /root/.claude/*
claude
```

### MCPサーバーが接続できない

```bash
# デバッグモードで起動
claude --mcp-debug

# 個別MCPをテスト
npx @upstash/context7-mcp --help
```

### GPUが認識されない

```bash
# コンテナ外でNVIDIA SMI確認
docker run --gpus all nvidia/cuda:12.4.1-base nvidia-smi

# コンテナ内でPyTorch確認
python -c "import torch; print(torch.cuda.is_available())"
```

### ビルドが遅い

```bash
# キャッシュを活用した再ビルド
docker compose build --no-cache  # 完全再ビルド
docker compose up -d             # キャッシュ活用
```

## ボリューム管理

認証情報やキャッシュは名前付きボリュームで永続化されています：

```bash
# ボリューム一覧
docker volume ls | grep docker-uv-template

# 認証情報のみリセット
docker volume rm docker-uv-template_claude-config

# 全ボリューム削除（注意：認証情報も削除されます）
docker compose down -v
```

## ライセンス

[MIT](LICENSE)