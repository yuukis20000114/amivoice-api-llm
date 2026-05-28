# Phase1 ASR結果作成 実装指示書

## 目的

このPhase1の目的は、Common Voice日本語testデータに対して複数のASR API/モデルを実行し、後段の評価フェーズおよびLLM誤り訂正フェーズで再利用しやすいCSVを作成することである。

Phase1では、ASR結果を得ることだけに集中する。CER/WER評価、LLM補正、LLM-as-a-judge、few-shot、類似検索、DEMANDなどの環境音付与は実装しない。

## Phase1の責務

Phase1で実装するものは以下に限定する。

1. Common Voice日本語testデータの取得
2. Common Voice test音声のメタデータCSV作成
3. Common Voice test音声への白色ノイズ付与
4. ASRごとの独立実行スクリプト作成
5. ASRごとの共通形式CSV出力
6. APIレスポンスまたはモデル出力のraw保存
7. 再実行時のスキップ処理

Phase1で実装しないものは以下。

- CER/WER/SERの算出
- テキスト正規化評価
- LLM誤り訂正
- LLM-as-a-judge
- few-shot補正
- 類似誤り例検索
- DEMAND、MUSAN、FSD50Kなどの環境音付与
- 図表作成
- Zenn記事本文作成

## 基本方針

複雑な共通ランナーは作らない。各ASRに対して1つの実行スクリプトを用意し、それぞれが同じ形式のCSVを `outputs/` に出力する。

共通化してよいのは、CSV書き込み、音声ファイル列挙、既存結果スキップなどの小さな補助関数のみとする。ASR固有処理は各スクリプト内に閉じ込める。

## ディレクトリ構成

以下の構成を作成する。

```text
docs/
  phase1_asr_implementation_plan.md

inputs/
  common_voice_ja_test/
    dataset/          # HF save_to_disk で保存 (全件)
    test.tsv
    metadata.csv
  audio_variants/
    clean/
    white_snr_30/
    white_snr_20/
    white_snr_10/
    white_snr_1/

outputs/
  asr_amivoice.csv
  asr_gcp_speech.csv
  asr_aws_transcribe.csv
  asr_whisper_large_v3.csv
  asr_whisper_large_v3_turbo.csv
  asr_kotoba_whisper.csv
  asr_reazonspeech.csv
  raw_responses/
    amivoice/
    gcp_speech/
    aws_transcribe/
    whisper_large_v3/
    whisper_large_v3_turbo/
    kotoba_whisper/
    reazonspeech/

src/
  01_download_common_voice_test.py
  02_add_white_noise.py
  11_run_amivoice.py
  12_run_gcp_speech.py
  13_run_aws_transcribe.py
  14_run_whisper_large_v3.py
  15_run_whisper_large_v3_turbo.py
  16_run_kotoba_whisper.py
  17_run_reazonspeech.py
  asr_csv.py
  audio_utils.py
```

Python スクリプトはすべて `src/` に置く。`scripts/` はセットアップ用シェルスクリプト専用とし、Phase1 の Python スクリプトは置かない。共通モジュール (`asr_csv.py`, `audio_utils.py`) も `src/` 直下に置き、ASR ごとの大きな抽象化クラスは作らない。

## 入力データ

対象データセットは Common Voice Japanese のtest split 全件とする。

データセットは HuggingFace ミラー (`fixie-ai/common_voice_17_0`) から取得し、`save_to_disk` で `inputs/common_voice_ja_test/dataset/` に一括保存する。個別の clip ファイルは作らない。

```bash
uv run python src/01_download_common_voice_test.py --seed 42
```

作成する `inputs/common_voice_ja_test/metadata.csv` のスキーマは以下。

```csv
sample_id,dataset,dataset_version,split,reference_text,original_audio_path,duration_sec
```

`sample_id` は再実行しても変わらない値にする。Common Voiceの元ファイル名をもとにしてよいが、拡張子やパスに依存しすぎないようにする。

例:

```text
cv_ja_test_000001
cv_ja_test_000002
```

## 白色ノイズ付与

白色ノイズ付与は、Common Voice取得とは別スクリプトにする。

```bash
uv run python src/02_add_white_noise.py --snr-db 30 20 10 1 --seed 42
```

作成する音声variantは以下。

```text
clean
white_snr_30
white_snr_20
white_snr_10
white_snr_1
```

ここでは「SNRを10ずつ下げ、下限として1dBまで作る」という意味で、デフォルトを `30, 20, 10, 1` dB とする。後で増やしたくなった場合でも `--snr-db` の引数を変えるだけで対応できるようにする。

白色ノイズ付与の仕様:

- 入力音声はCommon Voice test音声
- 出力はASRに渡しやすいWAVに統一する
- 推奨フォーマットは 16kHz mono WAV
- ノイズ生成は固定seedで決定的にする
- clippingを避けるため、混合後に必要ならピーク正規化する
- `reference_text` は変更しない
- 音声ファイル名は `sample_id` とvariantから一意にする

出力例:

```text
inputs/audio_variants/clean/cv_ja_test_000001.wav
inputs/audio_variants/white_snr_10/cv_ja_test_000001.wav
```

白色ノイズの混合式は以下を基本にする。

```text
signal_power = mean(signal ** 2)
noise_power_target = signal_power / (10 ** (snr_db / 10))
noise = standard_normal(...)
noise = noise / rms(noise) * sqrt(noise_power_target)
mixed = signal + noise
```

無音に近い音声で `signal_power` が極端に小さい場合は、そのサンプルをスキップするか、エラーとして記録する。

## ASR対象

Phase1で想定するASRは以下。

API:

- AmiVoice
- Google Cloud Speech-to-Text
- AWS Transcribe

ローカルモデル:

- Whisper large-v3
- Whisper large-v3-turbo
- kotoba-whisper
- ReazonSpeech

ただし、すべてを一度に完成させる必要はない。まず `AmiVoice` と `Whisper large-v3-turbo` を通し、その後に他ASRを追加する。

## ASRスクリプト共通仕様

各ASRスクリプトは独立して動作する。

例:

```bash
uv run python src/11_run_amivoice.py --variants clean white_snr_30 white_snr_20 white_snr_10 white_snr_1
run_gpu uv run python src/15_run_whisper_large_v3_turbo.py --variants clean white_snr_30 white_snr_20 white_snr_10 white_snr_1
```

各スクリプトは以下を行う。

1. `inputs/common_voice_ja_test/metadata.csv` を読む
2. `inputs/audio_variants/{variant}/` から音声を読む
3. 対象ASRを実行する
4. raw responseを `outputs/raw_responses/{asr_name}/` に保存する
5. 共通形式CSVを `outputs/` に保存する
6. 既存CSVに同じ `sample_id + audio_variant` があればスキップする
7. 失敗した場合も `status=error` の行をCSVに残す

ローカルGPUモデルは必ず `run_gpu` 経由で実行する想定にする。

## ASR結果CSVの共通スキーマ

各ASRスクリプトは、以下の列を持つCSVを出力する。

```csv
sample_id,audio_variant,audio_path,reference_text,asr_provider,asr_model,asr_text,status,error_message,processing_time_sec,raw_response_path
```

列の意味:

| 列 | 意味 |
|---|---|
| `sample_id` | metadata.csvのID |
| `audio_variant` | `clean`, `white_snr_30` など |
| `audio_path` | 実際にASRへ渡した音声ファイル |
| `reference_text` | Common Voiceの正解テキスト |
| `asr_provider` | `amivoice`, `gcp`, `aws`, `openai_whisper`, `kotoba`, `reazon` など |
| `asr_model` | 実際に使ったエンジン名またはモデル名 |
| `asr_text` | ASR結果テキスト |
| `status` | `ok`, `error`, `skipped` のいずれか |
| `error_message` | エラー時の短い説明。正常時は空 |
| `processing_time_sec` | 1音声あたりの処理時間 |
| `raw_response_path` | raw response保存先。保存しない場合は空 |

後段フェーズはこのCSVだけを読めばよい。評価フェーズでは `reference_text` と `asr_text` を使う。LLMフェーズでは `asr_text` を使う。

## 各ASRスクリプトの出力先

```text
src/11_run_amivoice.py              -> outputs/asr_amivoice.csv
src/12_run_gcp_speech.py            -> outputs/asr_gcp_speech.csv
src/13_run_aws_transcribe.py        -> outputs/asr_aws_transcribe.csv
src/14_run_whisper_large_v3.py      -> outputs/asr_whisper_large_v3.csv
src/15_run_whisper_large_v3_turbo.py -> outputs/asr_whisper_large_v3_turbo.csv
src/16_run_kotoba_whisper.py        -> outputs/asr_kotoba_whisper.csv
src/17_run_reazonspeech.py          -> outputs/asr_reazonspeech.csv
```

## APIキーと環境変数

APIキーや認証情報はコードに直書きしない。必要な値は `.env` または環境変数から読む。

想定する環境変数:

```text
AMIVOICE_API_KEY=
AMIVOICE_ENGINE=-a2-ja-general

GOOGLE_APPLICATION_CREDENTIALS=
GCP_PROJECT_ID=
GCP_LOCATION=asia-southeast1
GCP_SPEECH_MODEL=chirp_3

AWS_REGION=ap-northeast-1
AWS_TRANSCRIBE_S3_BUCKET=
AWS_TRANSCRIBE_S3_PREFIX=asr-phase1/

HF_TOKEN=
```

`.env.example` に必要な変数を追記する場合も、秘密情報は入れない。

## ASRごとの注意

### AmiVoice

- 同期HTTPを基本にする
- 短音声のみ扱う
- エンジンはデフォルトで `-a2-ja-general`
- `u` にAPIキー、`d` に `grammarFileNames={engine}` を渡す
- raw JSONを保存する

### Google Cloud Speech-to-Text

- Common Voiceの短音声を扱うため、同期recognizeを基本にする
- `language_codes` は `ja-JP`
- モデルはまず `chirp_3`
- 音声は16kHz mono WAVを渡す
- raw responseをJSONで保存する

### AWS Transcribe

- AWS Transcribeは基本的にS3上の音声に対してバッチジョブを実行する
- スクリプト内で音声をS3へアップロードし、ジョブ完了を待つ
- `LanguageCode` は `ja-JP`
- `MediaFormat` は `wav`
- raw transcript JSONを保存する
- S3バケット名は環境変数から読む

### Whisper large-v3 / large-v3-turbo

- GPU利用を想定する
- 必ず `run_gpu uv run python ...` で実行できるようにする
- languageは日本語指定
- taskはtranscribe
- 温度やbeamなどは固定し、スクリプト内または引数で記録できるようにする
- 推論時は `torch.no_grad()` を使い、終了後に可能なら `torch.cuda.empty_cache()` を呼ぶ

### kotoba-whisper

- Phase1ではまず `kotoba-tech/kotoba-whisper-v2.0` を候補にする
- v2.2は話者分離や句読点など追加処理が増えるため、Phase1では避ける
- Transformers pipelineまたはモデルカード推奨の方法を使う
- GPU利用時は `run_gpu` 経由

### ReazonSpeech

- `reazonspeech-nemo-v2` を候補にする
- 公式の `reazonspeech.nemo.asr` APIを使う
- GPU利用時は `run_gpu` 経由

## 依存関係

依存関係は `pip` ではなく `uv add` で追加する。

候補:

```bash
uv add datasets soundfile librosa numpy pandas requests python-dotenv tenacity tqdm
uv add google-cloud-speech boto3
uv add transformers accelerate torchaudio faster-whisper
uv add reazonspeech
```

実際には、最初に必要なものだけ追加する。Phase1の最初の実装では Common Voice取得、白色ノイズ生成、AmiVoice、Whisper large-v3-turbo を優先する。

## 実装順序

以下の順に実装する。

1. `01_download_common_voice_test.py`
2. `02_add_white_noise.py`
3. `src/asr_csv.py`
4. `11_run_amivoice.py`
5. `15_run_whisper_large_v3_turbo.py`
6. 他ASRスクリプト

最初から全ASRを実装しない。まずAmiVoiceとWhisper large-v3-turboのCSVが同じ形式で出ることを確認する。

## 動作確認

全件で実行する。

```bash
uv run python src/01_download_common_voice_test.py --seed 42
uv run python src/02_add_white_noise.py --snr-db 30 20 10 1 --seed 42
uv run python src/11_run_amivoice.py --variants clean white_snr_30 white_snr_20 white_snr_10 white_snr_1
run_gpu uv run python src/15_run_whisper_large_v3_turbo.py --variants clean white_snr_30 white_snr_20 white_snr_10 white_snr_1
```

この時点で以下が作られていればよい。

```text
inputs/common_voice_ja_test/dataset/
inputs/common_voice_ja_test/metadata.csv
inputs/audio_variants/clean/*.wav
inputs/audio_variants/white_snr_*/*.wav
outputs/asr_amivoice.csv
outputs/asr_whisper_large_v3_turbo.csv
outputs/raw_responses/
```

## 完了条件

Phase1の完了条件は以下。

1. Common Voice日本語testの対象サンプルを固定seedで準備できる
2. clean音声と白色ノイズ付与音声を作成できる
3. SNR 30, 20, 10, 1 dB のvariantを作成できる
4. 各ASRが独立したスクリプトで実行できる
5. 各ASRのCSV列が完全に一致している
6. 出力はすべて `outputs/` 配下に保存される
7. raw responseが保存される
8. 再実行時に処理済み行をスキップできる
9. 失敗時もCSVに `status=error` として記録される

## 守るべき制約

- 日本語でコメントやREADMEを書く場合は簡潔にする
- Pythonは3.10-3.11互換
- 実行は `uv run python` 経由
- GPUモデルは `run_gpu` 経由
- APIキーや認証情報を直書きしない
- `pip install` は使わない
- CSVの列名をASRごとに変えない
- ASRスクリプト同士を密結合させない
- Phase1で評価やLLM処理を実装しない

