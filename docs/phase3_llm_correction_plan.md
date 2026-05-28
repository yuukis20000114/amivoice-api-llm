# Phase3 AmiVoice LLM誤り訂正・意味的評価 実装指示書

## 目的

Phase3の目的は、Phase1で取得したAmiVoice ASR結果に対してローカルLLMによる誤り訂正を実施し、CER改善と意味的品質をPhase2の評価基盤およびLLM-as-a-Judge手法で定量化することである。

最終的に「AmiVoice + LLM後処理は実用的である」というストーリーを、CER改善率と意味的評価スコアの両面から示す。

Phase3では、LLM誤り訂正と評価に集中する。新たなASR実行、ノイズvariant（clean以外）の訂正、環境音付与、可視化、記事本文作成は実装しない。

## 現状の整理

### AmiVoice CER実績（Phase2結果）

- AmiVoice corpus-level CER: **20.51%**（clean, ja_surface_v1正規化）
- 対象: 6,261件中6,197件がstatus=ok（64件がrejected）
- CER=0（完全一致）: 約30%
- 他ASR参考: AWS Transcribe 13.0%, GCP 17.7%, kotoba-whisper 16.6%

### マシンスペック

- GPU: NVIDIA GeForce RTX 2070 SUPER（8GB VRAM）
- CUDA: 12.4
- 推論フレームワーク: Ollama（GGUF Q4_K_M量子化）

### AmiVoice raw responseの構造

AmiVoice raw response JSONにはtoken単位のconfidenceスコア（0.0-1.0）が含まれる:

```json
{
  "results": [{
    "tokens": [
      {"written": "光", "confidence": 0.75, "starttime": 676, "endtime": 1060},
      {"written": "の", "confidence": 1.0, "starttime": 1060, "endtime": 1220}
    ],
    "confidence": 0.88,
    "text": "光の白い制服を..."
  }],
  "text": "光の白い制服を..."
}
```

このtoken-level confidenceをLLM訂正のヒントとして活用する。

## Phase3の責務

Phase3で実装するもの:

1. AmiVoice ASR結果（clean variant）に対するローカルLLM誤り訂正
2. 訂正結果のCER/WER再評価（Phase2の `21_evaluate_asr.py` を再利用）
3. LLM-as-a-Judge による意味的品質評価（LASER方式LLM-CER + Intent Score + Entity Preservation）
4. CER + 意味的評価の統合レポートCSV

Phase3で実装しないもの:

- AmiVoice以外のASR結果の訂正
- 新規ASR実行
- ノイズvariant（white_snr_*）の訂正
- ファインチューニング・LoRA学習
- API経由のLLM利用（Claude, GPT等）
- 図表作成
- Zenn記事本文作成

## LLM選定

### 選定方針

8GB VRAMの制約下で、以下の基準で選定する:

1. Q4_K_M量子化でVRAM 8GB以内に収まること
2. 日本語性能が高いこと
3. 2026年時点で入手可能な最新モデルであること
4. GGUF形式が利用可能、またはOllamaで直接利用可能であること

### 候補モデル調査結果

| モデル | パラメータ | Q4_K_M VRAM | 日本語性能 | リリース | 備考 |
|--------|-----------|-------------|-----------|---------|------|
| **Qwen3.5-9B** | 9B | ~5.7 GB | 汎用最新最強 | 2026-02 | 256Kコンテキスト、GGUF公式提供 |
| **Qwen3-Swallow-8B-RL-v0.2** | 8B | ~5.0 GB | 日本語8B以下最強 | 2026-02 | 東工大、AWQ-INT4公式提供 |
| Qwen3.6-27B | 27B | ~16 GB | 最新最強 | 2026-04 | **8GBに入らない** |
| Qwen3.6-35B-A3B | 35B(3B active) | ~21 GB | MoE | 2026-04 | **MoEでも全重み必要、8GBに入らない** |
| Qwen3.5-4B | 4B | ~2.4 GB | 軽量 | 2026-02 | VRAMに余裕あるが性能は限定的 |

### 採用モデル

**訂正・Judge共通: Qwen3.5-9B**（Q4_K_M）

- 理由: 8GB VRAMに収まる最大かつ最新のQwenモデル。Qwen3.6は小型版が未リリースのため不可。
- VRAM使用量: ~5.7 GB（KVキャッシュ + オーバーヘッドに~2.3 GB残る）
- コンテキスト長: 8Kトークンに制限（VRAM制約のため）
- GGUF: `unsloth/Qwen3.5-9B-GGUF`（Q4_K_M、HuggingFaceで120万DL以上）
- Ollama: `ollama pull qwen3.5:9b`
- 訂正とJudge評価の両方に同一モデルを使用する

### MoEモデルについての注記

Qwen3.6-35B-A3B等のMoE（Mixture of Experts）モデルは、推論時にアクティブなパラメータ数は3Bだが、ルーティングメカニズムが全エキスパートの重みにアクセスする必要があるため、**全35Bパラメータ分のVRAMが必要**である。8GB VRAMでは動作しない。

CPU offloading（llama.cppで~30 tok/s）は技術的には可能だが、6,197件の処理には非実用的な時間がかかるため採用しない。

### 自己強化バイアスについて

訂正とJudge評価に同一モデル（Qwen3.5-9B）を使用するため、自己強化バイアス（self-enhancement bias）のリスクがある。MT-Bench（NeurIPS 2023）で報告されたこの偏りに対し、以下で緩和する:

- Judge評価はASR元テキストと訂正後テキストの**両方**に対して実行する。同一モデルのバイアスは両方に等しくかかるため、**相対的な改善度**（before vs after）の比較は妥当
- LASER方式のペナルティ分類は主観的な「良い/悪い」判断ではなく、客観的な差異分類であるため、バイアスの影響が小さい
- 従来CER（モデル非依存）との相関を確認し、LLM-CERの妥当性を検証する

### 推論フレームワーク: Ollama

- セットアップが容易（Docker内で `curl -fsSL https://ollama.com/install.sh | sh`）
- REST API（`localhost:11434`）経由でPythonから呼び出し可能
- GGUF量子化を自動処理
- GPU自動検出

## 入力

Phase1のAmiVoice ASR結果CSV:

```text
outputs/asr_amivoice_clean.csv
```

スキーマ:
```csv
sample_id,audio_variant,audio_path,reference_text,asr_provider,asr_model,asr_text,status,error_message,processing_time_sec,raw_response_path
```

AmiVoice raw response JSON（token confidence利用）:
```text
outputs/raw_responses/amivoice/clean/{sample_id}.json
```

Phase2の評価結果（ベースラインCER参照用）:
```text
outputs/evaluations/asr_eval_summary.csv
outputs/evaluations/asr_eval_utterances.csv
```

metadata:
```text
inputs/common_voice_ja_test/metadata.csv
```

## 出力

Phase3の出力は `outputs/llm_correction/` にまとめる。

```text
outputs/llm_correction/
  corrected_amivoice.csv
  correction_raw/
    {sample_id}.json
  judge_scores.csv
  judge_raw/
    {sample_id}.json
  phase3_eval_utterances.csv
  phase3_eval_summary.csv
  phase3_summary.csv
  phase3_config.json
```

### `corrected_amivoice.csv`

LLM訂正後のテキストを含むCSV。

```csv
sample_id,audio_variant,reference_text,asr_text_original,corrector_model,corrected_text,correction_changed,low_confidence_tokens,prompt_template_version,processing_time_sec,status,error_message,raw_response_path
```

| 列 | 意味 |
|---|---|
| `sample_id` | metadata.csvのID |
| `audio_variant` | `clean` |
| `reference_text` | Common Voiceの正解テキスト |
| `asr_text_original` | AmiVoice元テキスト（Phase1のasr_text） |
| `corrector_model` | `qwen3.5:9b-q4_K_M` 等 |
| `corrected_text` | LLM訂正後テキスト |
| `correction_changed` | `true`/`false` — 訂正により変化したか |
| `low_confidence_tokens` | confidence < 閾値のtokenリスト（JSON配列文字列） |
| `prompt_template_version` | `v1` 等 |
| `processing_time_sec` | LLM推論時間 |
| `status` | `ok`, `error`, `skipped` |
| `error_message` | エラー時の説明 |
| `raw_response_path` | LLM raw response保存先 |

### `judge_scores.csv`

LLM-as-a-Judge による意味的評価結果。LASER方式（EMNLP 2025）とSarvam AI方式を統合。

```csv
sample_id,audio_variant,judge_model,target_type,target_text,reference_text,llm_cer,major_penalty_count,minor_penalty_count,major_penalties_json,minor_penalties_json,no_penalties_json,intent_score,intent_reason,entities_in_ref,entities_preserved,entity_preservation,processing_time_sec,status,error_message,raw_response_path
```

| 列 | 意味 | 根拠 |
|---|---|---|
| `target_type` | `asr_original` or `llm_corrected` — 評価対象 | — |
| `llm_cer` | LASER方式の意味的CER (0.0-1.0) | LASER (EMNLP 2025) |
| `major_penalty_count` | Major-Penalty件数 | LASER |
| `minor_penalty_count` | Minor-Penalty件数 | LASER |
| `major_penalties_json` | Major-Penaltyの詳細（JSON配列文字列） | LASER |
| `minor_penalties_json` | Minor-Penaltyの詳細（JSON配列文字列） | LASER |
| `no_penalties_json` | No-Penaltyの詳細（JSON配列文字列） | LASER |
| `intent_score` | 意図保持スコア (0 or 1) | Sarvam AI |
| `intent_reason` | Intent判定理由 | Sarvam AI |
| `entities_in_ref` | 正解テキスト中のエンティティ一覧 | Sarvam AI |
| `entities_preserved` | 保持されたエンティティ一覧 | Sarvam AI |
| `entity_preservation` | エンティティ保持率 (0.0-1.0) | Sarvam AI |

`target_type` を分けることで、ASR元出力と訂正後出力の両方を同一Judgeで評価し、改善度を直接比較する。

### `phase3_eval_utterances.csv` / `phase3_eval_summary.csv`

訂正後テキストに対するCER/WER再評価結果。Phase2の `21_evaluate_asr.py` と同一フォーマット。訂正後テキストを `asr_text` 列に入れたCSVを作成し、Phase2パイプラインに通す。

### `phase3_summary.csv`

CER改善と意味的評価の統合サマリ。

```csv
corrector_model,n_total,n_ok,n_skipped,n_corrected,n_unchanged,n_improved_cer,n_degraded_cer,n_equal_cer,cer_before,cer_after,cer_relative_improvement_pct,llm_cer_before,llm_cer_after,llm_cer_relative_improvement_pct,intent_score_mean_before,intent_score_mean_after,entity_preservation_mean_before,entity_preservation_mean_after
```

### `phase3_config.json`

再現に必要な全設定:
- 入力CSVパス
- モデル名・バージョン（Ollama model info）
- プロンプトテンプレート全文（訂正用・LLM-CER用・Intent/Entity用）
- 正規化プロファイル
- temperature、max_tokens等の推論パラメータ
- confidence閾値
- Ollamaバージョン
- 依存ライブラリバージョン
- 実行日時
- 処理時間統計

## LLM訂正方針

### 信頼度スコア活用型の単一パス訂正

AmiVoiceのraw responseに含まれるtoken-level confidenceを活用し、「どこが誤りやすいか」のヒントをLLMに与える。

研究文献（arXiv:2408.16180、arXiv:2505.24347）では、zero-shot/few-shotのみのLLM訂正は過剰訂正により逆にCERが悪化するケースが報告されている。これを防ぐため:

1. **信頼度情報の提供**: confidence < 閾値のtokenをマーク付きで提示
2. **抑制的な訂正指示**: 明らかな誤りのみ訂正し、確信がなければ元のまま残す指示を明示
3. **few-shot例の提示**: AmiVoice特有の誤りパターン3例を含める
4. **修正不要ケースの例示**: CER=0サンプルを1例含め、「変更しない」判断を学習させる
5. **構造化出力**: 訂正テキストと訂正箇所をJSON形式で返す

### 訂正プロンプトテンプレート（v1）— 初版

```
あなたは日本語音声認識(ASR)の誤り訂正の専門家です。

## タスク
以下のASR認識結果テキストを読み、明らかな誤りのみを修正してください。
確信が持てない箇所は絶対に変更しないでください。

## ASR認識結果
{asr_text}

## 低信頼度トークン（誤りの可能性が高い箇所）
{low_confidence_tokens_formatted}

## 修正ルール
1. 音声として自然な日本語になるよう修正する
2. 文意が通らない箇所を優先的に修正する
3. 低信頼度トークンを重点的に確認する
4. 表記の好みの差（漢数字vsアラビア数字、ひらがなvsカタカナ等）は修正しない
5. 確信度が低い修正は行わない（元のテキストを保持する）
6. テキストが正しいと判断した場合は、そのまま返す

## 修正例
入力: "光の白い制服を気にかかりますすぐな道をまっすぐにいきました"
低信頼度: [{"token": "制", "confidence": 0.65}, {"token": "かかり", "confidence": 0.71}]
出力: {"corrected_text": "光の城の製服を気にかかる真っ直ぐな道をまっすぐにいきました", "corrections": [{"original": "白い制", "corrected": "城の製", "reason": "同音異義語の取り違え"}]}

入力: "挑戦という言葉を忘れた"
低信頼度: []
出力: {"corrected_text": "挑戦という言葉を忘れた", "corrections": []}

## 出力形式
以下のJSON形式のみで出力してください。説明文は不要です:
{"corrected_text": "修正後のテキスト", "corrections": [{"original": "元", "corrected": "修正後", "reason": "理由"}]}
```

### 訂正プロンプトテンプレート（v2）— 改善版

v1の問題点（過度に保守的で文脈的に明白な誤りも見逃す、corrections配列の生成負荷、few-shot例の品質）を学術・実務知見に基づき改善。

**v1→v2の主な変更点:**

| 変更 | 根拠 |
|------|------|
| `corrections`配列を廃止、`corrected_text`のみに | "Let Me Speak Freely" (2024): JSON構造の複雑化はLLM推論品質を10-15%劣化させる |
| XMLタグでセクション分離 | Anthropic推奨のプロンプト構造。Claude向け最適化 |
| 「絶対に変更しないでください」→「確信が持てない箇所は元のテキストを保持する」 | v1では文脈的に明白な誤り（「トヨタ自動車で話している」等）も見逃していた |
| ASR誤りのドメイン知識を注入 | arXiv:2408.16180: 音声的手がかりの明示が精度向上に寄与 |
| 「修正すべきもの/しないもの」の二分構造 | Cybozu/NTT Communications: 除外事項の明示が効果的 |
| 壊滅的崩壊テキストへの方針を明示 | v1では復元不可能なテキストに対する指針がなかった |
| 信頼度の限界を明示 | plan.md記載: confidence=1.0でも誤りの場合がある |
| few-shot例を実データ由来の代表的パターンに置換 | NAACL 2024: エラーパターン類似性での例選定が効果的 |
| system messageにASR専門家のロール設定 | Anthropic推奨: システムプロンプトでロール定義 |

```
System: あなたは日本語ASR誤り訂正システムです。入力されたASRテキストの誤認識を修正し、JSON形式で返してください。説明や思考は不要です。

User:
<task>
以下のASR（音声認識）結果テキストの誤認識を修正してください。
</task>

<asr_text>
{asr_text}
</asr_text>

<low_confidence_tokens>
{low_confidence_tokens_formatted}
</low_confidence_tokens>

<guidelines>
ASR誤認識の大半は音の取り違えに起因します。以下を判断基準としてください。

修正すべきもの:
- 同音異義語の取り違え（例: 城→白、製→制）
- 類音語の混同や文の区切りずれ（例: 「かかる真っ直ぐ」→「かかります。すぐ」）
- 文脈上明らかに不自然な語（前後の意味から正しい語が推測できる場合）

修正しないもの:
- 句読点の有無や位置の違い
- 表記スタイルの差（漢数字/アラビア数字、ひらがな/カタカナ、送り仮名等）
- 確信が持てない箇所（元のテキストを保持する）
- 元テキストが大幅に崩壊しており正しい文を推測できない箇所

信頼度スコアは参考情報です。高信頼度(1.0)でも誤りの場合があり、低信頼度でも正しい場合があります。最終判断は文脈の自然さで行ってください。
</guidelines>

<examples>
入力: "レンガ制の倉庫も海にかかります。すぐな橋も全てが瓦礫とかしていた"
低信頼度: [{"token": "制", "confidence": 0.82}, {"token": "ます", "confidence": 0.65}, {"token": "か", "confidence": 0.70}]
出力: {"corrected_text": "レンガ製の倉庫も海にかかる真っ直ぐな橋も全てが瓦礫と化していた"}

入力: "疲労回復には十分な睡眠が必要だ。"
低信頼度: [{"token": "に", "confidence": 0.70}]
出力: {"corrected_text": "疲労回復には十分な睡眠が必要だ。"}
</examples>

JSON形式のみで出力: {"corrected_text": "修正後のテキスト"}
```

### Few-shot例の選定方針

v2ではASR実データから代表的なエラーパターン2例を選定:

1. **同音異義語＋文区切り誤り（修正例）**: 制→製、ます。すぐ→真っ直ぐ、かし→化し — 音声的に説明可能な修正を複数含む
2. **修正不要（表記差のみ）**: 句読点位置の差異のみで内容は正しい — 変更しない判断を示す

v1の問題だった「不正確なfew-shot例」と「corrections配列の推論負荷」を排除。

### 信頼度スコアの抽出

```python
def extract_low_confidence_tokens(raw_response_path: str, threshold: float = 0.8) -> list[dict]:
    with open(raw_response_path) as f:
        data = json.load(f)
    tokens = []
    for result in data.get("results", []):
        for t in result.get("tokens", []):
            if t["confidence"] < threshold:
                tokens.append({"token": t["written"], "confidence": t["confidence"]})
    return tokens
```

閾値0.8はCLI引数で調整可能にする。

### confidence閾値の注意点

AmiVoiceのconfidenceは万能ではない。実データでは `confidence=1.0` でも誤認識の場合がある（例: 「白」confidence=1.0だが正解は「城」）。したがってconfidenceは補助的なヒントとして扱い、LLMの言語的推論に主体的な判断を委ねる。

## LLM-as-a-Judge方針

### 学術的根拠

Phase3で採用する評価手法は、以下の確立された研究に基づく。独自設計の基準は用いない。

#### 主要な先行研究

**1. LASER — LLM-based ASR Scoring and Evaluation Rubric（EMNLP 2025, arXiv:2510.07437）**

ASR評価に特化したLLM-as-a-Judgeの最も直接的な先行研究。Sarvam AI（Parulekar et al.）が提案。

- **手法**: 3段階ペナルティ分類（No-Penalty / Minor-Penalty / Major-Penalty）
  - No-Penalty (0.0): 数字表記差（「1300」vs「千三百」）、略語、音訳、固有名詞表記
  - Minor-Penalty (0.5): 意味を保持する1文字誤り、軽微な文法誤り（てにをは等）
  - Major-Penalty (1.0): 意味を変える単語置換、重大な省略・挿入、意味を変える語順変更
- **スコア**: `Score = 1 - (Total_Penalty / Reference_Word_Count)`、連続値0.0–1.0
- **検証**: Gemini 2.5 Proで**人間アノテーションとの相関94%**（ヒンディー語、771文対）。ファインチューニングしたLlama3-8Bで88.69%の単語ペア分類精度
- **プロンプト構造**: (1)トークン化・整列・分類の詳細指示、(2)各エラー種別の worked example、(3)CoT出力フォーマット

**2. Sarvam AI LLM-WER/LLM-CER（2025, GitHub公開）**

LASERの前身・関連研究。4つの評価指標を定義:

| 指標 | スケール | 定義 |
|------|---------|------|
| LLM-WER/LLM-CER | 0-100% | 意味的同等性を考慮したWER/CER。各不一致ペアに「意味的に等価か」をLLMが判定し、等価な差異をエラーから除外 |
| Intent Score | 0 or 1 | 核心メッセージの保持。1=主語・述語・目的語の意味が維持。0=主客転倒、否定反転、重要情報の欠落 |
| Entity Preservation | 0.0-1.0 | 重要エンティティ（人名・地名・日付・数値・組織名）の保持率 |

- **プロンプト構造**: 4カテゴリの等価性ルール（フォーマット等価、口語/書記等価、異表記等価、音声的縮約）。JSON入出力、2-3 few-shot例
- **リポジトリ**: https://github.com/sarvamai/llm_wer, https://github.com/sarvamai/llm_intent_entity

**3. Phukon et al.（Interspeech 2025, arXiv:2506.16528）**

ASR評価を人間判断に整合させる手法。3次元の統合スコア:

- S_NLI（重み0.40）: NLI entailment確率（RoBERTa-large、SNLI/MNLI/FEVER/ANLIでファインチューニング）
- S_S（重み0.28）: BERTScore F1（語彙的意味類似度）
- S_P（重み0.32）: Soundex + Jaro-Winkler類似度（音声的対応）
- 統合: `Score = 0.40*S_NLI + 0.28*S_S + 0.32*S_P`（全係数 p < 0.01）
- **検証**: **人間評価との相関0.890**（Speech Accessibility Project、6名アノテーター、評定者間相関0.78-0.93）

**4. 4段階ASR品質分類（arXiv:2604.21928, 2026）**

- 分類: Identical / Useful / Bad / Incomprehensible
- GPT-4.1で**人間との一致率79%**（全データ）、高合意サブセットで92-94%
- 従来WERの人間一致率は49%に留まる → LLM判定の優位性を実証

**5. G-Eval（EMNLP 2023, arXiv:2303.16634, 549+被引用）**

NLG評価におけるLLM-as-a-Judgeの標準的フレームワーク。SummEval（TACL 2021）で定義された4基準を採用:

- **Consistency（一貫性/忠実性）**: "factual alignment...contains only statements that are entailed by the source document"
- **Coherence（首尾一貫性）**: "well-structured and well-organized, not just a heap of information"
- **Fluency（流暢性）**: "no formatting problems, capitalization errors or obviously ungrammatical sentences"
- **Relevance（関連性/情報保持）**: "include only important information from the source document"
- スケール: 1-5。GPT-4 + CoT + 確率重み付きスコアリング
- **検証**: SummEvalベンチマークでSpearman相関0.514

**6. Prometheus（ICLR 2024, arXiv:2310.08491）**

カスタムルーブリックを用いるLLM-as-a-Judge。1,000種のルーブリックで学習:
- **検証**: 人間とのPearson相関**0.897**（GPT-4の0.882と同等）
- 各スコアレベルに具体的な定義を持つ1-5整数スケールが最も信頼性が高いことを実証

**7. MT-Bench / Chatbot Arena（NeurIPS 2023, arXiv:2306.05685, 961+被引用）**

LLM-as-a-Judgeパラダイムの確立論文:
- GPT-4 vs 人間: **一致率85%**（人間同士: 81%）
- 3構成: 単一回答評定(1-10)、ペアワイズ比較、参照回答付き評定
- バイアス緩和: position swap + 集約、長さバイアスへの指示、参照回答モード

### 採用する評価手法

上記の先行研究を踏まえ、Phase3では以下の2層構成で評価する。

#### 第1層: LASER方式 意味的CER（LLM-CER）

LASER（EMNLP 2025）とSarvam AI LLM-CERの手法を日本語ASRに適用する。

**概要**: 従来CERは「挑戦」と「挑戦。」の句読点差を1文字エラーと数える。LLM-CERでは、各不一致文字/トークンについてLLMが「意味的に等価か否か」を判定し、等価な差異（句読点、表記ゆれ等）をエラーから除外する。

**手法**: LASER論文の3段階ペナルティ分類を日本語に適応:

| ペナルティ | 値 | 日本語での適用 |
|-----------|-----|-------------|
| No-Penalty | 0.0 | 句読点の有無、漢数字/アラビア数字の差（「三」vs「3」）、ひらがな/カタカナの差（「ふぇあ」vs「フェア」）、送り仮名の差 |
| Minor-Penalty | 0.5 | 意味を保持する助詞の差（「は」vs「が」等）、活用形の軽微な差 |
| Major-Penalty | 1.0 | 同音異義語の取り違え（「城」vs「白」）、内容語の置換・欠落・挿入、否定の反転、主客の転倒 |

**スコア**: `LLM_CER = Total_Penalty / Reference_Char_Count`

LASER論文では `1 - Total_Penalty / N` だが、従来CERとの比較のためエラー率形式に変換する。従来CERとLLM-CERの差分が「意味を変えない表記差」の寄与分を示す。

**根拠**: LASER論文で94%の人間相関を達成。日本語特有の表記ゆれ（ひらがな/カタカナ、漢数字等）はNo-Penaltyに分類されるべきものであり、Phase2の `ja_kana_lenient_v1` 正規化で部分的に対処していた問題をより体系的に扱える。

#### 第2層: Sarvam AI方式 意味・エンティティ評価

Sarvam AI（2025）の Intent Score と Entity Preservation を日本語に適用する。

**Intent Score（意図保持スコア）**: 二値（0 or 1）

- 1: 文の核心メッセージ（主語・述語・目的語の関係）が維持されている
- 0: 主客転倒、否定反転、動作の変更など、核心的な意味が変わっている

**Entity Preservation（エンティティ保持率）**: 連続値 0.0-1.0

- `Entity_Preservation = 正しく保持されたエンティティ数 / 正解テキスト中のエンティティ数`
- 対象エンティティ: 人名、地名、組織名、日付、数値、固有名詞
- 正解テキストにエンティティがない場合はデフォルト1.0

**根拠**: Sarvam AIの評価フレームワークで定義された指標。ASR評価において「WER/CERでは測れない意味的品質」を補完する目的で設計され、LASERの基盤となった。arXiv:2601.21347（Zheng et al., 2026）でもIntent Accuracy、Slot Micro F1として同等の概念が採用されている。

### 評価手法と要件の対応

| Phase3の評価要件 | 採用手法 | 根拠論文 | 検証精度 |
|----------------|---------|---------|---------|
| CER改善の定量化 | 従来CER（Phase2パイプライン再利用） | Phase2で確立済み | — |
| 意味を変えない表記差の除外 | LASER方式 LLM-CER（3段階ペナルティ） | LASER (EMNLP 2025) | 94%人間相関 |
| 文意の漏れ・挿入の検出 | LLM-CERのMajor-Penalty分類 | LASER (EMNLP 2025) | 同上 |
| 核心的な文意の保持 | Intent Score（二値） | Sarvam AI (2025) | — |
| 固有名詞・数値の正確性 | Entity Preservation（0-1） | Sarvam AI (2025), Zheng et al. (2026) | — |

### 不採用とした手法と理由

- **G-Eval/SummEval方式（4基準1-5スケール）**: 要約評価向けに設計されており、ASR誤り訂正への直接適用は検証されていない。ASR特化のLASER方式が存在する以上、汎用フレームワークを転用する根拠が弱い。
- **Phukon et al. 統合スコア**: NLIモデル（RoBERTa-large）+ BERTScore + 音素類似度の3モデルが必要。8GB VRAMでLLMと同時に動かすのは困難。
- **4段階分類（arXiv:2604.21928）**: カテゴリカルな分類（Identical/Useful/Bad/Incomprehensible）はサマリレポートには有用だが、訂正の改善度を連続値で測れない。

### Judgeプロンプト設計

LASER論文のプロンプト構造に従い、以下の3段構成とする:

1. **詳細な分類ルール**: No-Penalty / Minor-Penalty / Major-Penaltyの日本語向け具体例
2. **Worked example**: 1つの完全な入出力例
3. **CoT出力フォーマット**: 分類結果 → ペナルティ計算 → スコア算出

#### LLM-CER Judgeプロンプトテンプレート（v1）

```
あなたは日本語ASR（音声認識）出力の品質を評価する専門家です。

## タスク
正解テキストと評価対象テキストを比較し、各差異を以下の3段階で分類してください。

## 正解テキスト
{reference_text}

## 評価対象テキスト
{target_text}

## ペナルティ分類ルール

### No-Penalty（0.0点）— 意味を変えない表記差
- 句読点の有無（「忘れた」vs「忘れた。」）
- ひらがな/カタカナの差（「ふぇあ」vs「フェア」）
- 漢数字/アラビア数字の差（「二十」vs「20」）
- 送り仮名の差（「行なう」vs「行う」）
- 全角/半角の差
- 音訳・外来語の表記差（「コンピューター」vs「コンピュータ」）

### Minor-Penalty（0.5点）— 軽微な文法的差異
- 意味を保持する助詞の差（「を」vs「は」で文意が変わらない場合）
- 活用形の軽微な差（「行った」vs「行きました」）
- フィラー・感嘆詞の有無（「えーと」等）

### Major-Penalty（1.0点）— 意味を変える差異
- 同音異義語の取り違え（「城」vs「白」、「製」vs「制」）
- 内容語の置換・欠落・挿入
- 否定の反転（「ありません」vs「あります」）
- 主語・目的語の取り違え
- 文の区切り位置の誤りによる意味変化

## 出力例

正解: "光の城の製服を着にかかる真っ直ぐな道"
評価: "光の白い制服を気にかかりますすぐな道"

分析:
- 「城の」→「白い」: 同音異義語の取り違え → Major-Penalty (1.0)
- 「製」→「制」: 同音異義語 → Major-Penalty (1.0)
- 「着に」→「気に」: 内容語の置換 → Major-Penalty (1.0)
- 「真っ直ぐ」→「ますすぐ」: 文区切り誤り → Major-Penalty (1.0)

Major: 4件 (4.0), Minor: 0件 (0.0), Reference文字数: 16
LLM_CER = 4.0 / 16 = 0.250

## 出力形式（JSONのみ）
{"major_penalties": [{"ref": "元", "hyp": "評価側", "reason": "理由"}], "minor_penalties": [...], "no_penalties": [...], "major_count": <int>, "minor_count": <int>, "ref_char_count": <int>, "llm_cer": <float>}
```

#### Intent Score + Entity Preservationプロンプトテンプレート（v1）

```
以下の正解テキストと評価対象テキストを比較し、2つの指標を評価してください。

## 正解テキスト
{reference_text}

## 評価対象テキスト
{target_text}

## 評価指標

### Intent Score（意図保持）
文の核心メッセージ（誰が・何を・どうした）が維持されているか。
- 1: 核心メッセージが維持されている（軽微な表記差・同義語の使用は許容）
- 0: 主客転倒、否定反転、動作の変更など、核心的な意味が変わっている

### Entity Preservation（エンティティ保持率）
正解テキスト中の固有名詞・数値・日付・地名・人名が正しく保持されているか。
- 保持率 = 正しく保持されたエンティティ数 / 正解中の全エンティティ数
- エンティティがない場合は1.0

## 出力形式（JSONのみ）
{"intent_score": <0 or 1>, "intent_reason": "判定理由", "entities_in_ref": ["entity1", ...], "entities_preserved": ["entity1", ...], "entity_preservation": <0.0-1.0>}
```

### Judge評価の効率化

- CER=0かつ `correction_changed=false` のサンプル（~30%）はJudge評価をスキップし、LLM-CER=0、Intent=1、Entity=1.0として記録する
- 実質的なJudge呼び出し回数: ~4,300件 × 2指標セット × 2（before/after）= ~17,200件
- ただしLLM-CERとIntent/Entityを1回のプロンプトに統合することで半減可能（~8,600件）
- Qwen3.5-9Bの推論速度（RTX 2070 SUPER、Q4_K_M）: ~40-60 tok/s → 1件あたり約5-10秒 → 全件で約12-24時間

## 評価の統合方針

### 成功指標

1. **CER改善**: 訂正後corpus-level CERがベースライン（20.51%）より低下
2. **LLM-CER改善**: LASER方式の意味的CERが訂正前より低下
3. **意図保持**: Intent Scoreが訂正前以上を維持（意味を壊していないことの確認）
4. **過剰訂正の抑制**: CERが悪化したサンプルの割合が全体の20%以下
5. **ストーリー**: 訂正後CERが他ASR（AWS 13.0%、GCP 17.7%等）と比較可能なレベルに到達すれば最善。到達しなくても、LLM-CERの改善やIntent Score改善により意味的品質の向上を示す

### 診断パターン

| CER | LLM-CER | Intent | 診断 |
|-----|---------|--------|------|
| CER改善 + LLM-CER改善 + Intent=1 | 理想: 表記・意味ともに改善 |
| CER悪化 + LLM-CER改善 + Intent=1 | 表記は変わったが意味は改善 — AmiVoice+LLMの価値を示す |
| CER改善 + LLM-CER悪化 + Intent=0 | 表面的に近づいたが意味が変わった — 過剰訂正 |
| CER悪化 + LLM-CER悪化 + Intent=0 | 完全な過剰訂正 — **アラート** |

### 従来CERとLLM-CERの相関検証

LLM-CERの妥当性を確認するため、以下の検証を行う:

- 従来CER=0のサンプルのLLM-CERが全て0であることを確認
- 従来CERとLLM-CERのSpearman順位相関を計算（正の相関が期待される）
- LLM-CER < 従来CER となるサンプルの差異が実際に意味を変えない表記差であることを抜き出し確認

### CER再評価

Phase2の `21_evaluate_asr.py` をそのまま再利用する。訂正後テキストを `asr_text` 列に入れたCSVを作成し、同一の正規化（`ja_surface_v1`）と評価パイプラインに通す。

```bash
uv run python src/21_evaluate_asr.py \
  --asr-csv outputs/llm_correction/corrected_for_eval.csv \
  --metadata inputs/common_voice_ja_test/metadata.csv \
  --normalization-profile ja_surface_v1 \
  --output-dir outputs/llm_correction
```

`corrected_for_eval.csv` はPhase1のASR CSVと同一スキーマ。`asr_text` 列に訂正後テキストを入れ、`asr_provider=qwen3.5-9b-corrected`、`asr_model=v1` とする。

## 実装方針

### 追加依存

```bash
# Ollama本体（Docker内）
curl -fsSL https://ollama.com/install.sh | sh

# モデルダウンロード
ollama pull qwen3.5:9b
```

Python追加パッケージ:
```bash
uv add ollama   # Ollama Python SDK
```

`requests` は既に依存済みだが、公式 `ollama` パッケージの方がAPI互換性が高い。

### 追加ファイル

```text
src/
  31_correct_asr_with_llm.py        # LLM誤り訂正メイン
  32_judge_correction.py            # LLM-as-a-Judge評価
  33_summarize_phase3.py            # CER+意味的評価の統合サマリ
  llm_utils.py                      # Ollama呼び出し・JSON抽出ヘルパ

scripts/
  setup_ollama.sh                   # Ollamaセットアップスクリプト

tests/
  test_llm_correction.py
  test_judge_scoring.py
```

### `src/llm_utils.py`

責務:
- `call_ollama(prompt: str, model: str, temperature: float, max_tokens: int) -> dict` — Ollama API呼び出し
- `extract_json_from_response(text: str) -> dict` — LLM応答からJSON部分を抽出（マークダウンコードブロック対応）
- `load_amivoice_confidence(raw_path: str, threshold: float) -> list[dict]` — raw responseからtoken confidence取得
- `format_low_confidence_tokens(tokens: list[dict]) -> str` — プロンプト用のフォーマット
- リトライ処理（tenacityによるexponential backoff）

### `src/31_correct_asr_with_llm.py`

実行例:

```bash
uv run python src/31_correct_asr_with_llm.py \
  --asr-csv outputs/asr_amivoice_clean.csv \
  --raw-response-dir outputs/raw_responses/amivoice/clean \
  --model qwen3.5:9b \
  --confidence-threshold 0.8 \
  --temperature 0 \
  --output-dir outputs/llm_correction \
  --prompt-version v1
```

処理フロー:

1. ASR CSV読み込み（`status=ok` のみ対象、`status=error` は `skipped` として記録）
2. 既存出力CSVの `sample_id` でスキップ判定（中断からの再開可能）
3. 各サンプルについて:
   a. raw response JSONからlow-confidence token抽出
   b. プロンプト構築（テンプレート + few-shot + サンプル固有情報）
   c. Ollama API呼び出し（`temperature=0` で再現性確保）
   d. 応答パース（JSON抽出、パース失敗時はraw textをcorrected_textとして使用）
   e. 結果CSVに追記
   f. raw response JSON保存
4. tqdmで進捗表示
5. 終了時にサマリ出力（処理件数、変更件数、エラー件数）

GPUロック: `flock` による自動直列化がhookで適用されるため、明示的なロック処理は不要。ただし、Ollamaサーバーが常駐するため、他のGPUジョブとの競合に注意。Ollama実行中は他のGPUモデル推論を行わないこと。

### `src/32_judge_correction.py`

実行例:

```bash
uv run python src/32_judge_correction.py \
  --corrected-csv outputs/llm_correction/corrected_amivoice.csv \
  --judge-model qwen3.5:9b \
  --temperature 0 \
  --output-dir outputs/llm_correction
```

処理フロー:

1. 訂正CSVを読む
2. CER=0かつ未変更サンプルはLLM-CER=0、Intent=1、Entity=1.0として記録（Judge呼び出しスキップ）
3. 各対象サンプルについて、2つの評価を実行:
   a. ASR元テキスト vs reference → `target_type=asr_original`
   b. 訂正後テキスト vs reference → `target_type=llm_corrected`
4. 各評価について:
   a. LASER方式プロンプトでLLM-CER（3段階ペナルティ分類）を算出
   b. Intent Score + Entity Preservationプロンプトで意味的評価を実行
   c. 応答からJSON抽出、パース失敗時はエラーとして記録
5. `judge_scores.csv` に書き込み
6. 再開可能（既存sample_id + target_typeでスキップ）

訂正とJudgeは同一モデル（Qwen3.5-9B）を使用するため、モデル切り替えは不要。

### `src/33_summarize_phase3.py`

実行例:

```bash
uv run python src/33_summarize_phase3.py \
  --corrected-csv outputs/llm_correction/corrected_amivoice.csv \
  --judge-csv outputs/llm_correction/judge_scores.csv \
  --baseline-eval outputs/evaluations/asr_eval_utterances.csv \
  --output-dir outputs/llm_correction
```

処理フロー:

1. 訂正CSVを読み、Phase1 ASR CSVフォーマットに変換して `corrected_for_eval.csv` を作成
2. `21_evaluate_asr.py` と同じロジック（text_normalization + jiwer）でCERを再計算
3. ベースラインCER（Phase2）と訂正後CERを比較
4. Judge スコアの集計（target_type別の平均・中央値・分布）
5. `phase3_summary.csv` 出力
6. `phase3_config.json` 出力

## テスト方針

### `tests/test_llm_correction.py`

```python
def test_extract_low_confidence_tokens():
    """raw response JSONから低信頼度tokenを正しく抽出できる."""

def test_extract_low_confidence_tokens_all_high():
    """全token高信頼度の場合は空リストを返す."""

def test_format_low_confidence_tokens():
    """低信頼度tokenがプロンプト用にフォーマットされる."""

def test_extract_json_from_response_clean():
    """クリーンなJSON応答からパースできる."""

def test_extract_json_from_response_with_markdown():
    """```json```マークダウンで囲まれたJSONからパースできる."""

def test_extract_json_from_response_malformed():
    """不正なJSONの場合にフォールバックが機能する."""

def test_build_correction_prompt():
    """訂正プロンプトにASRテキストと低信頼度情報が含まれる."""
```

### `tests/test_judge_scoring.py`

```python
def test_parse_llm_cer_response():
    """LASER方式のJudge応答からペナルティ分類とLLM-CERを正しく抽出できる."""

def test_llm_cer_perfect_match():
    """完全一致の場合LLM-CER=0を返す."""

def test_llm_cer_punctuation_only_diff():
    """句読点のみの差異はNo-Penaltyとなり、LLM-CER=0を返す."""

def test_parse_intent_entity_response():
    """Intent Score + Entity Preservationの応答を正しくパースできる."""

def test_intent_score_binary():
    """Intent Scoreが0または1のみであることを検証."""

def test_entity_preservation_range():
    """Entity Preservationが0.0-1.0の範囲内であることを検証."""
```

## 品質ゲート

Phase3の完了条件:

1. Ollamaが起動し、qwen3.5:9bが利用可能
2. `31_correct_asr_with_llm.py` でAmiVoice clean結果（6,197件）のLLM訂正が完了
3. `corrected_amivoice.csv` が作成され、全行に `status=ok` or `skipped`
4. 訂正後テキストのCER再評価が実行でき、`phase3_eval_utterances.csv` が作成される
5. `32_judge_correction.py` でJudge評価が完了し、`judge_scores.csv` が作成される
6. `33_summarize_phase3.py` で統合サマリが作成される
7. CER改善、LLM-CER改善、またはIntent Score改善のいずれかが確認できる
8. `uv run ruff check src tests` を通す
9. `uv run ruff format src tests` を通す
10. `uv run pytest tests/test_llm_correction.py tests/test_judge_scoring.py` を通す

## 実装順序

1. `scripts/setup_ollama.sh` — Ollamaセットアップ + モデルダウンロード
2. `src/llm_utils.py` — Ollama呼び出しヘルパ
3. `src/31_correct_asr_with_llm.py` — LLM訂正メイン
4. CER再評価（既存 `21_evaluate_asr.py` への訂正CSVの投入）
5. `src/32_judge_correction.py` — LLM-as-a-Judge
6. `src/33_summarize_phase3.py` — 統合サマリ
7. テスト作成

ステップ1-4を先に完了させ、CER改善を確認してからJudge評価に進む。

## 注意点

### 過剰訂正リスク

研究文献が繰り返し警告している通り、LLMはASRテキストを「より自然な日本語」に書き換えようとして、元の発話にない情報を挿入する傾向がある（hallucination）。8-9Bクラスのモデルではこの傾向がより顕著になる可能性がある。

対策:
1. プロンプトに「確信がなければ変更しない」を強調
2. `corrections` 配列で各修正の理由を出力させ、事後検証を可能にする
3. few-shotに「修正不要」ケースを含める
4. `temperature=0` で出力のランダム性を最小化
5. CER悪化サンプルの分析をサマリに含める

### ローカルLLMの品質限界

8-9BクラスのLLMは、大規模モデル（70B+やAPI LLM）と比較して以下の点で劣る:
- 同音異義語の文脈判断精度
- 長文における意味の一貫性保持
- 構造化出力（JSON）の信頼性

これらの限界を前提とし、JSONパース失敗時のフォールバック処理を手厚く実装する。

### GPUメモリ管理

RTX 2070 SUPER（8GB）では、Ollamaの常駐プロセスがVRAMを占有する。

- Qwen3.5-9B（Q4_K_M、~5.7GB）が常駐し、訂正とJudge評価の両方に使用する
- 他のGPU処理（Whisper推論等）はOllama停止後に行う

### AmiVoice固有の注意

- `status=error` の64件（confidence閾値以下でrejected）はLLM訂正対象外
- AmiVoiceは句読点を出力する（「忘れた。」）が、Phase2の正規化（ja_surface_v1）で句読点は除去される
- LLM訂正後も同一正規化を適用してCERを計算する

### 推論時間の見積もり

RTX 2070 SUPERでのQ4_K_M推論速度（参考値）:
- 9B Q4: ~40-60 tok/s
- 8B Q4: ~50-70 tok/s

1サンプルあたりの推論:
- 訂正: 入力~300 tok + 出力~200 tok → ~5-8秒
- Judge: 入力~400 tok + 出力~300 tok → ~8-12秒

全件処理時間:
- 訂正: 6,197件 × ~7秒 = ~12時間
- Judge: ~8,600件 × ~10秒 = ~24時間
- 合計: ~36時間（中断・再開可能な設計で対応）

### 再現性の確保

- Ollama `temperature=0` で出力のランダム性を排除
- `seed` パラメータが利用可能な場合は固定値（42）を使用
- raw responseをJSON保存し、再解析可能にする
- プロンプトテンプレート全文を `phase3_config.json` に記録する
- Ollamaバージョン、モデルダイジェスト（sha256）を記録する

## 参考資料

### ASR誤り訂正

- Benchmarking Japanese Speech Recognition on ASR-LLM Setups with Multi-Pass Augmented Generative Error Correction
  - https://arxiv.org/abs/2408.16180
- Three-stage ASR error correction framework (pre-detection + CoT correction + verification)
  - https://arxiv.org/abs/2505.24347
- LLM-based rare word ASR correction with phonetic context
  - https://arxiv.org/abs/2505.17410
- ASR Error Correction using LLMs
  - https://arxiv.org/abs/2409.09554

### LLM-as-a-Judge / 評価手法

- LASER: LLM-based ASR Scoring and Evaluation Rubric (EMNLP 2025) — **Phase3のLLM-CER手法の直接的根拠**
  - https://arxiv.org/abs/2510.07437
  - 94%人間相関。3段階ペナルティ分類（No/Minor/Major）によるASR評価
- Sarvam AI: Evaluating Indian Language ASR Beyond WER — **Phase3のIntent Score / Entity Preservation手法の根拠**
  - https://www.sarvam.ai/blogs/evaluating-indian-language-asr
  - https://github.com/sarvamai/llm_wer
  - https://github.com/sarvamai/llm_intent_entity
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (NeurIPS 2023, 961+被引用)
  - https://arxiv.org/abs/2306.05685
  - LLM-as-a-Judgeパラダイムの確立。GPT-4 vs 人間一致率85%
- G-Eval: NLG Evaluation using GPT-4 with Chain-of-Thought (EMNLP 2023, 549+被引用)
  - https://arxiv.org/abs/2303.16634
  - SummEval基準（Consistency/Coherence/Fluency/Relevance）のLLM評価
- Prometheus: Inducing Fine-grained Evaluation Capability in Language Models (ICLR 2024)
  - https://arxiv.org/abs/2310.08491
  - カスタムルーブリックのLLM評価。人間とのPearson相関0.897
- 4-tier ASR Quality Classification (arXiv:2604.21928, 2026)
  - Identical/Useful/Bad/Incomprehensible分類。GPT-4.1で人間一致率79%（WERは49%）
- Aligning ASR Evaluation with Human and LLM Judgments (Interspeech 2025)
  - https://arxiv.org/abs/2506.16528
  - NLI + BERTScore + 音素類似度の統合。人間相関0.890
- SummEval: Re-evaluating Summarization Evaluation (TACL 2021)
  - NLG評価基準（Consistency/Coherence/Fluency/Relevance）の基盤論文
- Towards Robust Dysarthric Speech Recognition: LLM-Agent Post-ASR Correction Beyond WER (2026)
  - https://arxiv.org/abs/2601.21347
  - WER + BERTScore + MENLI + Intent Accuracy + Slot F1の多層評価

### LLMモデル

- Qwen3.5 公式
  - https://qwen.ai/research
- Qwen3 Swallow (東工大)
  - https://swallow-llm.github.io/qwen3-swallow.en.html
- Qwen3-Swallow-8B-RL-v0.2 (HuggingFace)
  - https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2
- Swallow LLM Leaderboard v2
  - https://swallow-llm.github.io/swallow-leaderboard-v2.en.html

### 量子化・推論

- LLM Quantization Explained: GGUF vs GPTQ vs AWQ
  - https://tensorrigs.com/blog/llm-quantization-guide/
- Ollama公式
  - https://ollama.com
