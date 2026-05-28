# Phase2 ASR評価 実装指示書

## 目的

Phase2 の目的は、Phase1 で作成した ASR 結果 CSV に対して、正規化済みの
参照文と認識文を比較し、WER と CER による評価結果を再現可能な形で作成する
ことである。

特に日本語では空白区切りの「単語」が自明ではないため、主指標は CER とし、
WER は形態素解析器・辞書・分割モードを固定した副指標として扱う。

Phase2 では、評価指標と評価用正規化を確定することに集中する。LLM 誤り
訂正、LLM-as-a-judge、few-shot 補正、類似検索、可視化、記事本文作成は
実装しない。

## 入力

Phase1 の ASR 結果 CSV を入力とする。

```csv
sample_id,audio_variant,audio_path,reference_text,asr_provider,asr_model,asr_text,status,error_message,processing_time_sec,raw_response_path
```

最初の対象は以下。

```text
outputs/asr_amivoice.csv
```

後続の ASR でも同じ CSV スキーマであれば同じ評価スクリプトに渡せるように
する。

metadata の正本は以下とする。

```text
inputs/common_voice_ja_test/metadata.csv
```

ASR CSV にも `reference_text` は含まれているが、評価時には metadata と
`sample_id` で照合し、不一致があれば警告またはエラーにする。これにより、
古い CSV や別データセットの結果が混入した場合に検出できる。

## 出力

Phase2 の出力は `outputs/evaluations/` にまとめる。

```text
outputs/evaluations/
  asr_eval_utterances.csv
  asr_eval_summary.csv
  asr_eval_normalization_audit.csv
  asr_eval_config.json
```

### `asr_eval_utterances.csv`

1 音声 1 ASR 結果ごとの明細を保存する。

```csv
sample_id,audio_variant,asr_provider,asr_model,reference_text_raw,asr_text_raw,reference_text_norm,asr_text_norm,reference_tokens,asr_tokens,cer_s,cer_i,cer_d,cer_n,cer,wer_s,wer_i,wer_d,wer_n,wer,status,error_message,raw_response_path
```

`status=ok` の行を評価対象にする。`status=error` は明細には残すが、CER/WER
の分子・分母には入れない。評価対象外の件数は summary に必ず出す。

### `asr_eval_summary.csv`

`asr_provider + asr_model + audio_variant + normalization_profile` 単位で集計する。

```csv
asr_provider,asr_model,audio_variant,normalization_profile,n_total,n_ok,n_error,n_excluded_empty_ref,coverage,cer_errors,cer_n,cer,wer_errors,wer_n,wer,processing_time_sec_mean
```

複数の `audio_variant` を混ぜた総合値は、個別 variant の結果を見えにくくする
ためデフォルトでは出さない。必要な場合のみ `--include-overall` で作成する。

### `asr_eval_normalization_audit.csv`

正規化で文字列が大きく変わった例を確認するための監査用 CSV。

```csv
sample_id,audio_variant,field,before,after,changed_reason
```

句読点、空白、全半角、ASCII 大文字小文字などの正規化が想定通りかを小さな
サンプルで確認できるようにする。

### `asr_eval_config.json`

評価の再現に必要な設定を保存する。

- 入力 CSV パス
- metadata パス
- 正規化プロファイル名
- tokenizer 名
- tokenizer 分割モード
- 依存ライブラリのバージョン
- 実行日時
- 評価スクリプトの引数

## 評価指標

### CER

CER は文字単位の編集距離で評価する。

```text
CER = (S + I + D) / N
```

- `S`: 置換数
- `I`: 挿入数
- `D`: 削除数
- `N`: 参照文の正規化後文字数

日本語では単語境界が明示されないため、Phase2 の主指標は CER とする。
句読点や空白の差を取り除いたうえで、漢字、かな、カタカナ、英数字の表記差は
原則として文字誤りとして扱う。

### WER

WER は単語単位の編集距離で評価する。

```text
WER = (S + I + D) / N
```

日本語 WER は tokenizer 依存になるため、Phase2 では以下を固定する。

- tokenizer: SudachiPy
- dictionary: SudachiDict core
- split mode: `C`

`SplitMode.C` は長単位寄りの分割で、固有名詞や複合語を過度に細かく割りにくい。
ただし、WER の絶対値は tokenizer と辞書に依存するため、ASR 間の比較や同一 ASR
の noise variant 比較のための副指標として扱う。必要に応じて `SplitMode.A` の
感度分析を別ファイルで出すが、headline 指標にはしない。

## 集計方針

corpus-level の CER/WER は、文ごとの CER/WER の単純平均ではなく、全サンプルの
`S/I/D/N` を足し上げてから算出する。

```text
corpus_cer = sum(cer_s + cer_i + cer_d) / sum(cer_n)
corpus_wer = sum(wer_s + wer_i + wer_d) / sum(wer_n)
```

理由は、短い発話と長い発話を同じ重みで平均すると、短文の 1 文字誤りが全体に
過大な影響を与えるためである。utterance-level の CER/WER は誤り分析用に保存
するが、summary の代表値には corpus-level を使う。

WER は挿入が多い場合に 1.0 を超えることがある。これは異常値ではなく、参照語数
を分母にした編集距離であることによる。

## 正規化方針

正規化は参照文と ASR 認識文に完全に同じ関数を適用する。ASR 側だけ、または
参照文側だけに有利な変換は行わない。

### headline profile: `ja_surface_v1`

Phase2 の正式比較には `ja_surface_v1` を使う。

この profile は、音声から直接決まりにくい整形差を取り除きつつ、漢字・かな・
カタカナ・外来語表記などの表記責任は残す。

処理順序:

1. Unicode NFKC 正規化
2. ASCII を中心とした casefold
3. 改行、タブ、全角空白、半角空白などの空白除去
4. 句読点、引用符、括弧、疑問符、感嘆符、コロン、セミコロン、中黒、
   三点リーダなどの記号除去
5. 連続空文字の整理

残す文字:

- 漢字
- ひらがな
- カタカナ
- 長音記号 `ー`
- 繰り返し記号 `々`
- ASCII/全角由来の英数字
- アラビア数字

変換しないもの:

- 漢字を読みへ変換しない
- ひらがなとカタカナを同一視しない
- 漢数字をアラビア数字へ変換しない
- 同音異字を同一視しない
- 表記ゆれ辞書による置換は行わない

例:

| raw reference | raw hypothesis | normalized reference | normalized hypothesis |
|---|---|---|---|
| `挑戦という言葉を忘れた` | `挑戦という言葉を忘れた。` | `挑戦という言葉を忘れた` | `挑戦という言葉を忘れた` |
| `空が青いです。` | `空が多いです。` | `空が青いです` | `空が多いです` |
| `僕は車の中を見ながら、いるかな？` | `僕は車の中を見ながら、いるかなと、` | `僕は車の中を見ながらいるかな` | `僕は車の中を見ながらいるかなと` |
| `ふぇあ` | `フェア。` | `ふぇあ` | `フェア` |
| `二十面相` | `20瞑想` | `二十面相` | `20瞑想` |

最後の 2 例は、音としては近いが表記差が残る。これを headline 指標で誤りとして
数えるのは、ASR 出力を「正規化された文字起こし」として評価するためである。

### sensitivity profile: `ja_kana_lenient_v1`

`ja_surface_v1` に加えて、ひらがな・カタカナの差だけを無視した補助指標を作る。
これは Common Voice の参照文に、かな表記とカタカナ表記の揺れが含まれるためで
ある。

追加処理:

- カタカナをひらがなへ変換する

この profile は headline には使わない。`ふぇあ` と `フェア` のような表記差が
全体 CER にどれだけ影響しているかを把握するための補助指標とする。

### optional profile: `ja_number_lenient_v1`

数字表記の影響を調べるため、漢数字とアラビア数字の揺れを限定的に吸収する
profile を後続で検討する。

候補:

- `一二三四五六七八九十百千万億兆零〇` の連続を数値表現として変換
- アラビア数字は NFKC 後の ASCII 数字に統一
- 変換対象は単独または助数詞・名詞に隣接する数値表現に限定

ただし、日本語の漢数字は固有名詞や慣用表現にも含まれるため、Phase2 初期実装の
headline には入れない。導入する場合は audit CSV で変換例を必ず確認する。

## 評価対象外の扱い

以下は評価対象から除外し、summary に件数を出す。

- `status != ok`
- 正規化後の参照文が空になる行
- `sample_id + audio_variant + asr_provider + asr_model` が重複しており、
  `--dedupe` 方針が指定されていない行
- metadata と ASR CSV の `reference_text` が一致しない行

正規化後の ASR 認識文が空の場合は、`status=ok` であれば除外しない。参照文の
全文削除として評価する。

## 実装方針

### 追加依存

```bash
uv add jiwer sudachipy sudachidict-core
```

`pip install` は使わない。

### 追加ファイル

```text
src/
  21_evaluate_asr.py
  text_normalization.py

tests/
  test_text_normalization.py
  test_evaluate_asr.py
```

### `src/text_normalization.py`

責務:

- `normalize_ja_text(text: str, profile: str) -> str`
- `tokenize_ja_words(text: str, split_mode: str = "C") -> list[str]`
- 句読点・空白の除去ルールを 1 箇所に閉じ込める
- 正規化 profile 名とバージョンを定数として持つ

正規化は小さな関数を順番に適用する。

```python
def normalize_ja_surface_v1(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = remove_spaces(text)
    text = remove_punctuation(text)
    return text
```

句読点除去は手書きの `replace` の羅列ではなく、Unicode category `P*` を基本に
する。ただし、長音記号 `ー` は音価を持つため削除しない。

### `src/21_evaluate_asr.py`

実行例:

```bash
uv run python src/21_evaluate_asr.py \
  --asr-csv outputs/asr_amivoice.csv \
  --metadata inputs/common_voice_ja_test/metadata.csv \
  --normalization-profile ja_surface_v1 \
  --output-dir outputs/evaluations
```

複数 ASR CSV も渡せるようにする。

```bash
uv run python src/21_evaluate_asr.py \
  --asr-csv outputs/asr_amivoice.csv outputs/asr_whisper_large_v3_turbo.csv \
  --metadata inputs/common_voice_ja_test/metadata.csv \
  --normalization-profile ja_surface_v1 \
  --output-dir outputs/evaluations
```

処理:

1. metadata を読む
2. ASR CSV を読む
3. `sample_id` で metadata と照合する
4. 重複キーを検出する
5. `status=ok` の行に正規化を適用する
6. CER 用の文字列、WER 用の token list を作る
7. jiwer で alignment と S/I/D/N を取得する
8. utterance CSV を出す
9. summary CSV を出す
10. config JSON を出す

GPU は使わないため `run_gpu` は不要。

## テスト方針

### 正規化テスト

最低限、以下を固定テストにする。

```python
def test_surface_removes_punctuation() -> None:
    assert normalize_ja_text("挑戦という言葉を忘れた。", "ja_surface_v1") == (
        "挑戦という言葉を忘れた"
    )


def test_surface_keeps_long_vowel_mark() -> None:
    assert normalize_ja_text("スーパー。", "ja_surface_v1") == "スーパー"


def test_surface_does_not_merge_kana_scripts() -> None:
    assert normalize_ja_text("ふぇあ", "ja_surface_v1") != normalize_ja_text(
        "フェア",
        "ja_surface_v1",
    )


def test_kana_lenient_merges_kana_scripts() -> None:
    assert normalize_ja_text("ふぇあ", "ja_kana_lenient_v1") == normalize_ja_text(
        "フェア",
        "ja_kana_lenient_v1",
    )
```

### 指標テスト

- 完全一致なら CER/WER は 0
- 仮説が空文字なら deletion のみになる
- 参照が空文字なら評価対象外になる
- corpus summary は utterance 平均ではなく S/I/D/N の合算で計算される
- `status=error` は分母に入らないが `n_error` に入る

## 品質ゲート

Phase2 完了条件:

- `uv run python src/21_evaluate_asr.py ...` で AmiVoice CSV を評価できる
- `outputs/evaluations/asr_eval_utterances.csv` が作成される
- `outputs/evaluations/asr_eval_summary.csv` が作成される
- `status=error` と評価対象外の件数が summary に出る
- 正規化 profile と tokenizer 設定が `asr_eval_config.json` に保存される
- ruff を通す
- 正規化と集計のテストを通す

実行例:

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run pytest tests/test_text_normalization.py tests/test_evaluate_asr.py
```

## 注意点

### 参照文の表記は完全な正解ではない

Common Voice の参照文は人間向けの文章であり、句読点、かな・カタカナ、漢数字、
漢字選択などの表記ゆれを含む。したがって raw text のまま CER/WER を計算すると、
ASR の音声認識能力ではなく、整形・句読点・表記スタイルの差を過大に評価して
しまう。

このため headline では、句読点・空白・全半角・case を取り除いた
`ja_surface_v1` を使う。

### 日本語 WER は主指標にしない

日本語には空白による単語境界がない。WER を出す場合は必ず tokenizer と辞書の
バージョンを保存し、異なる tokenizer の WER と直接比較しない。

### 正規化を強くしすぎない

漢字を読みへ変換すると、同音異義語の誤りを消してしまう。

例:

```text
好意 / 行為
橋 / 箸
聞く / 訊く
```

これは ASR 出力を「文字起こし」として評価する目的から外れるため、headline
profile では行わない。必要であれば、読みベース CER は別フェーズの分析指標として
扱う。

## 参考資料

- Hugging Face Audio Course: Evaluation metrics for ASR
  - https://huggingface.co/learn/audio-course/chapter5/evaluation
- jiwer usage
  - https://jitsi.github.io/jiwer/usage/
- jiwer transforms
  - https://jitsi.github.io/jiwer/reference/transforms/
- Unicode Standard Annex #15: Unicode Normalization Forms
  - https://unicode.org/reports/tr15/
- Python `unicodedata.normalize`
  - https://docs.python.org/3.10/library/unicodedata.html
- SudachiPy API
  - https://worksapplications.github.io/sudachi.rs/python/api/sudachipy.html
