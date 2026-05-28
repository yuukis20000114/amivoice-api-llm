"""llm_utils と訂正スクリプトのユニットテスト."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_utils import (
    extract_json_from_response,
    format_low_confidence_tokens,
    load_amivoice_confidence,
)


@pytest.fixture
def raw_response_path(tmp_path: Path) -> Path:
    data = {
        "results": [
            {
                "tokens": [
                    {
                        "written": "挑戦",
                        "confidence": 1.0,
                        "starttime": 100,
                        "endtime": 200,
                    },
                    {
                        "written": "という",
                        "confidence": 0.6,
                        "starttime": 200,
                        "endtime": 300,
                    },
                    {
                        "written": "言葉",
                        "confidence": 0.98,
                        "starttime": 300,
                        "endtime": 400,
                    },
                    {
                        "written": "を",
                        "confidence": 0.75,
                        "starttime": 400,
                        "endtime": 500,
                    },
                ],
                "confidence": 0.88,
                "text": "挑戦という言葉を",
            },
        ],
        "text": "挑戦という言葉を",
    }
    p = tmp_path / "test_sample.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def raw_response_all_high(tmp_path: Path) -> Path:
    data = {
        "results": [
            {
                "tokens": [
                    {
                        "written": "挑戦",
                        "confidence": 1.0,
                        "starttime": 100,
                        "endtime": 200,
                    },
                    {
                        "written": "という",
                        "confidence": 0.95,
                        "starttime": 200,
                        "endtime": 300,
                    },
                ],
                "confidence": 1.0,
                "text": "挑戦という",
            },
        ],
        "text": "挑戦という",
    }
    p = tmp_path / "all_high.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_extract_low_confidence_tokens(raw_response_path: Path) -> None:
    tokens = load_amivoice_confidence(raw_response_path, threshold=0.8)
    assert len(tokens) == 2
    assert tokens[0]["token"] == "という"
    assert tokens[0]["confidence"] == 0.6
    assert tokens[1]["token"] == "を"
    assert tokens[1]["confidence"] == 0.75


def test_extract_low_confidence_tokens_all_high(raw_response_all_high: Path) -> None:
    tokens = load_amivoice_confidence(raw_response_all_high, threshold=0.8)
    assert tokens == []


def test_format_low_confidence_tokens_empty() -> None:
    result = format_low_confidence_tokens([])
    assert "なし" in result


def test_format_low_confidence_tokens_with_data() -> None:
    tokens = [{"token": "制", "confidence": 0.65}]
    result = format_low_confidence_tokens(tokens)
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["token"] == "制"


def test_extract_json_from_response_clean() -> None:
    text = '{"corrected_text": "テスト", "corrections": []}'
    result = extract_json_from_response(text)
    assert result is not None
    assert result["corrected_text"] == "テスト"
    assert result["corrections"] == []


def test_extract_json_from_response_with_markdown() -> None:
    text = '```json\n{"corrected_text": "テスト", "corrections": []}\n```'
    result = extract_json_from_response(text)
    assert result is not None
    assert result["corrected_text"] == "テスト"


def test_extract_json_from_response_with_surrounding_text() -> None:
    text = 'Here is the result:\n{"corrected_text": "修正後", "corrections": []}\nDone.'
    result = extract_json_from_response(text)
    assert result is not None
    assert result["corrected_text"] == "修正後"


def test_extract_json_from_response_malformed() -> None:
    text = "This is not JSON at all, just plain text."
    result = extract_json_from_response(text)
    assert result is None


def test_build_correction_prompt() -> None:
    import importlib

    mod = importlib.import_module("31_correct_asr_with_llm")

    asr_text = "挑戦という言葉を忘れた"
    low_conf = [{"token": "言葉", "confidence": 0.6}]

    prompt = mod._build_prompt(asr_text, low_conf, "v1")
    assert asr_text in prompt
    assert "言葉" in prompt
    assert "0.6" in prompt
    assert "JSON" in prompt


def test_build_correction_prompt_v2() -> None:
    import importlib

    mod = importlib.import_module("31_correct_asr_with_llm")

    asr_text = "レンガ制の倉庫"
    low_conf = [{"token": "制", "confidence": 0.65}]

    prompt = mod._build_prompt(asr_text, low_conf, "v2")
    assert asr_text in prompt
    assert "<asr_text>" in prompt
    assert "<guidelines>" in prompt
    assert "<examples>" in prompt
    assert "同音異義語" in prompt
    assert "corrected_text" in prompt


def test_build_correction_prompt_v2_no_low_confidence() -> None:
    import importlib

    mod = importlib.import_module("31_correct_asr_with_llm")

    prompt = mod._build_prompt("テスト", [], "v2")
    assert "なし" in prompt


def test_build_correction_prompt_unknown_version() -> None:
    import importlib

    mod = importlib.import_module("31_correct_asr_with_llm")

    with pytest.raises(ValueError, match="Unknown prompt version"):
        mod._build_prompt("テスト", [], "v99")


def test_get_system_message_v1() -> None:
    import importlib

    mod = importlib.import_module("31_correct_asr_with_llm")

    msg = mod._get_system_message("v1")
    assert "JSON" in msg


def test_get_system_message_v2() -> None:
    import importlib

    mod = importlib.import_module("31_correct_asr_with_llm")

    msg = mod._get_system_message("v2")
    assert "ASR" in msg
    assert "JSON" in msg


def test_extract_json_v2_format() -> None:
    """v2形式のJSON（correctionsなし）もパースできる."""
    text = '{"corrected_text": "修正後テキスト"}'
    result = extract_json_from_response(text)
    assert result is not None
    assert result["corrected_text"] == "修正後テキスト"
    assert "corrections" not in result
