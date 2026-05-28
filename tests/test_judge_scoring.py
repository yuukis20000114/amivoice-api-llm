"""Judge スコアパースのユニットテスト."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_utils import extract_json_from_response


def _import_judge():
    import importlib

    return importlib.import_module("32_judge_correction")


def test_parse_llm_cer_response() -> None:
    judge = _import_judge()
    parsed = {
        "major_penalties": [
            {"ref": "城の", "hyp": "白い", "reason": "同音異義語"},
        ],
        "minor_penalties": [],
        "no_penalties": [
            {"ref": "。", "hyp": "", "reason": "句読点"},
        ],
        "major_count": 1,
        "minor_count": 0,
        "ref_char_count": 10,
        "llm_cer": 0.1,
    }
    result = judge.parse_llm_cer_response(parsed)
    assert result["llm_cer"] == 0.1
    assert result["major_penalty_count"] == 1
    assert result["minor_penalty_count"] == 0


def test_parse_llm_cer_response_none() -> None:
    judge = _import_judge()
    result = judge.parse_llm_cer_response(None)
    assert result["llm_cer"] == ""
    assert result["major_penalty_count"] == ""


def test_llm_cer_perfect_match() -> None:
    judge = _import_judge()
    parsed = {
        "major_penalties": [],
        "minor_penalties": [],
        "no_penalties": [],
        "major_count": 0,
        "minor_count": 0,
        "ref_char_count": 10,
        "llm_cer": 0.0,
    }
    result = judge.parse_llm_cer_response(parsed)
    assert result["llm_cer"] == 0.0
    assert result["major_penalty_count"] == 0
    assert result["minor_penalty_count"] == 0


def test_llm_cer_recompute_from_penalties() -> None:
    judge = _import_judge()
    parsed = {
        "major_penalties": [{"ref": "a", "hyp": "b", "reason": "x"}],
        "minor_penalties": [{"ref": "c", "hyp": "d", "reason": "y"}],
        "no_penalties": [],
        "major_count": 1,
        "minor_count": 1,
        "ref_char_count": 10,
    }
    result = judge.parse_llm_cer_response(parsed)
    assert result["llm_cer"] == 0.15
    assert result["major_penalty_count"] == 1
    assert result["minor_penalty_count"] == 1


def test_parse_intent_entity_response() -> None:
    judge = _import_judge()
    parsed = {
        "intent_score": 1,
        "intent_reason": "核心メッセージが維持",
        "entities_in_ref": ["東京", "2026年"],
        "entities_preserved": ["東京"],
        "entity_preservation": 0.5,
    }
    result = judge.parse_intent_entity_response(parsed)
    assert result["intent_score"] == 1
    assert result["entity_preservation"] == 0.5
    assert "東京" in result["entities_in_ref"]


def test_parse_intent_entity_response_none() -> None:
    judge = _import_judge()
    result = judge.parse_intent_entity_response(None)
    assert result["intent_score"] == ""
    assert result["entity_preservation"] == ""


def test_intent_score_binary() -> None:
    judge = _import_judge()
    for score_val in [0, 1]:
        parsed = {
            "intent_score": score_val,
            "intent_reason": "test",
            "entities_in_ref": [],
            "entities_preserved": [],
            "entity_preservation": 1.0,
        }
        result = judge.parse_intent_entity_response(parsed)
        assert result["intent_score"] in (0, 1)


def test_intent_score_clamp() -> None:
    judge = _import_judge()
    parsed = {
        "intent_score": 5,
        "intent_reason": "test",
        "entities_in_ref": [],
        "entities_preserved": [],
        "entity_preservation": 1.0,
    }
    result = judge.parse_intent_entity_response(parsed)
    assert result["intent_score"] == 1


def test_entity_preservation_range() -> None:
    judge = _import_judge()
    for val in [0.0, 0.5, 1.0]:
        parsed = {
            "intent_score": 1,
            "intent_reason": "test",
            "entities_in_ref": ["A"],
            "entities_preserved": ["A"],
            "entity_preservation": val,
        }
        result = judge.parse_intent_entity_response(parsed)
        assert 0.0 <= result["entity_preservation"] <= 1.0


def test_entity_preservation_clamp() -> None:
    judge = _import_judge()
    parsed = {
        "intent_score": 1,
        "intent_reason": "test",
        "entities_in_ref": [],
        "entities_preserved": [],
        "entity_preservation": 1.5,
    }
    result = judge.parse_intent_entity_response(parsed)
    assert result["entity_preservation"] == 1.0


def test_full_llm_cer_json_extraction() -> None:
    text = '```json\n{"major_penalties": [], "minor_penalties": [], "no_penalties": [], "major_count": 0, "minor_count": 0, "ref_char_count": 5, "llm_cer": 0.0}\n```'
    parsed = extract_json_from_response(text)
    assert parsed is not None

    judge = _import_judge()
    result = judge.parse_llm_cer_response(parsed)
    assert result["llm_cer"] == 0.0


def test_full_intent_entity_json_extraction() -> None:
    text = '{"intent_score": 1, "intent_reason": "OK", "entities_in_ref": ["東京"], "entities_preserved": ["東京"], "entity_preservation": 1.0}'
    parsed = extract_json_from_response(text)
    assert parsed is not None

    judge = _import_judge()
    result = judge.parse_intent_entity_response(parsed)
    assert result["intent_score"] == 1
    assert result["entity_preservation"] == 1.0
