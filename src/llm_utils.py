"""Anthropic API 呼び出し・JSON 抽出・confidence 抽出ヘルパ."""

from __future__ import annotations

import json
import re
import time
from itertools import product
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client  # noqa: PLW0603
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic()
    return _async_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
def call_llm(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0,
    max_tokens: int = 4096,
    system_message: str | None = None,
) -> dict:
    """Anthropic Messages API を呼び出し、raw response を返す."""
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_message:
        kwargs["system"] = system_message

    t0 = time.monotonic()
    resp = _get_client().messages.create(**kwargs)
    elapsed = time.monotonic() - t0

    return {
        "content": resp.content[0].text,
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "elapsed_sec": round(elapsed, 3),
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
async def call_llm_async(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0,
    max_tokens: int = 4096,
    system_message: str | None = None,
) -> dict:
    """Anthropic Messages API を非同期で呼び出し、raw response を返す."""
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_message:
        kwargs["system"] = system_message

    t0 = time.monotonic()
    resp = await _get_async_client().messages.create(**kwargs)
    elapsed = time.monotonic() - t0

    return {
        "content": resp.content[0].text,
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "elapsed_sec": round(elapsed, 3),
    }


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _try_repair_truncated_json(text: str) -> dict | None:
    """閉じ括弧が欠けたJSONの修復を試みる."""
    for suffix in product(*([("]", "")] * 2 + [("}", "")] * 2)):
        try:
            return json.loads(text + "".join(suffix))
        except json.JSONDecodeError:
            continue
    return None


def extract_json_from_response(text: str) -> dict | None:
    """LLM 応答テキストから JSON オブジェクトを抽出する."""
    m = _JSON_BLOCK_RE.search(text)
    candidate = m.group(1).strip() if m else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = _try_repair_truncated_json(candidate)
    if repaired and isinstance(repaired, dict):
        return repaired

    m2 = _JSON_OBJECT_RE.search(candidate)
    if m2:
        try:
            return json.loads(m2.group())
        except json.JSONDecodeError:
            pass

    m3 = _JSON_OBJECT_RE.search(text)
    if m3:
        try:
            return json.loads(m3.group())
        except json.JSONDecodeError:
            pass

    return None


def load_amivoice_confidence(
    raw_path: Path,
    threshold: float = 0.8,
) -> list[dict]:
    """AmiVoice raw response JSON から低信頼度トークンを抽出する."""
    with raw_path.open(encoding="utf-8") as f:
        data = json.load(f)
    tokens = []
    for result in data.get("results", []):
        for t in result.get("tokens", []):
            if t["confidence"] < threshold:
                tokens.append(
                    {"token": t["written"], "confidence": t["confidence"]},
                )
    return tokens


def format_low_confidence_tokens(tokens: list[dict]) -> str:
    """低信頼度トークンをプロンプト埋め込み用にフォーマットする."""
    if not tokens:
        return "なし（全トークンが高信頼度）"
    return json.dumps(tokens, ensure_ascii=False)
