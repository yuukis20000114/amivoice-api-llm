"""日本語テキスト正規化と形態素分割."""

from __future__ import annotations

import unicodedata

from sudachipy import Dictionary, SplitMode

PROFILES = {
    "ja_surface_v1",
}

_KEEP_CHARS = {
    "ー",  # 長音記号 ー
    "々",  # 繰り返し記号 々
}

_sudachi_cache: dict[str, object] = {}


def _get_tokenizer():
    if "tok" not in _sudachi_cache:
        _sudachi_cache["tok"] = Dictionary().create()
    return _sudachi_cache["tok"]


def _remove_punctuation(text: str) -> str:
    return "".join(
        c
        for c in text
        if c in _KEEP_CHARS or not unicodedata.category(c).startswith("P")
    )


def _remove_spaces(text: str) -> str:
    return "".join(c for c in text if not c.isspace())


def _normalize_ja_surface_v1(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _remove_spaces(text)
    return _remove_punctuation(text)


def normalize_ja_text(text: str, profile: str) -> str:
    if profile == "ja_surface_v1":
        return _normalize_ja_surface_v1(text)
    msg = f"Unknown normalization profile: {profile}"
    raise ValueError(msg)


def tokenize_ja_words(text: str, split_mode: str = "C") -> list[str]:
    mode = getattr(SplitMode, split_mode)
    tok = _get_tokenizer()
    return [m.surface() for m in tok.tokenize(text, mode)]
