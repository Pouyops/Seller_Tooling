"""Checks that generated or user-supplied text is valid, well-formed Persian.

Used as a gate on LLM output in the listing generator: we never show a seller text that fails
``validate_persian``; the generator retries or falls back instead.
"""

from __future__ import annotations

import re
import unicodedata

from . import chars as C

_ARABIC_ONLY = set("يكىةۃ") | set(C.ARABIC_INDIC_DIGITS)
# UTF-8 Persian mis-decoded as Windows-1252/Latin-1 produces pairs such as "Ø§", "Ù…", "Ú©", "Û€".
_MOJIBAKE_RE = re.compile("[ØÙÚÛ][-¿ŒœŠšŸŽžƒˆ˜–-™]")


def script_ratio(text: str) -> dict[str, float]:
    """Share of letters in Arabic script vs Latin vs other scripts (digits/punctuation ignored)."""
    counts = {"persian": 0, "latin": 0, "other": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        if C.is_letter(ch):
            counts["persian"] += 1
        elif ch.isascii():
            counts["latin"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def detect_mojibake(text: str) -> bool:
    return bool(_MOJIBAKE_RE.search(text)) or "�" in text


def validate_persian(
    text: str,
    *,
    min_persian_ratio: float = 0.6,
    max_other_ratio: float = 0.02,
    require_normalized: bool = True,
) -> list[str]:
    """Return a list of issue codes; an empty list means the text is valid.

    Codes: ``empty``, ``mojibake``, ``control_chars``, ``arabic_chars``, ``low_persian_ratio``,
    ``foreign_script``, ``not_normalized``.
    """
    from .normalize import normalize  # avoid import cycle

    issues: list[str] = []
    if not text or not text.strip():
        return ["empty"]
    if detect_mojibake(text):
        issues.append("mojibake")
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\t" for ch in text):
        issues.append("control_chars")
    if any(ch in _ARABIC_ONLY for ch in text):
        issues.append("arabic_chars")
    ratio = script_ratio(text)
    if ratio["persian"] < min_persian_ratio:
        issues.append("low_persian_ratio")
    if ratio["other"] > max_other_ratio:
        issues.append("foreign_script")  # e.g. Chinese/Cyrillic leaking from a multilingual LLM
    if require_normalized and normalize(text) != text:
        issues.append("not_normalized")
    return issues


def is_valid_persian(text: str, **kwargs) -> bool:
    return not validate_persian(text, **kwargs)
