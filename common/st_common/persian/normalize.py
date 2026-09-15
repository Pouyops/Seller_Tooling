"""Canonical normalization for Persian text.

Two entry points:

* ``normalize(text)``: the canonical *storage/display* form. Lossless in meaning:
  fixes look-alike characters, ZWNJ (نیم‌فاصله) placement, spacing, and invisible junk.
* ``search_key(text)``: an aggressive, lossy key for dedupe and search. Spelling variants
  such as «کتاب‌ها», «كتابها» and «کتاب ها» collapse to the same key.

Design notes are in docs/decisions.md (ADR "Persian normalization").
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from . import chars as C
from .bidi import strip_bidi_controls
from .digits import to_latin_digits, to_persian_digits, unify_arabic_digits

_L = C.LETTER_RANGES
_D = C.DIACRITIC_RANGES

_PRESENTATION_RE = re.compile(f"[{C.PRESENTATION_FORM_RANGES}]")
_CHAR_TABLE = str.maketrans(C.CHAR_MAP)
_SPACE_TABLE = str.maketrans({ch: " " for ch in C.SPACE_VARIANTS})
_DIACRITICS_RE = re.compile(f"[{_D}]")
_MULTI_ZWNJ_RE = re.compile(f"{C.ZWNJ}{{2,}}")
_SPACED_ZWNJ_RE = re.compile(f"[ ]*{C.ZWNJ}[ ]*")

# «می روم» -> «می‌روم», «نمی خواهم» -> «نمی‌خواهم». Only at a word start.
_MI_PREFIX_RE = re.compile(f"(?<![{_L}{_D}{C.ZWNJ}])(ن?می)[ ]+(?=[{_L}])")
# «کتاب ها» -> «کتاب‌ها» (plural family). Redundant ZWNJs after non-joining letters are removed later.
_HA_SUFFIX_RE = re.compile(
    f"([{_L}])[ ]+(ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان)(?![{_L}{_D}])"
)
# «بزرگ ترین» -> «بزرگ‌ترین». Bare «تر» is deliberately not joined: it is also the word "wet".
_TARIN_SUFFIX_RE = re.compile(f"([{_L}])[ ]+(ترین|ترینِ)(?![{_L}{_D}])")
# «خانه ای» -> «خانه‌ای», «رفته اند» -> «رفته‌اند»: only after word-final ه.
_HEH_SUFFIX_RE = re.compile(f"(ه)[ ]+(ای|ایی|ام|ات|اش|ایم|اید|اند)(?![{_L}{_D}])")

_SPACE_BEFORE_PUNCT_RE = re.compile(f"[ ]+([{C.ARABIC_COMMA}{C.ARABIC_SEMICOLON}{C.ARABIC_QUESTION}])")
_SPACE_BEFORE_LATIN_PUNCT_RE = re.compile(f"(?<=[{_L}{_D}])[ ]+([.!:])")
_NO_SPACE_AFTER_PUNCT_RE = re.compile(
    f"([{C.ARABIC_COMMA}{C.ARABIC_SEMICOLON}{C.ARABIC_QUESTION}])(?=[{_L}A-Za-z])"
)
_OPEN_BRACKET_SPACE_RE = re.compile(r"([(\[«])[ ]+")
_CLOSE_BRACKET_SPACE_RE = re.compile(r"[ ]+([)\]»])")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")

# ASCII punctuation directly after a Persian letter becomes Persian punctuation.
_ASCII_COMMA_RE = re.compile(f"(?<=[{_L}{_D}])[ ]*,(?![0-9])")
_ASCII_QMARK_RE = re.compile(f"(?<=[{_L}{_D}])[ ]*\\?")
_ASCII_SEMI_RE = re.compile(f"(?<=[{_L}{_D}])[ ]*;")

DigitMode = Literal["unify", "persian", "latin", "keep"]


def _fold_presentation_forms(text: str) -> str:
    # NFKC only on presentation-form code points; NFKC on the whole string would also rewrite
    # things sellers mean literally («m²» -> «m2», «½» -> «1⁄2», «™» -> «TM»).
    return _PRESENTATION_RE.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)


def _fix_invisibles(text: str) -> str:
    text = text.replace(C.BOM, "").replace(C.SOFT_HYPHEN, "")
    if C.ZWSP in text:
        # Zero-width space between two Persian letters is almost always a mistyped ZWNJ.
        out = []
        for i, ch in enumerate(text):
            if ch == C.ZWSP:
                prev = text[i - 1] if i > 0 else ""
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if prev and nxt and (C.is_letter(prev) or C.is_diacritic(prev)) and C.is_letter(nxt):
                    out.append(C.ZWNJ)
                continue
            out.append(ch)
        text = "".join(out)
    return text.translate(_SPACE_TABLE)


def _clean_zwnj(text: str) -> str:
    """Keep a ZWNJ only where it changes rendering: after a joining letter and before a letter."""
    if C.ZWNJ not in text:
        return text
    text = _MULTI_ZWNJ_RE.sub(C.ZWNJ, text)
    text = _SPACED_ZWNJ_RE.sub(C.ZWNJ, text)
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch != C.ZWNJ:
            out.append(ch)
            continue
        # Look back past combining marks to the base letter.
        j = i - 1
        while j >= 0 and C.is_diacritic(text[j]):
            j -= 1
        prev_ok = j >= 0 and C.is_letter(text[j]) and text[j] not in C.NON_JOINING
        next_ok = i + 1 < n and C.is_letter(text[i + 1])
        if prev_ok and next_ok:
            out.append(ch)
    return "".join(out)


def _fix_affix_spacing(text: str) -> str:
    text = _MI_PREFIX_RE.sub(lambda m: m.group(1) + C.ZWNJ, text)
    text = _HA_SUFFIX_RE.sub(lambda m: m.group(1) + C.ZWNJ + m.group(2), text)
    text = _TARIN_SUFFIX_RE.sub(lambda m: m.group(1) + C.ZWNJ + m.group(2), text)
    text = _HEH_SUFFIX_RE.sub(lambda m: m.group(1) + C.ZWNJ + m.group(2), text)
    return text


def _fix_punctuation(text: str, persian_punctuation: bool) -> str:
    if persian_punctuation:
        text = _ASCII_COMMA_RE.sub(C.ARABIC_COMMA, text)
        text = _ASCII_QMARK_RE.sub(C.ARABIC_QUESTION, text)
        text = _ASCII_SEMI_RE.sub(C.ARABIC_SEMICOLON, text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_LATIN_PUNCT_RE.sub(r"\1", text)
    text = _NO_SPACE_AFTER_PUNCT_RE.sub(r"\1 ", text)
    text = _OPEN_BRACKET_SPACE_RE.sub(r"\1", text)
    text = _CLOSE_BRACKET_SPACE_RE.sub(r"\1", text)
    return text


def _fix_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES_RE.sub("\n\n", "\n".join(lines)).strip()


def normalize(
    text: str,
    *,
    digits: DigitMode = "unify",
    fix_spacing: bool = True,
    persian_punctuation: bool = True,
    remove_diacritics: bool = False,
    keep_tatweel: bool = False,
    strip_bidi: bool = True,
) -> str:
    """Return the canonical form of Persian (or mixed Persian/Latin) text.

    Args:
        digits: ``"unify"`` maps Arabic-Indic digits (٤٥٦) to Persian (۴۵۶) and leaves Latin
            digits alone. ``"persian"`` also converts Latin digits except inside Latin tokens
            (``iPhone 15``, ``SM-A525``, URLs). ``"latin"`` converts everything to 0-9.
            ``"keep"`` does nothing.
        fix_spacing: join common affixes with ZWNJ (می‌/نمی‌, ‌ها, ‌ترین, ه‌ای).
        persian_punctuation: ASCII , ? ; right after a Persian letter become ، ؟ ؛
        remove_diacritics: drop harakat (lossy; off by default).
        keep_tatweel: keep kashida «ـ» (decorative stretching; removed by default).
        strip_bidi: remove LRM/RLM/embedding/isolate controls (they are presentation
            artifacts and a spoofing vector; re-add at display time with ``display_text``).
    """
    if not text:
        return text
    text = _fold_presentation_forms(text)
    text = text.translate(_CHAR_TABLE)
    text = _fix_invisibles(text)
    if strip_bidi:
        text = strip_bidi_controls(text)
    if not keep_tatweel:
        text = text.replace(C.TATWEEL, "")
    if remove_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    if fix_spacing:
        text = _fix_affix_spacing(text)
    text = _clean_zwnj(text)
    text = _fix_punctuation(text, persian_punctuation)
    text = _fix_whitespace(text)
    if digits == "unify":
        text = unify_arabic_digits(text)
    elif digits == "persian":
        text = to_persian_digits(text)
    elif digits == "latin":
        text = to_latin_digits(text)
    return text


_SEARCH_FOLD = str.maketrans(
    {
        "آ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ی",
        "ة": "ه",
        "ۀ": "ه",
        "ء": None,
        C.ZWNJ: None,
    }
)


def search_key(text: str) -> str:
    """Lossy key for matching: variants of the same word map to the same string.

    Removes ZWNJ *and* the space in common affix positions, folds hamza/alef variants,
    drops diacritics, converts digits to Latin and lowercases Latin.
    """
    if not text:
        return ""
    t = normalize(text, digits="latin", remove_diacritics=True, persian_punctuation=False)
    t = t.translate(_SEARCH_FOLD)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t
