"""Bidirectional-text helpers for mixed Persian (RTL) / Latin (LTR) strings.

Storage form carries no bidi controls (``strip_bidi_controls``); they are added only at display
time. The two problems this solves in chat clients such as Telegram:

1. Paragraph direction comes from the *first strong character*. A Persian message that starts
   with ``iPhone`` is laid out left-to-right. ``rtl_paragraphs`` prefixes such lines with RLM.
2. Neutral characters (parentheses, ``-``, ``%``, ``:``) between an LTR run and RTL text can
   attach to the wrong side. ``isolate_ltr_runs`` brackets each Latin run.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from . import chars as C

_BIDI_RE = re.compile(f"[{re.escape(C.BIDI_CONTROLS)}]")

# A Latin run: starts with a Latin letter or digit and ends with one. Spaces, digits and light
# punctuation inside are kept together ("iPhone 15 Pro Max", "USB-C 2.0", "Wi-Fi 6E").
_LTR_RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .\-+_/#&'’]*[A-Za-z0-9]|[A-Za-z]")


def strip_bidi_controls(text: str) -> str:
    return _BIDI_RE.sub("", text) if text else text


def first_strong_direction(text: str) -> Literal["rtl", "ltr"] | None:
    for ch in text:
        d = unicodedata.bidirectional(ch)
        if d == "L":
            return "ltr"
        if d in ("R", "AL"):
            return "rtl"
    return None


def _has_rtl(text: str) -> bool:
    return any(unicodedata.bidirectional(ch) in ("R", "AL") for ch in text)


def rtl_paragraphs(text: str) -> str:
    """Prefix RLM to every line that contains RTL text but doesn't start with it."""
    lines = text.split("\n")
    out = []
    for line in lines:
        if _has_rtl(line) and first_strong_direction(line) != "rtl" and not line.startswith(C.RLM):
            line = C.RLM + line
        out.append(line)
    return "\n".join(out)


def isolate_ltr_runs(text: str, mode: Literal["lrm", "isolate"] = "lrm") -> str:
    """Bracket Latin runs inside RTL text.

    ``mode="lrm"`` puts LRM on both sides of the run (every client supports LRM).
    ``mode="isolate"`` wraps the run in LRI … PDI (cleaner, but old Telegram clients on
    Android < 4.3 draw them as boxes).
    Text without any RTL characters is returned unchanged.
    """
    if not text or not _has_rtl(text):
        return text
    left, right = (C.LRI, C.PDI) if mode == "isolate" else (C.LRM, C.LRM)

    def wrap(m: re.Match[str]) -> str:
        run = m.group()
        # Bare numbers are weak-directional and already render correctly inside RTL text.
        if not any(ch.isascii() and ch.isalpha() for ch in run):
            return run
        return left + run + right

    return _LTR_RUN_RE.sub(wrap, text)


def display_text(text: str, *, isolate: Literal["lrm", "isolate", "none"] = "lrm") -> str:
    """Prepare already-normalized text for an RTL chat UI."""
    if not text:
        return text
    t = strip_bidi_controls(text)
    if isolate != "none":
        t = isolate_ltr_runs(t, mode=isolate)
    return rtl_paragraphs(t)
