"""Digit conversion, number parsing and price formatting for Persian/Arabic/Latin numerals."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from . import chars as C

_TO_LATIN = str.maketrans(
    {**dict(zip(C.PERSIAN_DIGITS, C.LATIN_DIGITS)), **dict(zip(C.ARABIC_INDIC_DIGITS, C.LATIN_DIGITS)),
     **dict(zip(C.FULLWIDTH_DIGITS, C.LATIN_DIGITS))}
)
_ARABIC_TO_PERSIAN = str.maketrans(dict(zip(C.ARABIC_INDIC_DIGITS, C.PERSIAN_DIGITS)))
_LATIN_TO_PERSIAN = str.maketrans(dict(zip(C.LATIN_DIGITS, C.PERSIAN_DIGITS)))

_ANY_DIGIT = "0-9٠-٩۰-۹"

# Spans whose digits must stay Latin when converting to Persian digits.
_PROTECTED_RE = re.compile(
    r"(?:https?://|www\.)\S+"  # URLs
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"  # e-mails
    r"|[A-Za-z][A-Za-z0-9_\-]*[0-9][A-Za-z0-9_\-]*"  # A52, SM-G991B, iPhone15, v2
    r"|[0-9]+[A-Za-z][A-Za-z0-9_\-]*"  # 5G, 256GB, 4K
)
_DIGIT_RUN_RE = re.compile(f"[{_ANY_DIGIT}]+")
# Characters skipped when looking for the nearest "strong" neighbour of a digit run.
_NEUTRAL = set(" \t-–.,:/()%+×x٫٬،")


def to_latin_digits(text: str) -> str:
    """Convert Persian, Arabic-Indic and full-width digits to 0-9."""
    return text.translate(_TO_LATIN) if text else text


def unify_arabic_digits(text: str) -> str:
    """Arabic-Indic ٠-٩ -> Persian ۰-۹. Mixing them in Persian text renders ۴/٤ ۵/٥ ۶/٦ differently."""
    return text.translate(_ARABIC_TO_PERSIAN) if text else text


def _neighbour_is_latin(text: str, start: int, end: int) -> bool:
    i = start - 1
    while i >= 0 and (text[i] in _NEUTRAL or text[i].isdigit()):
        i -= 1
    if i >= 0 and text[i].isascii() and text[i].isalpha():
        return True
    j = end
    while j < len(text) and (text[j] in _NEUTRAL or text[j].isdigit()):
        j += 1
    return j < len(text) and text[j].isascii() and text[j].isalpha()


def to_persian_digits(text: str, *, protect_ltr: bool = True) -> str:
    """Convert digits to Persian, leaving digits that belong to Latin text untouched.

    ``"گوشی iPhone 15 Pro با 256GB"`` keeps ``15`` and ``256GB`` Latin, while
    ``"قیمت 1200 تومان"`` becomes ``"قیمت ۱۲۰۰ تومان"``.
    """
    if not text:
        return text
    if not protect_ltr:
        return unify_arabic_digits(text).translate(_LATIN_TO_PERSIAN)
    protected = [m.span() for m in _PROTECTED_RE.finditer(text)]

    def in_protected(pos: int) -> bool:
        return any(a <= pos < b for a, b in protected)

    out = []
    last = 0
    for m in _DIGIT_RUN_RE.finditer(text):
        s, e = m.span()
        out.append(text[last:s])
        run = m.group()
        if in_protected(s) or _neighbour_is_latin(text, s, e):
            out.append(run)
        else:
            out.append(unify_arabic_digits(run).translate(_LATIN_TO_PERSIAN))
        last = e
    out.append(text[last:])
    return "".join(out)


_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_GROUPED_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_SLASH_DECIMAL_RE = re.compile(r"^[+-]?\d+/\d+$")


def parse_number(text: str) -> Decimal | None:
    """Parse a number written with any digit set and Persian/Latin separators.

    Accepts ``۱۲٬۵۰۰``, ``12,500``, ``۱٫۵``, ``1.5``, and ``۲/۵`` (in Iran «/» is the everyday
    decimal separator: «۲/۵ کیلو» means 2.5 kg). Returns ``None`` if it isn't a single number.
    """
    if text is None:
        return None
    t = to_latin_digits(str(text)).strip()
    t = t.replace(C.PERSIAN_THOUSANDS_SEP, ",").replace(C.PERSIAN_DECIMAL_SEP, ".").replace(C.ARABIC_COMMA, ",")
    t = t.replace("−", "-").replace(" ", "")
    if _SLASH_DECIMAL_RE.match(t):
        t = t.replace("/", ".")
    if _GROUPED_RE.match(t):
        t = t.replace(",", "")
    if not _NUMBER_RE.match(t):
        return None
    try:
        return Decimal(t)
    except InvalidOperation:  # pragma: no cover - regex guarantees validity
        return None


_MULTIPLIERS = {"هزار": Decimal(1_000), "میلیون": Decimal(1_000_000), "ملیون": Decimal(1_000_000),
                "میلیارد": Decimal(1_000_000_000)}
_AMOUNT_RE = re.compile(
    rf"(?P<num>[{_ANY_DIGIT}][{_ANY_DIGIT},.٫٬،/]*)\s*(?P<mult>هزار|میلیون|ملیون|میلیارد)?\s*(?P<cur>تومان|تومن|ریال)?"
)


def parse_amount_toman(text: str) -> int | None:
    """Parse a price phrase to an integer amount in toman.

    ``"۲۵۰ هزار تومان"`` -> 250000; ``"۱٫۵ میلیون"`` -> 1500000; ``"12,500,000 ریال"`` -> 1250000.
    Without a currency word, the amount is assumed to be toman (the unit sellers quote in).
    Number words («دویست و پنجاه») are not supported.
    """
    if not text:
        return None
    from .normalize import normalize  # local import to avoid a cycle

    t = normalize(text, digits="keep", fix_spacing=False, persian_punctuation=False)
    m = _AMOUNT_RE.search(t)
    if not m:
        return None
    num = parse_number(m.group("num").rstrip(",.٫٬،/"))
    if num is None:
        return None
    if m.group("mult"):
        num *= _MULTIPLIERS[m.group("mult")]
    if m.group("cur") == "ریال":
        num /= 10
    return int(num.to_integral_value())


def format_number_fa(value: int | float | Decimal, decimals: int = 0) -> str:
    """``1250000`` -> ``"۱٬۲۵۰٬۰۰۰"``; ``2.5, decimals=1`` -> ``"۲٫۵"``."""
    s = f"{Decimal(str(value)):,.{decimals}f}"
    s = s.replace(",", C.PERSIAN_THOUSANDS_SEP).replace(".", C.PERSIAN_DECIMAL_SEP)
    return s.translate(_LATIN_TO_PERSIAN)


def format_price_toman(value: int) -> str:
    return f"{format_number_fa(int(value))} تومان"
