"""Persian text handling: normalization, digits, bidi, validation.

Typical use:

    from st_common.persian import normalize, to_persian_digits, display_text

    title = normalize(raw_title)                 # canonical storage form
    key = search_key(raw_title)                  # for dedupe / search
    msg = display_text(title)                    # safe for an RTL chat client
"""

from .bidi import display_text, first_strong_direction, isolate_ltr_runs, rtl_paragraphs, strip_bidi_controls
from .digits import (
    format_number_fa,
    format_price_toman,
    parse_amount_toman,
    parse_number,
    to_latin_digits,
    to_persian_digits,
)
from .normalize import normalize, search_key
from .validate import detect_mojibake, is_valid_persian, script_ratio, validate_persian

__all__ = [
    "detect_mojibake",
    "display_text",
    "first_strong_direction",
    "format_number_fa",
    "format_price_toman",
    "is_valid_persian",
    "isolate_ltr_runs",
    "normalize",
    "parse_amount_toman",
    "parse_number",
    "rtl_paragraphs",
    "script_ratio",
    "search_key",
    "strip_bidi_controls",
    "to_latin_digits",
    "to_persian_digits",
    "validate_persian",
]
