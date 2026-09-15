"""Character tables shared by the Persian text modules.

Kept explicit (no clever Unicode-property tricks) so reviewers can audit exactly what changes.
"""

ZWNJ = "‌"  # نیم‌فاصله
ZWJ = "‍"  # kept as-is: needed inside emoji sequences
ZWSP = "​"
BOM = "﻿"
SOFT_HYPHEN = "­"
TATWEEL = "ـ"
LRM = "‎"
RLM = "‏"
ALM = "؜"
LRI, RLI, FSI, PDI = "⁦", "⁧", "⁨", "⁩"
LRE, RLE, PDF, LRO, RLO = "‪", "‫", "‬", "‭", "‮"
BIDI_CONTROLS = LRM + RLM + ALM + LRE + RLE + PDF + LRO + RLO + LRI + RLI + FSI + PDI

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"
FULLWIDTH_DIGITS = "０１２３４５６７８９"

PERSIAN_DECIMAL_SEP = "٫"  # ٫
PERSIAN_THOUSANDS_SEP = "٬"  # ٬
ARABIC_COMMA = "،"  # ،
ARABIC_SEMICOLON = "؛"  # ؛
ARABIC_QUESTION = "؟"  # ؟

# Arabic-script letters (no digits, punctuation, tatweel or combining marks).
LETTER_RANGES = (
    "ء-غ"
    "ف-ي"
    "ٮ-ٯ"
    "ٱ-ۓ"
    "ە"
    "ۮ-ۯ"
    "ۺ-ۼ"
    "ۿ"
)
# Harakat and other Arabic combining marks.
DIACRITIC_RANGES = "ً-ٰٟۖ-ۜ۟-۪ۨ-ۭ"

# Letters that never connect to the following letter, so a ZWNJ after them is redundant.
NON_JOINING = frozenset("ءآأؤإادذرزژوۀةٱە")

# Canonical Persian forms for Arabic / Urdu look-alikes. Only unambiguous mappings.
CHAR_MAP = {
    "ي": "ی",  # ي Arabic yeh        -> ی
    "ى": "ی",  # ى alef maksura      -> ی
    "ے": "ی",  # ے yeh barree (Urdu) -> ی
    "ك": "ک",  # ك Arabic kaf        -> ک
    "ڪ": "ک",  # ڪ swash kaf         -> ک
    "ھ": "ه",  # ھ heh doachashmee   -> ه
    "ہ": "ه",  # ہ heh goal (Urdu)   -> ه
    "ە": "ه",  # ە ae                -> ه
}

# Horizontal whitespace variants collapsed to a plain space.
SPACE_VARIANTS = "              　"

PRESENTATION_FORM_RANGES = "ﭐ-﷿ﹰ-﻾"


def is_letter(ch: str) -> bool:
    """True for Arabic-script letters (Persian included)."""
    o = ord(ch)
    return (
        0x0621 <= o <= 0x063A
        or 0x0641 <= o <= 0x064A
        or 0x066E <= o <= 0x066F
        or 0x0671 <= o <= 0x06D3
        or o == 0x06D5
        or 0x06EE <= o <= 0x06EF
        or 0x06FA <= o <= 0x06FC
        or o == 0x06FF
    )


def is_diacritic(ch: str) -> bool:
    o = ord(ch)
    return (
        0x064B <= o <= 0x065F
        or o == 0x0670
        or 0x06D6 <= o <= 0x06DC
        or 0x06DF <= o <= 0x06E8
        or 0x06EA <= o <= 0x06ED
    )
