import pytest

from st_common.persian import normalize, search_key

ZWNJ = "‌"


# ---- look-alike characters -------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("كتاب يك", "کتاب یک"),  # Arabic kaf / yeh
        ("موسى", "موسی"),  # alef maksura
        ("ڪتاب", "کتاب"),  # swash kaf
        ("ھمه", "همه"),  # heh doachashmee (Urdu keyboards)
        ("ﺳﻼﻡ", "سلام"),  # presentation forms incl. lam-alef ligature
        ("ﮐﺘﺎﺏ", "کتاب"),  # Persian keheh presentation form
    ],
)
def test_character_mapping(raw, expected):
    assert normalize(raw) == expected


def test_does_not_rewrite_meaningful_symbols():
    # Whole-string NFKC would turn these into "m2", "1⁄2", "TM".
    assert normalize("متراژ ۲۰ m²") == "متراژ ۲۰ m²"
    assert normalize("½ کیلو") == "½ کیلو"
    assert normalize("برند Acme™") == "برند Acme™"


def test_hamza_forms_are_kept_in_storage_form():
    assert normalize("مسئله") == "مسئله"
    assert normalize("مؤسسه") == "مؤسسه"


# ---- ZWNJ (نیم‌فاصله) ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        (f"کتاب{ZWNJ}{ZWNJ}ها", f"کتاب{ZWNJ}ها"),  # repeated
        (f"کتاب {ZWNJ}ها", f"کتاب{ZWNJ}ها"),  # stray space next to ZWNJ
        (f"کتاب{ZWNJ} ها", f"کتاب{ZWNJ}ها"),
        (f"دیوار{ZWNJ}ها", "دیوارها"),  # redundant after non-joining letter
        (f"{ZWNJ}سلام{ZWNJ}", "سلام"),  # leading / trailing
        (f"کتاب{ZWNJ}abc", "کتابabc"),  # before Latin
        (f"کتاب{ZWNJ}۱۲", "کتاب۱۲"),  # before digit
        (f"می{ZWNJ}روم", f"می{ZWNJ}روم"),  # already correct
        (f"کتاب​ها", f"کتاب{ZWNJ}ها"),  # zero-width space typed instead of ZWNJ
        ("abc​def", "abcdef"),
    ],
)
def test_zwnj_cleanup(raw, expected):
    assert normalize(raw) == expected


def test_zwnj_kept_after_diacritic_on_joining_letter():
    assert normalize(f"کتابِ{ZWNJ}ها") == f"کتابِ{ZWNJ}ها"


def test_emoji_zwj_sequences_survive():
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    assert normalize(f"{family} سلام") == f"{family} سلام"


# ---- affix spacing ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("می روم", f"می{ZWNJ}روم"),
        ("نمی خواهم", f"نمی{ZWNJ}خواهم"),
        ("کتاب ها", f"کتاب{ZWNJ}ها"),
        ("کتاب های من", f"کتاب{ZWNJ}های من"),
        ("دیوار ها", "دیوارها"),
        ("بزرگ ترین", f"بزرگ{ZWNJ}ترین"),
        ("خانه ای", f"خانه{ZWNJ}ای"),
        ("رفته اند", f"رفته{ZWNJ}اند"),
    ],
)
def test_affix_spacing(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "رسمی است",  # «می» at word end is not a prefix
        "کمی آب",
        "لباس تر",  # «تر» = wet; must not be joined
        "کتاب هادی",  # «ها» followed by letters is another word
        "خرید گوشی Samsung Galaxy A54 با گارانتی ۱۸ ماهه",
    ],
)
def test_affix_spacing_leaves_non_affixes_alone(text):
    assert normalize(text) == text


def test_fix_spacing_can_be_disabled():
    assert normalize("می روم", fix_spacing=False) == "می روم"


# ---- punctuation & whitespace ----------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("سلام, خوبی?", "سلام، خوبی؟"),
        ("رنگ; سایز", "رنگ؛ سایز"),
        ("سلام ، دنیا", "سلام، دنیا"),
        ("سلام،دنیا", "سلام، دنیا"),
        ("تمام شد .", "تمام شد."),
        ("( سلام )", "(سلام)"),
        ("« سلام »", "«سلام»"),
        ("  سلام   دنیا  \n\n\n\nخط", "سلام دنیا\n\nخط"),
        ("سلام دنیا", "سلام دنیا"),
        ("سلام\r\nدنیا", "سلام\nدنیا"),
    ],
)
def test_punctuation_and_whitespace(raw, expected):
    assert normalize(raw) == expected


def test_latin_punctuation_in_latin_or_numbers_untouched():
    assert normalize("hello, world?") == "hello, world?"
    assert normalize("قیمت 1,000") == "قیمت 1,000"


# ---- tatweel, diacritics, invisible & bidi controls ------------------------------------------
def test_tatweel_removed_by_default():
    assert normalize("سلامـــت") == "سلامت"
    assert normalize("سلامـــت", keep_tatweel=True) == "سلامـــت"


def test_diacritics_optional():
    assert normalize("کِتاب") == "کِتاب"
    assert normalize("کِتاب", remove_diacritics=True) == "کتاب"


def test_invisible_junk_removed():
    assert normalize("﻿سلام") == "سلام"
    assert normalize("سلا­م") == "سلام"


def test_bidi_controls_stripped():
    assert normalize("‏سلام‎") == "سلام"
    # Trojan-source style override must not survive into storage.
    assert normalize("abc‮فارسی") == "abcفارسی"
    assert normalize("⁧سلام⁩", strip_bidi=False) == "⁧سلام⁩"


# ---- digits ----------------------------------------------------------------------------------
def test_digits_unify_default():
    assert normalize("٤٥٦") == "۴۵۶"
    assert normalize("123") == "123"


def test_digits_persian_mode_protects_latin_tokens():
    assert normalize("قیمت 1200 تومان", digits="persian") == "قیمت ۱۲۰۰ تومان"
    assert normalize("گوشی iPhone 15 Pro", digits="persian") == "گوشی iPhone 15 Pro"


def test_digits_latin_mode():
    assert normalize("کد ۱۲۳٤", digits="latin") == "کد 1234"


def test_digits_keep_mode():
    assert normalize("٤ و ۴ و 4", digits="keep") == "٤ و ۴ و 4"


# ---- general properties ----------------------------------------------------------------------
def test_empty():
    assert normalize("") == ""


CORPUS = [
    "كتاب هاي خوب",
    "می روم  به بازار ,  امروز?",
    f"کتاب{ZWNJ}{ZWNJ} ها",
    "گوشی iPhone 15 Pro Max 256GB – رنگ مشکی",
    "زعفران سرگل ۴٫۶ گرمی (بسته بندی خاتم)",
    "فرش دستباف ۱۲ متری ، کرک ، گل ابریشم",
    "‏(iPhone) مدل ٢٠٢٤",
    "ﺳﻼﻡ دنیا ـ خوش آمدید",
    "",
]


@pytest.mark.parametrize("text", CORPUS)
def test_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once


@pytest.mark.parametrize("text", CORPUS)
def test_idempotent_persian_digits(text):
    once = normalize(text, digits="persian")
    assert normalize(once, digits="persian") == once


# ---- search keys -----------------------------------------------------------------------------
def test_search_key_collapses_spelling_variants():
    variants = [f"کتاب{ZWNJ}ها", "كتابها", "کتاب ها", "کتابها"]
    assert len({search_key(v) for v in variants}) == 1


def test_search_key_folds_alef_hamza_digits_case():
    assert search_key("آب") == search_key("اب")
    assert search_key("گوشی ۱۲۸ گیگ") == search_key("گوشی 128 گیگ")
    assert search_key("iPhone") == "iphone"
    assert search_key("مسئله") == search_key("مسیله")


def test_search_key_strips_punctuation_and_diacritics():
    assert search_key("سلامِ، دنیا!") == "سلام دنیا"
    assert search_key("") == ""
