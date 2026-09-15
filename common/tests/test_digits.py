from decimal import Decimal

import pytest

from st_common.persian import (
    format_number_fa,
    format_price_toman,
    parse_amount_toman,
    parse_number,
    to_latin_digits,
    to_persian_digits,
)


def test_to_latin_all_digit_sets():
    assert to_latin_digits("۱۲۳٤٥٦") == "123456"
    assert to_latin_digits("１２") == "12"
    assert to_latin_digits("") == ""


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("قیمت 1200 تومان", "قیمت ۱۲۰۰ تومان"),
        ("سایز 42", "سایز ۴۲"),
        ("ابعاد 20×30 سانتی", "ابعاد ۲۰×۳۰ سانتی"),
        ("وزن ٤٥ گرم", "وزن ۴۵ گرم"),
        # digits that belong to Latin text stay Latin
        ("مدل SM-A525F", "مدل SM-A525F"),
        ("حافظه 256GB", "حافظه 256GB"),
        ("شبکه 5G", "شبکه 5G"),
        ("گوشی iPhone 15 Pro", "گوشی iPhone 15 Pro"),
        ("15 Pro مشکی", "15 Pro مشکی"),
        ("لینک https://example.com/p/123 را ببینید", "لینک https://example.com/p/123 را ببینید"),
        ("ایمیل shop1@example.com", "ایمیل shop1@example.com"),
    ],
)
def test_to_persian_protects_ltr(raw, expected):
    assert to_persian_digits(raw) == expected


def test_to_persian_unprotected():
    assert to_persian_digits("abc 12", protect_ltr=False) == "abc ۱۲"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("۱۲٬۵۰۰", Decimal("12500")),
        ("12,500", Decimal("12500")),
        ("۱٫۵", Decimal("1.5")),
        ("1.5", Decimal("1.5")),
        ("۲/۵", Decimal("2.5")),  # everyday Iranian decimal separator
        ("-۳", Decimal("-3")),
        (" ۴۲ ", Decimal("42")),
        ("abc", None),
        ("1,2", None),
        ("۱۴۰۳/۰۵/۱۲", None),  # a date, not a number
        ("", None),
        (None, None),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("۲۵۰ هزار تومان", 250_000),
        ("۱٫۵ میلیون", 1_500_000),
        ("2/5 میلیون تومان", 2_500_000),
        ("12,500,000 ریال", 1_250_000),
        ("قیمت: ۸۵۰٬۰۰۰ تومان", 850_000),
        ("۳ میلیارد", 3_000_000_000),
        ("450000", 450_000),
        ("بدون قیمت", None),
        ("", None),
    ],
)
def test_parse_amount_toman(raw, expected):
    assert parse_amount_toman(raw) == expected


def test_formatting():
    assert format_number_fa(1_250_000) == "۱٬۲۵۰٬۰۰۰"
    assert format_number_fa(2.5, 1) == "۲٫۵"
    assert format_number_fa(0) == "۰"
    assert format_price_toman(85_000) == "۸۵٬۰۰۰ تومان"


def test_format_parse_roundtrip():
    for n in (0, 7, 999, 1000, 123_456_789):
        assert parse_number(format_number_fa(n)) == Decimal(n)
