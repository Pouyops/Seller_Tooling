from st_common.persian import (
    display_text,
    first_strong_direction,
    isolate_ltr_runs,
    rtl_paragraphs,
    strip_bidi_controls,
)

LRM, RLM, LRI, PDI = "‎", "‏", "⁦", "⁩"


def test_strip_bidi_controls():
    assert strip_bidi_controls(f"{RLM}سلام{LRM}‮{LRI}x{PDI}") == "سلامx"
    assert strip_bidi_controls("") == ""


def test_first_strong_direction():
    assert first_strong_direction("123 iPhone سلام") == "ltr"
    assert first_strong_direction("۱۲ سلام iPhone") == "rtl"
    assert first_strong_direction("123 - 456") is None


def test_rtl_paragraphs_prefixes_only_mixed_lines_starting_ltr():
    text = "iPhone 15 نو\nسلام\nhello\n"
    assert rtl_paragraphs(text) == f"{RLM}iPhone 15 نو\nسلام\nhello\n"
    # idempotent
    assert rtl_paragraphs(rtl_paragraphs(text)) == rtl_paragraphs(text)


def test_rtl_paragraphs_number_first_line():
    # A leading number is weak; first *strong* char is Persian, so no mark is needed.
    assert rtl_paragraphs("۱۲ عدد") == "۱۲ عدد"


def test_isolate_ltr_runs_lrm():
    assert isolate_ltr_runs("گوشی iPhone 15 Pro مشکی") == f"گوشی {LRM}iPhone 15 Pro{LRM} مشکی"


def test_isolate_ltr_runs_isolate_mode():
    assert isolate_ltr_runs("کابل USB-C 2.0 اصل", mode="isolate") == f"کابل {LRI}USB-C 2.0{PDI} اصل"


def test_isolate_leaves_numbers_and_pure_ltr_alone():
    assert isolate_ltr_runs("قیمت 1200 تومان") == "قیمت 1200 تومان"
    assert isolate_ltr_runs("hello world") == "hello world"
    assert isolate_ltr_runs("") == ""


def test_display_text():
    assert display_text("iPhone 15 نو") == f"{RLM}{LRM}iPhone 15{LRM} نو"
    assert display_text("سلام دنیا", isolate="none") == "سلام دنیا"
    # stray controls from the input are replaced, not duplicated
    assert display_text(f"{LRM}{LRM}سلام") == "سلام"
