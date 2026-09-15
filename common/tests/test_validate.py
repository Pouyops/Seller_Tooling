from st_common.persian import detect_mojibake, is_valid_persian, script_ratio, validate_persian


def test_valid_mixed_listing_title():
    assert validate_persian("گوشی سامسونگ مدل A54 با گارانتی ۱۸ ماهه") == []
    assert is_valid_persian("زعفران سرگل ممتاز")


def test_arabic_characters_flagged():
    issues = validate_persian("كتاب")
    assert "arabic_chars" in issues
    assert "not_normalized" in issues


def test_mojibake_detected():
    garbled = "سلام دنیا".encode("utf-8").decode("cp1252", errors="replace")
    assert detect_mojibake(garbled)
    assert "mojibake" in validate_persian(garbled)
    assert not detect_mojibake("سلام دنیا")


def test_replacement_char_is_mojibake():
    assert detect_mojibake("سلا�م")


def test_low_persian_ratio():
    assert "low_persian_ratio" in validate_persian("This is English")


def test_foreign_script_leak():
    assert "foreign_script" in validate_persian("گوشی 手机 سامسونگ")


def test_empty_and_whitespace():
    assert validate_persian("") == ["empty"]
    assert validate_persian("   ") == ["empty"]


def test_not_normalized():
    assert "not_normalized" in validate_persian("کتاب ها")
    assert validate_persian("کتاب ها", require_normalized=False) == []


def test_control_chars():
    assert "control_chars" in validate_persian("سلام\x07")


def test_script_ratio():
    r = script_ratio("سلام ab 12")
    assert abs(r["persian"] - 4 / 6) < 1e-9
    assert abs(r["latin"] - 2 / 6) < 1e-9
    assert script_ratio("123") == {"persian": 0.0, "latin": 0.0, "other": 0.0}
