"""Listing generator: parsing, Persian normalization, validation, repair and fallback.

No network: the LLM is stubbed. The real model is exercised separately by tools/listing_demo.py.
"""

import json

import pytest

from st_common.persian import normalize, search_key
from st_inference.listing import ListingGenerator, SellerFields
from st_inference.listing.llm_client import LLMUnavailable
from st_inference.listing.schema import KEYWORDS_MIN, TITLE_MAX

CARPET = SellerFields(category="فرش دستباف", material="پشم", size="۱۲ متری", color="لاکی",
                      origin="کاشان", price_toman=85_000_000, notes="بدون لک و پارگی")


class StubLLM:
    model = "stub"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, schema=None, **kw):
        self.calls.append(messages)
        if not self.replies:
            raise LLMUnavailable("stub exhausted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def reply(**over) -> str:
    base = {
        "title": "فرش دستباف کاشان ۱۲ متری",
        "description": "فرش دستباف کاشان با بافت پشم مرغوب. رنگ‌بندی لاکی و سرمه‌ای مناسب پذیرایی.",
        "attributes": {"جنس": "پشم", "اندازه": "۱۲ متری"},
        "keywords": ["فرش دستباف", "فرش کاشان", "فرش پشمی", "فرش ۱۲ متری", "فرش لاکی", "فرش دستبافت"],
        "category_guess": "فرش و گلیم",
    }
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


async def gen(*replies, fields=CARPET, image=b"\xff\xd8fake"):
    llm = StubLLM(*replies)
    result = await ListingGenerator(llm).generate(image, fields)
    return result, llm


async def test_happy_path_is_normalized_persian():
    result, llm = await gen(reply())
    assert result.source == "model" and result.ok and result.attempts == 1
    assert result.listing.title == "فرش دستباف کاشان ۱۲ متری"
    assert result.listing.attributes["جنس"] == "پشم"
    assert len(result.listing.keywords) >= KEYWORDS_MIN
    # the image was actually attached to the prompt
    content = llm.calls[0][1]["content"]
    assert any(part.get("type") == "image_url" for part in content)


async def test_arabic_characters_and_spacing_are_fixed():
    raw = reply(title="فرش دستباف كاشان ۱۲ مترى", description="اين فرش را مى توان در پذيرايى استفاده كرد و تميز است.")
    result, _ = await gen(raw)
    title = result.listing.title
    assert "ك" not in title and "ى" not in title
    assert title == normalize(title)
    assert "می‌توان" in result.listing.description


async def test_json_in_code_fences_is_parsed():
    result, _ = await gen(f"اینم خروجی:\n```json\n{reply()}\n```\nموفق باشید")
    assert result.source == "model" and result.listing.title.startswith("فرش")


async def test_bad_json_is_repaired_on_the_second_attempt():
    result, llm = await gen("نه JSON است و نه چیز دیگر", reply())
    assert result.source == "model_repaired" and result.attempts == 2
    repair_prompt = llm.calls[1][-1]["content"]
    assert "JSON" in repair_prompt and "ایراد" in repair_prompt


async def test_persistent_garbage_falls_back_to_a_template():
    result, _ = await gen("نه", "همچنان نه")
    assert result.source == "template_fallback" and not result.ok
    listing = result.listing
    assert listing.title == "فرش دستباف لاکی ۱۲ متری"
    assert "پشم" in listing.description and "کاشان" in listing.description
    assert listing.attributes["جنس"] == "پشم" and listing.attributes["قیمت (تومان)"] == "۸۵۰۰۰۰۰۰"
    assert len(listing.keywords) >= 3
    assert "بدون لک و پارگی" in listing.description


async def test_llm_down_falls_back_without_crashing():
    result, _ = await gen(LLMUnavailable("connection refused"))
    assert result.source == "template_fallback" and result.issues == ["llm_unavailable"]
    assert result.listing.title


async def test_english_output_is_rejected_then_falls_back():
    english = json.dumps({"title": "Handmade Persian Carpet 12m", "description": "A beautiful handmade carpet from Kashan.",
                          "attributes": {}, "keywords": []}, ensure_ascii=False)
    result, _ = await gen(english, english)
    assert result.source == "template_fallback"
    assert "low_persian_ratio" in result.issues


async def test_long_title_is_truncated_but_not_fatal():
    long_title = "فرش دستباف کاشان طرح لچک ترنج دوازده متری پشم مرغوب رنگ لاکی سرمه‌ای بسیار زیبا و تمیز بدون لک"
    # a too-long title earns one rewrite request; if the model repeats it, we truncate and ship
    result, _ = await gen(reply(title=long_title), reply(title=long_title))
    assert result.source == "model_repaired" and result.ok
    assert len(result.listing.title) <= TITLE_MAX
    assert "title_too_long" in result.issues
    assert not result.listing.title.endswith(" ")


async def test_keywords_are_deduped_by_meaning_and_topped_up():
    thin = ["فرش‌ها", "فرش ها", "فرشها", "  "]
    result, _ = await gen(reply(keywords=thin), reply(keywords=thin))
    keys = [search_key(k) for k in result.listing.keywords]
    assert len(keys) == len(set(keys)), result.listing.keywords
    assert len(result.listing.keywords) >= KEYWORDS_MIN
    assert "too_few_keywords" in result.issues


async def test_attributes_are_capped_and_cleaned():
    attrs = {f"ویژگی {i}": f"مقدار {i}" for i in range(20)}
    attrs["خالی"] = "   "
    result, _ = await gen(reply(attributes=attrs))
    assert len(result.listing.attributes) <= 12
    assert "خالی" not in result.listing.attributes


async def test_emoji_are_stripped():
    result, _ = await gen(reply(title="فرش دستباف کاشان 🔥🎉", keywords=["فرش دستباف ✨"]))
    assert "🔥" not in result.listing.title and "✨" not in " ".join(result.listing.keywords)


async def test_repetitive_description_triggers_one_rewrite():
    padded = ("فرش دستباف کاشان با جنس پشم و رنگ لاکی مناسب پذیرایی خانه شماست. "
              "فرش دستباف کاشان با جنس پشم و رنگ لاکی مناسب پذیرایی خانه شما است.")
    # first reply repeats itself, second is clean -> repaired rather than fallback
    result, llm = await gen(reply(description=padded), reply())
    assert result.source == "model_repaired" and result.ok
    assert "تکراری" in llm.calls[1][-1]["content"]

    # if the model never improves, a repetitive-but-valid listing still beats the template
    result2, _ = await gen(reply(description=padded), reply(description=padded))
    assert result2.source == "model_repaired" and "repetitive" in result2.issues
    assert result2.listing.description.startswith("فرش دستباف کاشان")


async def test_fallback_without_any_seller_fields():
    result, _ = await gen("junk", "junk", fields=SellerFields())
    assert result.listing.title == "محصول" and result.listing.description


async def test_image_is_optional():
    result, llm = await gen(reply(), image=None)
    assert result.ok
    assert all(part.get("type") != "image_url" for part in llm.calls[0][1]["content"])
