"""Prompt construction. Persian instructions, because the output must be Persian."""

from __future__ import annotations

import base64
import json

from .schema import ATTRIBUTES_MAX, DESCRIPTION_MAX, FIELD_LABELS_FA, KEYWORDS_MAX, KEYWORDS_MIN, TITLE_MAX, SellerFields

SYSTEM = f"""تو کارشناس نگارش آگهی برای فروشگاه‌های اینترنتی ایران هستی (دیجی‌کالا، باسلام، ترب، اینستاگرام).
با نگاه به عکس محصول و اطلاعاتی که فروشنده داده، یک آگهی حرفه‌ای به زبان فارسی بنویس.

قوام‌های اجباری:
۱. فقط فارسی بنویس. نام برند و مدل لاتین می‌تواند لاتین بماند.
۲. عنوان حداکثر {TITLE_MAX} نویسه، بدون قیمت، بدون شماره تماس و بدون ایموجی.
۳. توضیحات بین دو تا چهار جمله و حداکثر {DESCRIPTION_MAX} نویسه.
۴. بین {KEYWORDS_MIN} تا {KEYWORDS_MAX} کلمه کلیدی که خریدار ایرانی جستجو می‌کند.
۵. حداکثر {ATTRIBUTES_MAX} ویژگی. فقط چیزی را بنویس که در عکس دیده می‌شود یا فروشنده گفته است.
۶. چیزی از خودت نساز: اگر جنس یا اندازه معلوم نیست، آن ویژگی را ننویس.
۶.۱. ادعای پزشکی، درمانی یا شیمیایی ننویس و عدد و درصدی که به تو داده نشده نساز.
۷. از نیم‌فاصله درست استفاده کن (مثل «می‌شود» و «کتاب‌ها»).
۸. خروجی فقط یک شیء JSON باشد، بدون متن اضافه."""

# Deliberately a category we do not serve much of: a same-category example gets copied almost
# verbatim by small models (observed with Qwen3.5-2B on a carpet photo).
EXAMPLE = {
    "title": "قابلمه استیل دو جداره ۲۴ سانتی با درب شیشه‌ای",
    "description": "قابلمه استیل دو جداره با قطر ۲۴ سانتی‌متر و درب شیشه‌ای مقاوم. بدنه براق با دسته‌های نسوز و کف مناسب اجاق گاز. مناسب پخت روزانه در خانه و رستوران.",
    "attributes": {"جنس": "استیل", "قطر": "۲۴ سانتی‌متر", "نوع درب": "شیشه‌ای", "تعداد": "۱ عدد"},
    "keywords": ["قابلمه استیل", "قابلمه ۲۴ سانتی", "قابلمه درب شیشه‌ای", "ظروف آشپزخانه", "قابلمه دو جداره"],
    "category_guess": "ظروف پخت و پز",
}


def seller_block(fields: SellerFields) -> str:
    filled = fields.filled()
    if not filled:
        return "فروشنده اطلاعاتی وارد نکرده است؛ فقط از روی عکس بنویس."
    lines = [f"- {FIELD_LABELS_FA.get(k, k)}: {v}" for k, v in filled.items()]
    return "اطلاعات فروشنده:\n" + "\n".join(lines)


def user_text(fields: SellerFields) -> str:
    return (
        f"{seller_block(fields)}\n\n"
        f"بازار هدف: {fields.marketplace}\n\n"
        "یک نمونه‌ی درست از خروجی (فقط برای قالب، محتوایش را کپی نکن):\n"
        f"{json.dumps(EXAMPLE, ensure_ascii=False)}\n\n"
        "حالا برای این محصول آگهی بنویس و فقط JSON برگردان."
    )


def shrink(image: bytes, max_side: int = 768, quality: int = 85) -> tuple[bytes, str]:
    """A full-size photo is ~1500 prompt tokens and most of the latency; 768 px is plenty to
    describe a product. Returns the original bytes unchanged if anything goes wrong."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(image)) as im:
            im = im.convert("RGB")
            if max(im.size) <= max_side:
                return image, "image/jpeg"
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        return image, "image/jpeg"


def build_messages(fields: SellerFields, image: bytes | None, image_mime: str = "image/jpeg",
                   max_side: int = 768) -> list[dict]:
    content: list[dict] = []
    if image:
        image, image_mime = shrink(image, max_side)
        url = f"data:{image_mime};base64,{base64.b64encode(image).decode()}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": user_text(fields)})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def repair_messages(fields: SellerFields, image: bytes | None, bad_output: str, issues: list[str]) -> list[dict]:
    messages = build_messages(fields, image)
    messages.append({"role": "assistant", "content": bad_output[:2000]})
    messages.append({"role": "user", "content": "خروجی قبلی این ایرادها را داشت:\n- " + "\n- ".join(issues)
                     + "\nدوباره و فقط به صورت JSON معتبر و کاملاً فارسی بنویس."})
    return messages


ISSUE_FA = {
    "empty": "متن خالی بود",
    "mojibake": "متن به‌هم‌ریخته بود",
    "arabic_chars": "حروف عربی (ي/ك) به جای فارسی (ی/ک) استفاده شده بود",
    "low_persian_ratio": "متن به اندازه کافی فارسی نبود",
    "foreign_script": "حروف غیرفارسی و غیرلاتین در متن بود",
    "control_chars": "نویسه‌های کنترلی در متن بود",
    "not_normalized": "نیم‌فاصله یا فاصله‌گذاری درست نبود",
    "title_too_long": f"عنوان بیش از {TITLE_MAX} نویسه بود",
    "too_few_keywords": f"کمتر از {KEYWORDS_MIN} کلمه کلیدی داشت",
    "repetitive": "جمله‌های توضیحات تکراری بودند؛ هر جمله باید اطلاعات تازه بدهد",
    "bad_json": "خروجی JSON معتبر نبود",
    "missing_fields": "بعضی کلیدهای لازم نبودند",
}
