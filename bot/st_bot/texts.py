"""All user-facing strings. Persian, RTL, normalized (a unit test asserts every string is already
in canonical form and passes the Persian validator)."""

from __future__ import annotations

from st_common.persian import display_text, format_number_fa, format_price_toman, to_persian_digits

COMMANDS = [
    ("start", "شروع"),
    ("help", "راهنما"),
    ("quota", "اعتبار من"),
    ("buy", "خرید اعتبار"),
    ("background", "رنگ پس‌زمینه"),
]

WELCOME = (
    "سلام! 👋\n"
    "من ربات حذف پس‌زمینه‌ی عکس محصول هستم.\n"
    "عکس محصولتان را بفرستید تا نسخه‌ی بدون پس‌زمینه را تحویل بگیرید. ثبت‌نام لازم نیست.\n\n"
    "🎁 هر ماه {free} عکس رایگان دارید.\n"
    "📎 برای بهترین کیفیت، عکس را به‌صورت «فایل» بفرستید، نه «عکس»."
)

HELP = (
    "راهنما:\n"
    "۱. عکس محصول را بفرستید (تا {album} عکس با هم).\n"
    "۲. خروجی به‌صورت فایل PNG با پس‌زمینه‌ی شفاف ارسال می‌شود.\n"
    "۳. با دستور /background می‌توانید پس‌زمینه‌ی سفید را برای دیجی‌کالا و باسلام انتخاب کنید.\n"
    "۴. با /quota اعتبار باقی‌مانده را ببینید و با /buy اعتبار بخرید.\n\n"
    "نکته: محصول را روی زمینه‌ای با رنگ متفاوت و در نور کافی عکاسی کنید."
)

QUOTA = "اعتبار شما در {period}:\nرایگان: {free_left} از {free_total}\nاعتبار خریداری‌شده: {credits}"
RECEIVED = "عکس دریافت شد ✅ در حال پردازش…"
RECEIVED_ALBUM = "{n} عکس دریافت شد ✅ در حال پردازش…"
QUEUED_LATER = "صف پردازش شلوغ است. به محض آماده شدن، نتیجه را برایتان می‌فرستم ⏳"
DONE_CAPTION = "✂️ پس‌زمینه حذف شد.\nاعتبار باقی‌مانده: {left}"
FAILED = "متأسفانه پردازش این عکس انجام نشد و اعتبارتان برگشت داده شد. لطفاً دوباره تلاش کنید."
NOT_IMAGE = "این فایل تصویر نیست. لطفاً عکس JPG یا PNG بفرستید."
TOO_LARGE = "حجم فایل بیشتر از حد مجاز است. لطفاً عکس کوچک‌تری بفرستید."
RATE_LIMITED = "کمی آهسته‌تر 🙂 لطفاً {seconds} ثانیه صبر کنید و دوباره بفرستید."
OUT_OF_QUOTA = "اعتبار رایگان این ماه تمام شده است.\nبرای ادامه با دستور /buy اعتبار بخرید."
ALBUM_TOO_BIG = "حداکثر {album} عکس را با هم می‌توانم پردازش کنم. بقیه را جداگانه بفرستید."
SERVICE_DOWN = "سرویس موقتاً در دسترس نیست. عکس‌تان ذخیره نشد؛ چند دقیقه بعد دوباره بفرستید."

BUY_INTRO = "بسته‌های اعتبار:"
PACK_BUTTON = "{credits} عکس – {price}"
PAYMENTS_DISABLED = (
    "صورتحساب {invoice} برای بسته‌ی {credits} عکس به مبلغ {price} ساخته شد.\n"
    "درگاه پرداخت هنوز فعال نشده است. به‌زودی امکان پرداخت اضافه می‌شود."
)
PAID = "پرداخت تأیید شد ✅ {credits} اعتبار به حساب شما اضافه شد."

BG_CHOOSE = "پس‌زمینه‌ی خروجی را انتخاب کنید:"
BG_TRANSPARENT = "شفاف (PNG)"
BG_WHITE = "سفید (JPG)"
BG_SET = "پس‌زمینه‌ی خروجی تنظیم شد: {choice}"


def fa(template: str, **kw) -> str:
    """Fill a template with Persian digits and bidi-safe layout for Telegram."""
    vals = {}
    for k, v in kw.items():
        if isinstance(v, int):
            vals[k] = format_number_fa(v)
        else:
            vals[k] = v
    return display_text(to_persian_digits(template.format(**vals)))


def pack_label(credits: int, price_toman: int) -> str:
    return PACK_BUTTON.format(credits=format_number_fa(credits), price=format_price_toman(price_toman))


ALL_TEMPLATES = [
    WELCOME, HELP, QUOTA, RECEIVED, RECEIVED_ALBUM, QUEUED_LATER, DONE_CAPTION, FAILED, NOT_IMAGE, TOO_LARGE,
    RATE_LIMITED, OUT_OF_QUOTA, ALBUM_TOO_BIG, SERVICE_DOWN, BUY_INTRO, PAYMENTS_DISABLED, PAID, BG_CHOOSE,
    BG_TRANSPARENT, BG_WHITE, BG_SET,
] + [d for _, d in COMMANDS]
