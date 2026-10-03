"""Gregorian → Jalali (Solar Hijri) conversion, for monthly quota periods sellers recognise."""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

MONTHS_FA = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
TEHRAN = timezone(timedelta(hours=3, minutes=30))  # Iran has had no DST since 2022

_G_DAYS = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Arithmetic conversion (33-year cycle), valid for the practical range 1600-2600 CE."""
    gy2 = gy + 1 if gm > 2 else gy
    days = 355666 + 365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400 + gd + _G_DAYS[gm - 1]
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm, jd = 1 + days // 31, 1 + days % 31
    else:
        jm, jd = 7 + (days - 186) // 30, 1 + (days - 186) % 30
    return jy, jm, jd


def jalali_today(now: datetime | None = None) -> tuple[int, int, int]:
    now = (now or datetime.now(timezone.utc)).astimezone(TEHRAN)
    d: date = now.date()
    return gregorian_to_jalali(d.year, d.month, d.day)


def period_key(now: datetime | None = None) -> str:
    jy, jm, _ = jalali_today(now)
    return f"{jy:04d}{jm:02d}"


def period_label_fa(now: datetime | None = None) -> str:
    from st_common.persian import to_persian_digits

    jy, jm, _ = jalali_today(now)
    return to_persian_digits(f"{MONTHS_FA[jm - 1]} {jy}", protect_ltr=False)
