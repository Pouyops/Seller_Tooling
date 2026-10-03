from __future__ import annotations

from functools import lru_cache

from st_common.config import CommonSettings


class BotSettings(CommonSettings):
    telegram_token: str = ""
    telegram_api_base: str = ""  # e.g. a self-hosted Bot API server, or the test mock
    inference_url: str = "http://localhost:8000"

    free_quota_per_month: int = 20  # resets on the 1st of each Jalali (Solar Hijri) month
    rate_limit_per_minute: int = 10
    max_album_photos: int = 10
    album_collect_ms: int = 1500  # wait this long for the rest of a media group
    job_wait_s: float = 25.0  # wait inline this long, then deliver later from the pending set
    deliver_poll_s: float = 3.0
    job_give_up_s: float = 6 * 3600  # jobs still unfinished after this are refunded and reported

    payments_dev_confirm: bool = False  # allow /devpay to mark a stub invoice paid (never in production)
    admin_user_ids: str = ""

    @property
    def admins(self) -> set[int]:
        return {int(x) for x in self.admin_user_ids.split(",") if x.strip().isdigit()}


@lru_cache
def get_bot_settings() -> BotSettings:
    return BotSettings()
