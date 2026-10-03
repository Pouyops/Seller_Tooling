"""Inference service settings (env prefix ``ST_``; see .env.example)."""

from __future__ import annotations

from functools import lru_cache

from st_common.config import CommonSettings


class ServiceSettings(CommonSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    matting_model: str = "birefnet_lite"
    allowed_models: str = "birefnet_lite,birefnet,ben2_base,inspyrenet_base,inspyrenet_fast"

    # batching
    max_batch: int = 4
    batch_wait_ms: int = 40

    # device policy
    device: str = "auto"  # auto | cuda | cuda:N | cpu
    allow_cpu_fallback: bool = False  # False: jobs wait in the queue while the GPU is down

    # input limits
    max_upload_mb: float = 15.0
    max_pixels: int = 40_000_000
    max_side: int = 4096

    # queue semantics
    visibility_timeout_ms: int = 300_000
    max_attempts: int = 3
    max_wait_s: float = 30.0

    # worker
    worker_metrics_port: int = 9101
    worker_heartbeat_s: float = 5.0
    gpu_retry_max_s: float = 60.0

    @property
    def allowed(self) -> list[str]:
        names = [m.strip() for m in self.allowed_models.split(",") if m.strip()]
        if self.matting_model not in names:
            names.insert(0, self.matting_model)
        return names


@lru_cache
def get_settings() -> ServiceSettings:
    return ServiceSettings()
