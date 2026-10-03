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

    # Results are derived artifacts: if a power cut loses one, the job is simply re-run, and the API
    # already answers 410 for a missing blob. fsync costs 174 ms/image here vs 2 ms without (measured,
    # docs/loadtest.md). Uploaded originals are always fsync'ed — those cannot be recomputed.
    blob_fsync_results: bool = False

    # queue semantics. Worst-case recovery after a worker dies is one visibility timeout, so keep it
    # a small multiple of the slowest expected inference, not minutes (docs/loadtest.md, outage run).
    visibility_timeout_ms: int = 60_000
    max_attempts: int = 3
    max_wait_s: float = 30.0

    # local LLM for listing generation (llama.cpp server speaking the OpenAI protocol)
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "local"
    llm_timeout_s: float = 180.0

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
