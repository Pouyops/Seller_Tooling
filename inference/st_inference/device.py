"""GPU discovery and health, shared by the worker, the API and the benchmark."""

from __future__ import annotations

import platform
from functools import lru_cache
from typing import Any


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch missing or broken driver
        return False


def select_device(pref: str = "auto") -> str:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref.startswith("cuda"):
        return pref if cuda_available() else "unavailable"
    return "cuda" if cuda_available() else "cpu"


def gpu_healthy(device: str = "cuda") -> bool:
    """A real round-trip, not just ``is_available``: catches Xid errors / driver resets / lost devices."""
    if not device.startswith("cuda"):
        return False
    try:
        import torch

        x = torch.ones(8, device=device)
        return float((x * 2).sum().item()) == 16.0
    except Exception:
        return False


def fp16_unreliable(device: str = "cuda") -> bool:
    """GPUs whose fp16 kernels are known to return NaN/garbage for vision transformers.

    GeForce GTX 16xx (Turing TU116/TU117 without tensor cores) is the well-known case: the same
    reason Stable Diffusion front-ends force ``--no-half`` there. Unknown GPUs are probed at load
    time instead (``MattingModel._outputs_finite``).
    """
    if not device.startswith("cuda"):
        return False
    try:
        import torch

        name = torch.cuda.get_device_name(device)
    except Exception:
        return False
    return "GTX 16" in name.upper()


@lru_cache
def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def vram_used_mb(index: int = 0) -> float | None:
    nv = _nvml()
    if nv is None:
        return None
    try:
        h = nv.nvmlDeviceGetHandleByIndex(index)
        return nv.nvmlDeviceGetMemoryInfo(h).used / 2**20
    except Exception:
        return None


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        import psutil

        info["cpu_cores_logical"] = psutil.cpu_count()
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_vram_gb"] = round(p.total_memory / 2**30, 2)
            info["gpu_capability"] = f"{p.major}.{p.minor}"
    except Exception:
        pass
    nv = _nvml()
    if nv is not None:
        try:
            info["nvidia_driver"] = nv.nvmlSystemGetDriverVersion()
            h = nv.nvmlDeviceGetHandleByIndex(0)
            info["gpu_power_limit_w"] = round(nv.nvmlDeviceGetPowerManagementLimit(h) / 1000, 1)
        except Exception:
            pass
    return info
