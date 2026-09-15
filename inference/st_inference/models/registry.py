"""Name -> model factory. Only licence-cleared models can be registered (enforced at import and load)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from st_common.licensing import check_model_allowed

from .adapters import BEN2BaseModel, BiRefNetModel, InSPyReNetModel
from .base import MattingModel


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cls: Callable[..., MattingModel]
    repo_id: str
    weights: str  # path relative to the models dir
    input_size: tuple[int, int] | None
    precision: str
    download: dict  # how `make setup` fetches the weights
    notes: str = ""


def _hf(repo, allow=None):
    return {"type": "hf", "repo": repo, "allow_patterns": allow}


REGISTRY: dict[str, ModelSpec] = {
    s.name: s
    for s in [
        ModelSpec("birefnet_lite", BiRefNetModel, "ZhengPeng7/BiRefNet_lite", "ZhengPeng7__BiRefNet_lite", (1024, 1024), "fp16",
                  _hf("ZhengPeng7/BiRefNet_lite"), "Swin-T backbone"),
        ModelSpec("birefnet", BiRefNetModel, "ZhengPeng7/BiRefNet", "ZhengPeng7__BiRefNet", (1024, 1024), "fp16",
                  _hf("ZhengPeng7/BiRefNet"), "Swin-L backbone, general-use weights"),
        ModelSpec("birefnet_dynamic", BiRefNetModel, "ZhengPeng7/BiRefNet_dynamic", "ZhengPeng7__BiRefNet_dynamic", (1024, 1024), "fp16",
                  _hf("ZhengPeng7/BiRefNet_dynamic"), "Swin-L, trained on dynamic resolutions; run at 1024 for comparability"),
        ModelSpec("birefnet_hr", BiRefNetModel, "ZhengPeng7/BiRefNet_HR", "ZhengPeng7__BiRefNet_HR", (2048, 2048), "fp16",
                  _hf("ZhengPeng7/BiRefNet_HR"), "Swin-L at 2048x2048"),
        ModelSpec("ben2_base", BEN2BaseModel, "PramaLLC/BEN2", "PramaLLC__BEN2", (1024, 1024), "amp",
                  _hf("PramaLLC/BEN2", ["BEN2.py", "config.json", "model.safetensors"]), "BEN2 Base; fp32 weights with internal fp16 autocast"),
        ModelSpec("inspyrenet_base", InSPyReNetModel, "plemeri/InSPyReNet", "inspyrenet/ckpt_base.pth", (1024, 1024), "fp32",
                  {"type": "url", "url": "https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base.pth",
                   "md5": "d692e3dd5fa1b9658949d452bebf1cda"}, "Swin-B, transparent-background 'base' checkpoint"),
        ModelSpec("inspyrenet_fast", InSPyReNetModel, "plemeri/InSPyReNet", "inspyrenet/ckpt_fast.pth", (384, 384), "fp32",
                  {"type": "url", "url": "https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_fast.pth",
                   "md5": "9efdbfbcc49b79ef0f7891c83d2fd52f"}, "Swin-B at 384x384"),
    ]
}

# Fail fast if anyone ever registers a blocked model.
for _spec in REGISTRY.values():
    check_model_allowed(_spec.repo_id, _spec.weights)


def default_models_dir() -> Path:
    return Path(os.environ.get("ST_DATA_DIR", Path.cwd() / ".data")) / "models"


def create_model(name: str, *, models_dir: Path | str | None = None, device: str = "cuda",
                 precision: str | None = None, input_size: tuple[int, int] | None = None) -> MattingModel:
    try:
        spec = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(REGISTRY)}") from None
    root = Path(models_dir) if models_dir else default_models_dir()
    return spec.cls(
        name=spec.name,
        repo_id=spec.repo_id,
        weights_path=root / spec.weights,
        input_size=input_size or spec.input_size,
        device=device,
        precision=precision or spec.precision,
    )


def available_models(models_dir: Path | str | None = None) -> dict[str, bool]:
    root = Path(models_dir) if models_dir else default_models_dir()
    return {name: (root / spec.weights).exists() for name, spec in REGISTRY.items()}
