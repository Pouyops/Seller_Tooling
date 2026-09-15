"""Concrete adapters for the licence-cleared matting models (docs/licenses.md §1)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from .base import MattingModel


class BiRefNetModel(MattingModel):
    """BiRefNet family (MIT). Loaded from a local snapshot via transformers remote code."""

    def _build(self):
        import torch
        from transformers import AutoModelForImageSegmentation

        prev = torch.get_float32_matmul_precision()
        model = AutoModelForImageSegmentation.from_pretrained(str(self.weights_path), trust_remote_code=True)
        torch.set_float32_matmul_precision(prev)
        return model

    def forward(self, x):
        return self.model(x)[-1].sigmoid()


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


class BEN2BaseModel(MattingModel):
    """BEN2 Base (MIT).

    Upstream decorates ``forward`` with an fp16 autocast. ``precision="amp"`` keeps that behaviour;
    ``"fp32"`` calls the undecorated function (needed on GPUs with broken fp16, see device.fp16_unreliable).
    """

    def _build(self):
        import torch
        from safetensors.torch import load_file

        prev = torch.get_float32_matmul_precision()
        spec = importlib.util.spec_from_file_location("st_vendor_ben2", self.weights_path / "BEN2.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # upstream seeds RNGs and sets matmul precision at import
        torch.set_float32_matmul_precision(prev)
        net = mod.BEN_Base()
        net.load_state_dict(load_file(str(self.weights_path / "model.safetensors")), strict=True)
        self._raw_forward = _unwrap(mod.BEN_Base.forward)
        return net

    def load(self):
        from ..device import fp16_unreliable

        requested = self.precision
        if requested in ("amp", "fp16") and fp16_unreliable(self.device):
            self.precision_note = "upstream fp16 autocast disabled on this GPU (known NaN outputs); running fp32"
            requested = "fp32"
        elif requested == "fp16":
            requested = "amp"
        self.precision = "fp32"  # weights always fp32
        super().load()
        self.precision = requested
        return self

    @property
    def _dtype(self):
        import torch

        return torch.float16 if (self.precision == "amp" and self.device.startswith("cuda")) else torch.float32

    def forward(self, x):
        if self.precision == "amp":
            return self.model.forward(x)
        return self._raw_forward(self.model, x)


def _import_inspyrenet():
    """Import InSPyReNet from the `transparent-background` package without running its __init__,
    which pulls a GUI toolkit and albumentations (whose compiled deps don't build on this box)."""
    if "transparent_background" not in sys.modules:
        spec = importlib.util.find_spec("transparent_background")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("transparent-background is not installed (pip install transparent-background==1.3.4 --no-deps)")
        pkg = types.ModuleType("transparent_background")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["transparent_background"] = pkg
    from transparent_background.InSPyReNet import InSPyReNet_SwinB

    return InSPyReNet_SwinB


class InSPyReNetModel(MattingModel):
    """InSPyReNet Swin-B (MIT) with the `transparent-background` checkpoints."""

    def _build(self):
        import torch

        InSPyReNet_SwinB = _import_inspyrenet()
        model = InSPyReNet_SwinB(depth=64, pretrained=False, threshold=None, base_size=list(self.input_size))
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model

    def forward(self, x):
        return self.model(x)


def weights_root(models_dir: Path | str) -> Path:
    return Path(models_dir)
