"""Common interface for background-removal models.

The service worker and the evaluation harness both go through this class, so benchmark
numbers are measured on exactly the code path production runs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from st_common.licensing import check_model_allowed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Timing:
    preprocess_ms: float = 0.0
    forward_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.forward_ms + self.postprocess_ms


@dataclass
class ModelInfo:
    name: str
    repo_id: str
    backend: str
    precision: str
    input_size: tuple[int, int] | None
    extra: dict[str, Any] = field(default_factory=dict)


class MattingModel(ABC):
    """Takes RGB uint8 images of any size, returns float32 alpha mattes at the original size."""

    backend = "torch"

    def __init__(
        self,
        name: str,
        repo_id: str,
        weights_path: Path,
        input_size: tuple[int, int] | None,
        device: str = "cuda",
        precision: str = "fp16",
    ):
        self.name = name
        self.repo_id = repo_id
        self.weights_path = Path(weights_path)
        self.input_size = input_size  # (W, H)
        self.device = device
        self.precision = precision
        self.model = None
        self.precision_note: str | None = None
        self.last_timing = Timing()

    # ---- lifecycle ---------------------------------------------------------------------------
    def load(self) -> MattingModel:
        check_model_allowed(self.repo_id, self.weights_path.name)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"weights for {self.name} not found at {self.weights_path} (run `make setup`)")
        import torch

        from ..device import fp16_unreliable

        model = self._build()
        model.eval()
        model.to(self.device)
        cuda = self.device.startswith("cuda")
        if self.precision == "fp16" and cuda and fp16_unreliable(self.device):
            self.precision_note = f"fp16 requested but disabled: known NaN outputs on {torch.cuda.get_device_name(self.device)}"
            self.precision = "fp32"
        if self.precision == "fp16" and cuda:
            model.half()
        else:
            # Weights may be stored in fp16 (transformers>=5 keeps the stored dtype); force what we asked for.
            model.float()
        self.model = model
        torch.set_grad_enabled(False)
        if self.precision == "fp16" and cuda and not self._outputs_finite():
            self.precision_note = "fp16 produced non-finite outputs on a probe input; fell back to fp32"
            self.precision = "fp32"
            self.model.float()
        return self

    def _outputs_finite(self) -> bool:
        import torch

        W, H = self.input_size or (512, 512)
        g = torch.Generator(device="cpu").manual_seed(0)
        probe = (torch.rand((1, H, W, 3), generator=g) * 255).to(torch.uint8).numpy()
        with torch.inference_mode():
            y = self.forward(self.preprocess([probe]))
        return bool(torch.isfinite(y).all())

    def unload(self) -> None:
        import torch

        self.model = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def _build(self):
        """Construct the torch module with weights loaded (on CPU)."""

    # ---- inference ---------------------------------------------------------------------------
    @property
    def _dtype(self):
        import torch

        return torch.float16 if (self.precision == "fp16" and self.device.startswith("cuda")) else torch.float32

    def preprocess(self, images: list[np.ndarray]):
        import torch

        W, H = self.input_size
        resized = [
            cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA if im.shape[1] > W else cv2.INTER_LINEAR) for im in images
        ]
        batch = torch.from_numpy(np.stack(resized)).to(self.device, non_blocking=True)
        x = batch.permute(0, 3, 1, 2).to(self._dtype).div_(255.0)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self._dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self._dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    @abstractmethod
    def forward(self, x):
        """Normalized batch (B,3,H,W) -> foreground probability (B,1,h,w) in [0,1]."""

    def postprocess(self, probs, sizes: list[tuple[int, int]]) -> list[np.ndarray]:
        import torch.nn.functional as F

        out = []
        for i, (h, w) in enumerate(sizes):
            p = F.interpolate(probs[i : i + 1].float(), size=(h, w), mode="bilinear", align_corners=False)
            out.append(p[0, 0].clamp_(0, 1).cpu().numpy())
        return out

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        if self.model is None:
            raise RuntimeError(f"{self.name} is not loaded")
        sync = torch.cuda.synchronize if self.device.startswith("cuda") else (lambda: None)
        t0 = time.perf_counter()
        x = self.preprocess(images)
        sync()
        t1 = time.perf_counter()
        with torch.inference_mode():
            probs = self.forward(x)
        sync()
        t2 = time.perf_counter()
        out = self.postprocess(probs, [im.shape[:2] for im in images])
        t3 = time.perf_counter()
        self.last_timing = Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
        return out

    def info(self) -> ModelInfo:
        return ModelInfo(self.name, self.repo_id, self.backend, self.precision, self.input_size)
