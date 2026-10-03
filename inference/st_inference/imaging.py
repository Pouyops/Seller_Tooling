"""Image validation, decoding and cutout encoding."""

from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None  # we enforce our own limit before decoding pixels

_FORMAT_EXT = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "GIF": "gif", "TIFF": "tif"}
_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")
_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0)}


class ImageRejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def probe_image(data: bytes, max_pixels: int) -> dict:
    """Cheap header check: format and size, without decoding pixels."""
    if not data:
        raise ImageRejected("empty", "empty upload")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except (UnidentifiedImageError, OSError) as e:
        raise ImageRejected("not_an_image", f"not a supported image: {e}") from None
    if fmt not in _FORMAT_EXT:
        raise ImageRejected("unsupported_format", f"unsupported image format {fmt}")
    if w * h > max_pixels:
        raise ImageRejected("too_many_pixels", f"{w}x{h} exceeds {max_pixels} pixels")
    return {"format": fmt, "ext": _FORMAT_EXT[fmt], "width": w, "height": h}


def decode_rgb(data: bytes, max_side: int = 4096) -> np.ndarray:
    """Decode to RGB uint8, applying EXIF orientation (phone photos) and flattening any alpha on white."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            im = im.convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return np.asarray(im).copy()
    except (UnidentifiedImageError, OSError) as e:
        raise ImageRejected("decode_failed", f"could not decode image: {e}") from None


def parse_background(value: str | None) -> tuple[int, int, int] | None:
    """``None``/"transparent" -> None; "white" / "#ffcc00" -> RGB tuple."""
    if value is None or value == "" or value == "transparent":
        return None
    if value in _NAMED:
        return _NAMED[value]
    m = _HEX.match(value)
    if not m:
        raise ValueError(f"background must be 'transparent', 'white', 'black' or #rrggbb, got {value!r}")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def encode_cutout(rgb: np.ndarray, alpha: np.ndarray, background: str | None = "transparent") -> tuple[bytes, str, str]:
    """RGBA PNG for transparent output; JPEG composited on a solid colour otherwise (marketplace white)."""
    color = parse_background(background)
    a = np.clip(alpha, 0, 1).astype(np.float32)
    buf = io.BytesIO()
    if color is None:
        rgba = np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=3)
        return buf.getvalue(), "png", "image/png"
    out = rgb.astype(np.float32) * a[..., None] + np.asarray(color, np.float32) * (1 - a[..., None])
    Image.fromarray((out + 0.5).clip(0, 255).astype(np.uint8), "RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue(), "jpg", "image/jpeg"


def encode_mask(alpha: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(buf, "PNG", compress_level=3)
    return buf.getvalue()
