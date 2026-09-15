"""Machine-enforced copy of the DO-NOT-SHIP list in docs/licenses.md.

Every model loader calls ``check_model_allowed`` with the upstream repo id *and* the local
weight filename, so re-uploads and quantized mirrors (``someone/RMBG-2.0-onnx``,
``bartowski/Qwen2.5-3B-Instruct-GGUF``) are caught too.
"""

from __future__ import annotations

import re

# Exact upstream repo ids (lower-case).
DO_NOT_SHIP: frozenset[str] = frozenset(
    {
        "briaai/rmbg-1.4",
        "briaai/rmbg-2.0",
        "briaai/vrmbg-3.0",
        "black-forest-labs/flux.1-dev",
        "black-forest-labs/flux.1-fill-dev",
        "black-forest-labs/flux.1-depth-dev",
        "black-forest-labs/flux.1-canny-dev",
        "black-forest-labs/flux.1-redux-dev",
        "black-forest-labs/flux.1-kontext-dev",
        "black-forest-labs/flux.2-klein-9b",
        "black-forest-labs/flux.2-klein-base-9b",
        "black-forest-labs/flux.2-klein-9b-kv",
        "qwen/qwen2.5-3b-instruct",
        "qwen/qwen2.5-vl-3b-instruct",
        "coherelabs/aya-expanse-8b",
        "coherelabs/aya-vision-8b",
        "universitytehran/persianmind-v1.0",
        "viraintelligentdatamining/persianllama-13b",
        "openbmb/minicpm-v-2_6",
    }
)

# Name fragments that identify blocked weights wherever they are re-hosted.
DO_NOT_SHIP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"rmbg[-_.]?(1\.4|2\.0|2|v?3)",
        r"flux\.?1[-_.](\w+[-_.])?dev",
        r"flux[-_.]dev",
        r"flux\.?2[-_.]klein[-_.](base[-_.])?9b",
        r"qwen2\.5[-_.](vl[-_.])?3b",
        r"aya[-_.](expanse|vision)",
        r"tiny[-_.]aya",
        r"north[-_.]small[-_.]translate",
        r"persianmind",
        r"persianllama",
        r"minicpm[-_.]?v",
    )
)

# Allowed only with an explicit sign-off (HUMAN_NEEDED). Loaders refuse unless allow_conditional=True.
CONDITIONAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"stable[-_.]diffusion[-_.]3\.5",
        r"sd3\.5",
        r"llama[-_.]?3",
        r"dorna",
        r"gemma[-_.]3",
        r"qwen3\.8[-_.]flash",
        r"trendyol/background[-_.]removal",
    )
)


class LicenseViolation(RuntimeError):
    """Raised when code tries to load weights that are not cleared for commercial use."""


def license_status(identifier: str) -> str:
    """Return ``"blocked"``, ``"conditional"`` or ``"allowed"`` for a repo id or file name."""
    ident = identifier.strip().lower().replace("\\", "/")
    if ident in DO_NOT_SHIP or any(p.search(ident) for p in DO_NOT_SHIP_PATTERNS):
        return "blocked"
    if any(p.search(ident) for p in CONDITIONAL_PATTERNS):
        return "conditional"
    return "allowed"


def check_model_allowed(*identifiers: str, allow_conditional: bool = False) -> None:
    for ident in identifiers:
        if not ident:
            continue
        status = license_status(ident)
        if status == "blocked":
            raise LicenseViolation(f"{ident!r} is on the DO-NOT-SHIP list (docs/licenses.md)")
        if status == "conditional" and not allow_conditional:
            raise LicenseViolation(
                f"{ident!r} is CONDITIONAL (docs/licenses.md); needs sign-off and allow_conditional=True"
            )
