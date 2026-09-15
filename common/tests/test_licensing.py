import re
from pathlib import Path

import pytest

from st_common.licensing import LicenseViolation, check_model_allowed, license_status

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "ident",
    [
        "briaai/RMBG-2.0",
        "briaai/RMBG-1.4",
        "briaai/VRMBG-3.0",
        "someone/RMBG-2.0-onnx",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "city96/FLUX.1-dev-gguf",
        "flux1-dev-fp8.safetensors",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-9b-fp8",
        "Qwen/Qwen2.5-3B-Instruct",
        "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "CohereLabs/aya-expanse-8b",
        "CohereLabs/tiny-aya-l2-thinker",
        "universitytehran/PersianMind-v1.0",
        "openbmb/MiniCPM-V-2_6",
        r"D:\models\RMBG-2.0\model.onnx",
    ],
)
def test_blocked(ident):
    assert license_status(ident) == "blocked"
    with pytest.raises(LicenseViolation):
        check_model_allowed(ident, allow_conditional=True)


@pytest.mark.parametrize(
    "ident",
    [
        "ZhengPeng7/BiRefNet",
        "ZhengPeng7/BiRefNet_lite",
        "ZhengPeng7/BiRefNet_HR",
        "PramaLLC/BEN2",
        "plemeri/InSPyReNet",
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.2-klein-4B",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3.5-4B",
        "unsloth/Qwen3.5-4B-GGUF",
        "Qwen3.5-4B-Q4_K_M.gguf",
        "google/gemma-4-E4B-it",
        "xinntao/Real-ESRGAN",
    ],
)
def test_allowed(ident):
    assert license_status(ident) == "allowed"
    check_model_allowed(ident)


@pytest.mark.parametrize(
    "ident",
    [
        "stabilityai/stable-diffusion-3.5-large",
        "meta-llama/Llama-3.1-8B-Instruct",
        "PartAI/Dorna2-Llama3.1-8B-Instruct",
        "google/gemma-3-12b-it",
        "Qwen/Qwen3.8-Flash-Next",
    ],
)
def test_conditional_needs_signoff(ident):
    assert license_status(ident) == "conditional"
    with pytest.raises(LicenseViolation):
        check_model_allowed(ident)
    check_model_allowed(ident, allow_conditional=True)


def test_every_id_in_docs_do_not_ship_section_is_blocked():
    doc = (REPO_ROOT / "docs" / "licenses.md").read_text(encoding="utf-8")
    section = doc.split("## DO-NOT-SHIP list", 1)[1].split("\n## ", 1)[0]
    ids = re.findall(r"`([^`]+)`", section)
    ids = [i for i in ids if "/" in i or "-" in i]
    ids = [i for i in ids if not i.endswith(".py")]
    assert len(ids) >= 12
    for ident in ids:
        probe = ident.replace("*", "x")
        assert license_status(probe) == "blocked", ident
