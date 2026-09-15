# Licence audit — open-weight model candidates

**Audit date:** 2026-09-15. **Method:** licence tags and `license_name`/`license_link` pulled from the
Hugging Face model API for each repo, then the primary licence text fetched and read for every
model whose tag was `other`, missing, or research/non-commercial. Rows marked *(tag only)* rely on
the repo's licence tag plus a well-known upstream licence; everything else was read at the source.

> This is an engineering audit, not legal advice. Items needing counsel are in `HUMAN_NEEDED.md`
> (H-002 sanctions/export control, H-007 training-data provenance, H-008 Stability trade-control clause).

**Verdict legend**

| Verdict | Meaning |
|---|---|
| **SHIP-OK** | Permissive (MIT / Apache-2.0 / BSD). Keep the notice file. |
| **SHIP-OK+OBL** | Commercial use allowed, but with obligations we must implement (use-restrictions pass-through, attribution). |
| **CONDITIONAL** | Commercial use only under thresholds/registration/policies we can't guarantee long-term. Flagged; not a default. |
| **DO-NOT-SHIP** | Non-commercial, research-only, needs a paid agreement we don't have, or licence text unverifiable. Enforced in code. |

---

## 1. Background removal / matting (the core product)

| Model | Licence | Commercial use | Source | Verdict | Notes |
|---|---|---|---|---|---|
| BiRefNet (general, Swin-L, 1024²) | MIT | yes | https://huggingface.co/ZhengPeng7/BiRefNet · code https://github.com/ZhengPeng7/BiRefNet | **SHIP-OK** | Weights + code MIT. Trained on academic DIS/HRSOD-style datasets → provenance question H-007. |
| BiRefNet_lite (Swin-T) | MIT | yes | https://huggingface.co/ZhengPeng7/BiRefNet_lite | **SHIP-OK** | Same caveat. |
| BiRefNet_dynamic (variable res) | MIT | yes | https://huggingface.co/ZhengPeng7/BiRefNet_dynamic | **SHIP-OK** | Same caveat. |
| BiRefNet_HR (2048²) | MIT | yes | https://huggingface.co/ZhengPeng7/BiRefNet_HR | **SHIP-OK** | Same caveat. |
| BiRefNet-matting / _HR-matting / _lite-matting | MIT | yes | https://huggingface.co/ZhengPeng7 | **SHIP-OK** | Same caveat. |
| BEN2 **Base** | MIT | yes | https://huggingface.co/PramaLLC/BEN2 · https://github.com/PramaLLC/BEN2 | **SHIP-OK** | Only the *Base* model is open. The "full" BEN2 is a commercial API (not used). Card: trained on DIS5K + a 22K proprietary set (H-007). |
| InSPyReNet (via `transparent-background`) | MIT (code and package) | yes | https://github.com/plemeri/InSPyReNet · https://github.com/plemeri/transparent-background (weights: GitHub release 1.2.12) | **SHIP-OK** | Weights are distributed as GitHub release assets of the MIT repo. H-007 applies. |
| RMBG-1.4 (BRIA) | `bria-rmbg-1.4` (BRIA model licence) | **no** without a BRIA commercial agreement | https://huggingface.co/briaai/RMBG-1.4 · https://bria.ai/bria-huggingface-model-license-agreement/ | **DO-NOT-SHIP** | |
| RMBG-2.0 (BRIA) | CC BY-NC 4.0 (card's `license_link`) | **no** | https://huggingface.co/briaai/RMBG-2.0 | **DO-NOT-SHIP** | Same architecture as BiRefNet, so there's little reason to want it. |
| VRMBG-3.0 (BRIA, video) | `bria-vrmbg-3.0`, gated | **no** — "commercial use requires a BRIA AI agreement" | https://huggingface.co/briaai/VRMBG-3.0 | **DO-NOT-SHIP** | |
| Trendyol background-removal | CC BY-SA 4.0 *(tag only)* | conditional (attribution; ShareAlike on adapted weights) | https://huggingface.co/Trendyol/background-removal | **CONDITIONAL** | Not evaluated. |

## 2. Generation / inpainting (background replacement, future features)

| Model | Licence | Commercial use | Source | Verdict | Notes |
|---|---|---|---|---|---|
| SDXL-inpainting 0.1 | CreativeML OpenRAIL++-M | yes, with use-based restrictions | https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 | **SHIP-OK+OBL** | Must pass the licence's use restrictions through to users (ToS clause). Fits 24 GB comfortably. |
| SDXL base 1.0 | CreativeML OpenRAIL++-M | yes, with use-based restrictions | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 | **SHIP-OK+OBL** | Same. |
| FLUX.1-schnell | Apache-2.0 | yes | https://huggingface.co/black-forest-labs/FLUX.1-schnell | **SHIP-OK** | HF repo is click-through gated; a human must accept (I don't accept terms). ~12B transformer → needs FP8/offload on 24 GB. |
| FLUX.1-dev (and Fill/Depth/Canny/Redux/Kontext [dev]) | FLUX.1 [dev] Non-Commercial License v1.1.1 | **no** ("non-commercial and non-production use") | https://huggingface.co/black-forest-labs/FLUX.1-dev · https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev | **DO-NOT-SHIP** | The licence itself names the Fill/Kontext variants. |
| FLUX.2-klein-4B (and base-4B, fp8/nvfp4) | Apache-2.0 | yes | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B | **SHIP-OK** | Released 2026-01; licence verified from the card, model **not evaluated**. The most interesting generation candidate for 24 GB. |
| FLUX.2-klein-9B (and base-9B, 9b-kv) | FLUX Non-Commercial License | **no** | https://huggingface.co/black-forest-labs/FLUX.2-klein-9B | **DO-NOT-SHIP** | Easy to confuse with the 4B. The registry blocks it by repo id. |
| Stable Diffusion 3.5 Large / Medium | Stability AI Community License | conditional | https://huggingface.co/stabilityai/stable-diffusion-3.5-large · https://stability.ai/community-license-agreement | **CONDITIONAL** — flagged, kept | Commercial use only while company + affiliates earn < US$1M/yr; **must register** with Stability; "Powered by Stability AI" attribution; outputs can't be used to train foundation models; **must comply with Trade Control Laws** (H-008). Workaround is an Enterprise licence → expensive/uncertain, so kept and flagged per brief. |

## 3. Upscaling

| Model | Licence | Commercial use | Source | Verdict | Notes |
|---|---|---|---|---|---|
| Real-ESRGAN (RealESRGAN_x4plus / x2plus, official weights) | BSD-3-Clause | yes | https://github.com/xinntao/Real-ESRGAN (weights in its GitHub releases) | **SHIP-OK** | Keep the BSD notice. Use the official xinntao release, **not** `ai-forever/Real-ESRGAN` (no licence tag on that mirror). |

## 4. Persian-capable LLM / VLM (listing generator)

| Model | Licence | Commercial use | Source | Verdict | Notes |
|---|---|---|---|---|---|
| Qwen3.5-2B / 4B / 9B | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3.5-4B (etc.) | **SHIP-OK** | Native image-text-to-text. GGUF (unsloth) + mmproj available. **Chosen dev default** (see ADR in decisions.md). |
| Qwen3.6-27B / Qwen3.8-27B | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3.8-27B | **SHIP-OK** | 27.8B params; 24 GB only at ~4-bit and alongside nothing else. |
| Qwen3.8-Flash-Next | Qwen Community License 1.0 | conditional | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | **CONDITIONAL** | Separate licence needed if we are a "Model as a Service" or "AI Work Assistant" business, and name display above 100M MAU / US$20M monthly revenue. We're neither, but 180B params makes it irrelevant for 24 GB anyway. |
| Qwen3-VL-2B / 4B / 8B-Instruct | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | **SHIP-OK** | Official GGUFs exist. |
| Qwen3-4B / Qwen3-8B | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-8B | **SHIP-OK** | Text-only. |
| Qwen2.5-7B-Instruct / Qwen2.5-1.5B-Instruct | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | **SHIP-OK** | |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | **SHIP-OK** | |
| **Qwen2.5-3B-Instruct** | **Qwen RESEARCH License** | **no** | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE | **DO-NOT-SHIP** | Trap: its 1.5B and 7B siblings are Apache. |
| **Qwen2.5-VL-3B-Instruct** | **Qwen RESEARCH License** | **no** | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE | **DO-NOT-SHIP** | Same trap. |
| Gemma 4 (E2B / E4B / 12B / 31B) | Apache-2.0 (Google's Gemma 4 licence page is Apache 2.0) | yes | https://huggingface.co/google/gemma-4-E4B-it · https://ai.google.dev/gemma/docs/gemma_4_license | **SHIP-OK** | Gemma 4 moved from the Gemma Terms to Apache. Self-hosted weights, no Google API dependency. 140+ languages. |
| Gemma 3 (4B / 12B) | Gemma Terms of Use + Prohibited Use Policy | conditional | https://huggingface.co/google/gemma-3-12b-it | **CONDITIONAL** | Superseded by Gemma 4 (Apache). Prefer Gemma 4. |
| Llama 3.1 8B Instruct | Llama 3.1 Community License | conditional | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | **CONDITIONAL** | < 700M MAU, Acceptable Use Policy, "Built with Llama" attribution, manual gated approval. |
| Dorna2-Llama3.1-8B-Instruct (PartAI, Persian fine-tune) | Llama 3.1 Community License (inherited) | conditional | https://huggingface.co/PartAI/Dorna2-Llama3.1-8B-Instruct | **CONDITIONAL** | Strong Persian candidate; inherits Llama conditions. Text-only. |
| Dorna-Llama3-8B-Instruct | Llama 3 Community License | conditional | https://huggingface.co/PartAI/Dorna-Llama3-8B-Instruct | **CONDITIONAL** | Superseded by Dorna2. |
| Mistral-7B-Instruct-v0.3 | Apache-2.0 | yes | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 | **SHIP-OK** | Weak Persian; not a real candidate. |
| Phi-3.5-vision-instruct | MIT | yes | https://huggingface.co/microsoft/Phi-3.5-vision-instruct | **SHIP-OK** | Weak Persian. |
| Cohere North-Micro-Vision-Instruct | Apache-2.0 | yes | https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct | **SHIP-OK** | Card's language list has Arabic but not Persian. Not evaluated. |
| **Aya Expanse 8B / Aya Vision 8B / tiny-aya** | **CC BY-NC 4.0** | **no** | https://huggingface.co/CohereLabs/aya-expanse-8b | **DO-NOT-SHIP** | Probably the best Persian quality in its size class. Excluded anyway. |
| **PersianMind-v1.0** | **CC BY-NC-SA 4.0** | **no** | https://huggingface.co/universitytehran/PersianMind-v1.0 | **DO-NOT-SHIP** | |
| **PersianLLaMA-13B** | **CC BY-NC 4.0** | **no** | https://huggingface.co/ViraIntelligentDataMining/PersianLLaMA-13B | **DO-NOT-SHIP** | |
| **MiniCPM-V 2.6** | MiniCPM Model License — **text could not be retrieved** (gated repo; upstream link 404) | unverified | https://huggingface.co/openbmb/MiniCPM-V-2_6 | **DO-NOT-SHIP** (until reviewed) | Unverifiable licence = don't ship. |
| Community Persian fine-tunes tagged Apache/MIT on a Llama/Gemma-3 base (e.g. `mshojaei77/gemma-3-4b-persian-v0`, `zpm/Llama-3.1-PersianQA`) | Tag conflicts with base licence | treat as base licence | — | **CONDITIONAL** | A fine-tune can't relicense its base. |

---

## DO-NOT-SHIP list

Machine-readable copy: `common/st_common/licensing.py` (`DO_NOT_SHIP`). The inference model
registry refuses to load any of these, and a unit test enforces it. Use of these weights is not
allowed in production, staging, demos or benchmarks.

1. `briaai/RMBG-1.4`: BRIA licence, commercial agreement required
2. `briaai/RMBG-2.0`: CC BY-NC 4.0
3. `briaai/VRMBG-3.0`: commercial agreement required
4. `black-forest-labs/FLUX.1-dev`: FLUX.1 [dev] Non-Commercial
5. `black-forest-labs/FLUX.1-Fill-dev`, `FLUX.1-Depth-dev`, `FLUX.1-Canny-dev`, `FLUX.1-Redux-dev`, `FLUX.1-Kontext-dev`: same licence
6. `black-forest-labs/FLUX.2-klein-9B`, `FLUX.2-klein-base-9B`, `FLUX.2-klein-9b-kv` (+ fp8/nvfp4 variants): FLUX Non-Commercial
7. `Qwen/Qwen2.5-3B-Instruct`: Qwen Research License
8. `Qwen/Qwen2.5-VL-3B-Instruct`: Qwen Research License
9. `CohereLabs/aya-expanse-8b`, `CohereLabs/aya-vision-8b`, `CohereLabs/tiny-aya-*`, `CohereLabs/North-Small-Translate-1.0*`: CC BY-NC 4.0
10. `universitytehran/PersianMind-v1.0`: CC BY-NC-SA 4.0
11. `ViraIntelligentDataMining/PersianLLaMA-13B`: CC BY-NC 4.0
12. `openbmb/MiniCPM-V-2_6`: licence text unverifiable

## CONDITIONAL (flagged, allowed only with sign-off)

- `stabilityai/stable-diffusion-3.5-*`: revenue cap, registration, trade-control clause (H-008)
- `meta-llama/Llama-3.1-*`, `PartAI/Dorna*`: Llama Community License + AUP
- `google/gemma-3-*`: Gemma Terms (use Gemma 4 instead)
- `Qwen/Qwen3.8-Flash-Next`: Qwen Community License 1.0
- `Trendyol/background-removal`: CC BY-SA 4.0

---

## Runtime dependencies (non-model)

Licences below come from each project's published licence *(well-known; re-verify
automatically with `pip-licenses` in CI — TODO in `tools/tasks.py lint`)*. The Redis, aiogram and
python-telegram-bot rows were read at source.

| Component | Licence | Commercial use | Notes |
|---|---|---|---|
| PyTorch, torchvision | BSD-3-Clause | yes | |
| ONNX Runtime | MIT | yes | |
| NVIDIA CUDA / cuDNN / TensorRT | NVIDIA proprietary SLAs | yes (deployment); redistribution limited; **export-control clauses** | H-002. TensorRT is optional: the service runs on ONNX Runtime CUDA EP or plain PyTorch. |
| llama.cpp (`llama-server`) | MIT | yes | Local LLM runtime. |
| transformers, timm, kornia | Apache-2.0 | yes | |
| FastAPI | MIT | yes | |
| uvicorn | BSD-3-Clause | yes | |
| redis-py | MIT | yes | Client only. |
| **Valkey 8** (queue server) | BSD-3-Clause | yes | Chosen over Redis. |
| Redis ≥ 8 | RSALv2 / SSPLv1 / AGPLv3 (tri) | internal use likely fine | **Avoided** (ADR-004). Redis ≤ 7.2 remains BSD. |
| aiogram 3 | MIT | yes | Telegram framework used. |
| python-telegram-bot | LGPLv3 / GPLv3 dual | yes as a library | Not used (it's installed globally on the build box, not in the project venv). |
| prometheus-client | Apache-2.0 | yes | |
| OpenCV (≥ 4.5) | Apache-2.0 | yes | |
| Pillow | MIT-CMU (HPND) | yes | |
