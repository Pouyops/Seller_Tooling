# Architecture Decision Records

Short ADRs for every non-obvious technical choice. Format: context → decision →
consequences → alternative considered. Newest at the bottom.

---

## ADR-001 — Monorepo with a shared `common/` package
**Context.** The brief fixes top-level dirs `/inference /bot /web /eval /docs /infra`.
Persian text normalization, config, logging, and the queue/job schema are needed by
the API, the GPU worker, the bot and the eval harness.
**Decision.** Add `/common` (`st_common`), a small dependency-light package. Each of
`common`, `inference`, `eval`, `bot` is its own pip-installable package; one dev venv
installs all of them editable.
**Consequences.** The bot does not import torch. Production images install only what they need.
**Alternative.** Copy the normalizer into each package (drift risk), or put it in `inference` (drags torch into the bot).

## ADR-002 — `tools/tasks.py` is the single task runner; `Makefile` and `make.cmd` are thin shims
**Context.** Build box is Windows without GNU make; CI/production is Linux.
**Decision.** Targets are implemented once in Python. `make bench` (Linux/WSL) and
`.\make.cmd bench` (Windows) call the same function.
**Alternative.** `just`/`invoke`/`nox` — extra tool to install for every contributor.

## ADR-003 — Large artifacts outside the repo under `ST_DATA_DIR`
**Context.** Weights (~1 GB per matting model, several GB per LLM), the benchmark set, TensorRT
engines and job blobs must not enter git. The build machine's system drive has <7 GB free.
**Decision.** Everything large lives under `ST_DATA_DIR` (default `D:/seller-tooling` here,
`/data` in containers). Benchmark images are *regenerated deterministically* from a seed, so the
set is reproducible without shipping files.

## ADR-004 — Valkey instead of Redis server
**Context.** Redis ≥ 7.4 is no longer BSD (RSALv2/SSPLv1, and AGPLv3 from Redis 8). Internal
unmodified use is very likely fine, but it is one more licence to explain to counsel.
**Decision.** Run **Valkey 8** (BSD-3, Linux Foundation fork, wire-compatible) in `infra/`.
Client is `redis-py` (MIT). Code says "Redis" because the protocol is Redis.
**Alternative.** Redis 7.2 (last BSD release) — works, but frozen.

## ADR-005 — Synthetic, procedurally rendered benchmark with exact ground truth
**Context.** No licensed real photos with mattes (H-005); no scraping (H-003).
**Decision.** Render 200 images of the six hard categories at 2× and area-downsample, which gives exact
alpha and silhouette. Deterministic from a seed, regenerated instead of versioned in git. Every
image carries hard-case tags so results can be sliced by failure mode.
**Consequences.** Good regression signal and failure-mode coverage; unknown correlation with real
photos. `docs/dataset-gap.md` defines the real set and a rank-correlation check.
**Alternative.** Composite real cutouts onto backgrounds (still needs licensed cutouts; GT inherits
their errors). Academic sets (DIS5K etc.) have no Iranian categories and research-only terms.

## ADR-006 — Metrics: IoU + two boundary F tolerances + thin-structure recall
**Context.** IoU is dominated by large interiors: losing an entire carpet fringe barely moves it.
**Decision.** Report IoU, DAVIS boundary F (tolerance 0.8% of the diagonal), a strict BF@3px, a
thin-structure recall (GT pixels removed by a 7-px opening), plus alpha MAE globally and in a 10-px
band. Rank by Q = mean of the category-balanced IoU, BF, BF@3px and thin recall.
**Alternative.** SAD/MSE/Grad/Conn matting metrics. Useful later for true matting; our production
output is a cutout judged mostly at the silhouette.

## ADR-007 — Weights in plain local directories, not the Hugging Face cache
**Context.** The HF cache uses symlinks; on Windows without Developer Mode `snapshot_download` fails
(`WinError 1314`). The cache layout also makes pinning and mirroring (H-002) awkward.
**Decision.** `make weights` downloads with `local_dir=` into `$ST_DATA_DIR/models/<org>__<repo>`;
URL downloads are MD5-verified. The registry points at those paths.

## ADR-008 — fp16 is opt-in per GPU: known-bad list + NaN probe, fall back to fp32
**Context.** On the GTX 1650, BiRefNet in fp16 returns all-NaN mattes (every score was 0). This is the
known GTX 16xx half-precision problem. Also, transformers ≥ 5 keeps the stored weight dtype, so
"fp32" silently stayed fp16 until the loader forced `.float()`.
**Decision.** `MattingModel.load` disables fp16 on GPUs in `device.fp16_unreliable` (GTX 16xx), and for
any other GPU runs a probe forward and falls back to fp32 on non-finite output. The effective precision
and the reason are recorded in every benchmark result and shown in the report. The benchmark also
aborts a model on any non-finite prediction instead of silently scoring zeros.
**Consequences.** Numbers on the build box are fp32; on a 3090/4090, fp16 will be used (and verified by the probe).

## ADR-009 — Detect VRAM "spill" instead of relying on OOM
**Context.** On Windows (WDDM) the NVIDIA driver's sysmem fallback pages CUDA allocations into system
RAM rather than raising OOM. BiRefNet_HR at 2048² "fit" in 4 GB at 28 s/image.
**Decision.** Compare the allocator peak with physical VRAM. Batch sizes that exceed it are flagged
"spill", excluded from best-throughput, and larger batches are skipped. Very slow models get a
category-balanced image cap (`birefnet_hr#24`), and the report marks them partial.
**Alternative.** Disable sysmem fallback in the NVIDIA control panel (system setting; not changed by me)
or run on Linux, where OOM is raised normally.
