# Seller Tooling — image & listing tooling for Iranian e-commerce sellers

Background removal and Persian listing generation for sellers on Digikala, Basalam, Torob and
Instagram shops. Every model is open-weight and self-hosted: **no OpenAI / Anthropic / Google API,
nothing that can be geo-blocked at runtime.** Target deployment is a single 24 GB consumer GPU
(RTX 3090/4090).

> **Status: foundations built, not production.** Read `HUMAN_NEEDED.md` first — it lists every open
> decision with options and a recommendation. All benchmark numbers so far were measured on a
> **GTX 1650 (4 GB)**, not the target hardware (H-001), and on a **synthetic** benchmark set (H-005).

## What exists today

| Area | State |
|---|---|
| **Licence audit** (`docs/licenses.md`) | ~50 candidate models checked at the source. DO-NOT-SHIP list is enforced in code (`st_common.licensing`) and unit-tested against the doc. |
| **Benchmark set** (`eval/st_eval/synth/`) | 200 procedurally rendered images with exact alpha ground truth, across six hard Iranian categories (carpet fringe, gold chains, saffron threads, glassware, handicrafts, clothing on hangers). Deterministic from a seed. |
| **Eval harness** (`eval/`) | IoU, boundary F (DAVIS + strict 3 px), thin-structure recall, alpha MAE; latency p50/p90/p99, VRAM peak, throughput sweeps with OOM **and VRAM-spill** detection; resumable; renders `eval/results.md`. |
| **Inference service** (`inference/`) | FastAPI + Redis-Streams queue + GPU worker. Content-hash dedupe, micro-batching, OOM batch splitting, crash recovery, disk spool when Redis is down, `/healthz`, `/metrics`, JSON logs. |
| **Telegram bot** (`bot/`) | Working end to end: photo (or image file) in → PNG cutout out, albums, Jalali-month quota, rate limiting, credit ledger with refunds, stub payments, Persian UI, deferred delivery that survives a restart. Verified against a mock Bot API — **no token exists yet** (H-004). |
| **Persian text** (`common/`) | ZWNJ (نیم‌فاصله) repair, ی/ک normalization, digit forms, bidi handling for mixed LTR/RTL, validation. 100% line coverage. |
| **Unit economics** (`docs/economics.md`) | Built from measured throughput, payload sizes and pipeline timings; business prices are declared assumptions (H-014). Headline: **~5 toman/image**, so compute is ~0.1% of a 50k subscription. |
| **Listing generator** (`inference/st_inference/listing/`) | Working: product photo + seller fields → Persian title, description, attributes, keywords. Local Qwen3.5-2B (Apache-2.0) over llama.cpp; validate-and-repair with a deterministic fallback so a listing is never empty. Model quality needs a bigger model and a native-speaker review (H-013). |


### Measured results (Tesla T4 16 GB, synthetic set, 200 images) — full tables in [`eval/results.md`](eval/results.md)

| model | precision | Q | IoU | BF@3px | thin recall | p50 latency | img/s | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| `inspyrenet_base` | fp32 | **0.865** | 0.908 | 0.833 | 0.818 | 649 ms | 1.5 | 2.7 GB |
| `ben2_base` | amp | 0.844 | 0.916 | 0.844 | 0.708 | 546 ms | 1.8 | 2.4 GB |
| `birefnet` | fp16 | 0.834 | 0.940 | 0.816 | 0.690 | 371 ms | 2.7 | 1.6 GB |
| `birefnet_lite` | fp16 | 0.811 | 0.909 | 0.780 | 0.702 | **122 ms** | **7.8** | 0.9 GB |
| `inspyrenet_fast` | fp32 | 0.780 | 0.905 | 0.730 | 0.641 | 82 ms | 11.7 | 1.0 GB |

Recommendation (reasoning in [`eval/analysis.md`](eval/analysis.md)): **`birefnet_lite` fp16 for the
free tier, `inspyrenet_base` for paid/hard categories** — the queue already routes per-job model and priority.

Three findings worth knowing before touching this code:
- **fp16 costs nothing and saves 2.6–2.8×** on tensor-core GPUs: identical Q to fp32, far faster. But it
  produces **all-NaN mattes on GTX 16xx**, so the loader probes and falls back (ADR-008).
- **Batching currently doesn't help** (throughput is flat to falling from bs=1 to bs=16) because
  preprocessing resizes on the CPU, one image at a time. Fix that before judging batch sizes.
- **Loose saffron threads defeat every model** (thin recall 0.02–0.52) — a fine-tuning target, not a
  model-selection one.

## Layout

```
common/      st_common    Persian text, licence guard, job queue, blob store, marketplace stubs
inference/   st_inference FastAPI API, GPU worker, model adapters/registry, imaging
bot/         st_bot       Telegram bot (aiogram): quota, ledger, payments stub, Persian UI
eval/        st_eval      synthetic benchmark generator, metrics, bench runner, report
web/         index.html   RTL seller page (no framework, no CDN), served by the API at /
infra/                    docker-compose (Valkey, Prometheus, optional GPU/bot/llm profiles)
docs/                     licenses.md · decisions.md (ADRs) · dataset-gap.md · economics.md (todo)
tools/tasks.py            the real task runner behind `make`
HUMAN_NEEDED.md           every open decision, with options and a recommendation
```

## Setup

Requirements: Python ≥ 3.11, an NVIDIA GPU for inference (CPU works for everything except the
benchmark), Docker for Valkey.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cp .env.example .env          # then set ST_DATA_DIR to a disk with ~15 GB free
make setup                    # Windows: .\make.cmd setup
```

`make setup` installs the packages editable, downloads the licence-cleared weights
(~2.4 GB, pinned, MD5-checked) and builds the benchmark set (~5 min on 4 CPU workers).
**Weights, datasets and job blobs live under `ST_DATA_DIR`, never in git.**

### Targets

| Target | What it does |
|---|---|
| `make test` | unit tests (no GPU, no weights, no network) |
| `make bench` | full model comparison → `eval/results.md` (resumable; GPU) |
| `make bench-quick` | 2 models × 24 images smoke check |
| `make bench-kaggle` | run the benchmark on a free Kaggle T4 (16 GB) and fetch the results |
| `make economics` | recompute `docs/economics.md` from the measured runs |
| `make serve` | Valkey + inference API + GPU worker |
| `make bot` | Telegram bot (needs `ST_TELEGRAM_TOKEN`) |
| `make up` / `make down` | docker compose |

On Windows without GNU make, use `.\make.cmd <target>` — it runs the same `tools/tasks.py` code.

## Persian listing generator

```bash
python -m st_inference.listing.llm_server                  # llama.cpp + Qwen3.5-2B (vision)
python tools/listing_demo.py --image <photo> --preset carpet
curl -F file=@photo.jpg -F category=زعفران -F brand=سحرخیز http://localhost:8000/v1/listing
```

Real output from the carpet photo above (Qwen3.5-2B on a GTX 1650, 19.6 s, first attempt):

> **عنوان:** فرش دستباف طرح هندسی کاشان پشمی لاکی بدون لک پارگی ۱۲ متری
> **توضیحات:** فرش دستباف با طرح ستاره‌ای و رنگ‌های جذاب، با جنس پشم طبیعی و رنگ لاکی. تولید شده در کاشان با ضمانت عدم لک و پارگی.

Every string is normalized (ZWNJ, ی/ک, digit forms) and validated as Persian before a seller sees it.
Output is classified as `model`, `model_repaired` or `template_fallback`, so you always know whether a
model actually wrote it. Three defects seen in testing and handled: small models copy a same-category
few-shot example verbatim (example moved to an unrelated category), pad descriptions with a repeated
sentence (detected → one rewrite), and invent claims (prompt forbids medical/chemical claims; a native
speaker still needs to review — H-013).

## Trying it in a browser

```bash
python tools/serve_demo.py            # API + GPU worker + web page, one process, no Docker needed
python tools/serve_demo.py --bot      # ...and the Telegram bot too (needs ST_TELEGRAM_TOKEN in .env)
# then open http://127.0.0.1:8000/
```

With `make up` running (Valkey), the same command uses the real queue and quotas survive a restart;
without it, the queue runs in-process and it says so.

> **`.env` gotcha:** keep comments on their own lines. `KEY=value  # note` stores the comment as part
> of the value — that silently turned `ST_TELEGRAM_API_BASE` into a comment string and broke the bot.
> `.env.example` no longer uses trailing comments.

`make serve` is the real layout (separate API and worker, Valkey as the queue). `serve_demo.py` is the
convenience version for a laptop: if no Redis/Valkey is reachable it runs the queue in-process and says
so. Opening `web/index.html` straight from disk will *not* work — the page has no API to talk to, and it
now says exactly that instead of blaming your internet connection.

## Seeing the bot work, without a Telegram token

```bash
python tools/e2e_demo.py --image <any product photo>     # real model, real queue, mock Telegram
```

Drives the production path — handlers → service → API → queue → GPU worker → blob store → delivery —
with only Telegram's HTTP layer faked. On the dev laptop (GTX 1650, `birefnet_lite`): **18 s for the
first photo** (cold model load) and **1.3 s end to end** for the next one. `bot/tests/test_e2e_bot.py`
runs the same wiring with a CPU stand-in model, including albums, quota exhaustion, rate limiting,
refunds on failure, and deferred delivery.

## Benchmarking on a Kaggle GPU

The dev laptop's GTX 1650 (4 GB) can't answer the questions that matter: batches larger than 1 spill
into system RAM, and fp16 returns NaN on that chip. A free Kaggle kernel gives a **Tesla T4 (14.6 GB,
Linux)** — real batch sweeps, real OOM, and working fp16 tensor cores.

```bash
python tools/kaggle_bench.py push     # private kernel, clones this repo at your current HEAD
python tools/kaggle_bench.py status --wait 90
python tools/kaggle_bench.py fetch    # -> eval/runs/kaggle-synth-v1-s1403-n200/
python -m st_eval.report eval/runs/kaggle-synth-v1-s1403-n200 --out eval/results.md
```

Credentials: a classic `~/.kaggle/kaggle.json` (username + key), or `KAGGLE_USERNAME`/`KAGGLE_KEY`.

> **Note for Iran:** `api.kaggle.com` — the host the modern `kaggle` CLI uses — is network-blocked
> (403 from Google's frontend), which also makes the newer `KGAT_…` tokens unusable, since the CLI
> validates them against that host. The **legacy REST API at `www.kaggle.com/api/v1` is reachable**,
> so `tools/kaggle_bench.py` talks to it directly and doesn't use the CLI at all. See H-009.

## Continuing on another machine (or a cloud session)

Nothing machine-specific is in git. After cloning:

1. `cp .env.example .env` and point `ST_DATA_DIR` at a disk with ~15 GB free.
2. `make setup` — re-downloads weights and **regenerates the identical benchmark set** from its seed.
3. `make test` should pass without a GPU.
4. `make bench` needs a CUDA GPU. Without one, the two completed results in
   `eval/runs/synth-v1-s1403-n200/` are kept in git so analysis can continue.

### Next tasks, in order

1. Get a real Telegram token (H-004) and point the bot at it — everything else is done and tested.
2. Verify the economics price inputs (H-014) and re-run `make economics`.
3. Pipeline CPU encode/store behind GPU inference: worth more than a faster GPU (`docs/economics.md` §4).
4. Real benchmark photos (H-005), then re-run `make bench` and compare rankings against the synthetic set.
5. A native Persian speaker reviews ~50 generated listings (H-013) before the listing feature ships.

## Ground rules this repo follows

- No model on the DO-NOT-SHIP list ships, even temporarily — enforced in code, not by convention.
- No scraping of Torob/Digikala/Basalam: interfaces and stubs only, pending a ToS review (H-003).
- No real payment gateway, no account registration, no money spent.
- No fabricated numbers. Anything not measured says so.

© 2026. All rights reserved. Third-party model licences are catalogued in `docs/licenses.md`.
