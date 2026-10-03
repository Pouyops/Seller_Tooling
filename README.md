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
| **Telegram bot** (`bot/`) | Core done: Jalali-month quota, rate limiting, credit ledger with refunds, stub payments, Persian texts. **Handlers and end-to-end flow are the next task.** |
| **Persian text** (`common/`) | ZWNJ (نیم‌فاصله) repair, ی/ک normalization, digit forms, bidi handling for mixed LTR/RTL, validation. 100% line coverage. |
| **Listing generator** | Not started (next after the bot). Local LLM over an OpenAI-compatible llama.cpp server. |
| **Economics model** | Not started — blocked on clean speed numbers (see below). |

### Measured results so far (GTX 1650 4 GB, fp32, synthetic set, 200 images)

| model | balanced IoU | BF | BF@3px | thin recall | p50 latency |
|---|---|---|---|---|---|
| `birefnet` (Swin-L) | 0.939 | 0.891 | 0.816 | 0.689 | 3.3 s |
| `birefnet_lite` (Swin-T) | 0.909 | 0.855 | 0.780 | 0.702 | 0.60 s |

Four more models (BEN2, InSPyReNet ×2, BiRefNet_HR) had not finished when the run was interrupted;
`make bench` resumes where it stopped. Throughput/latency numbers from that run are **contaminated**
by concurrent test runs and must be re-measured with `--speed-only` on an idle machine.

**fp16 is unusable on GTX 16xx** (all-NaN mattes — the known half-precision bug); the loader detects
this and falls back to fp32, recording the reason in every result. On a 3090/4090 fp16 is used after
a numeric probe (ADR-008).

## Layout

```
common/      st_common    Persian text, licence guard, job queue, blob store, marketplace stubs
inference/   st_inference FastAPI API, GPU worker, model adapters/registry, imaging
bot/         st_bot       Telegram bot (aiogram): quota, ledger, payments stub, Persian UI
eval/        st_eval      synthetic benchmark generator, metrics, bench runner, report
web/                      (placeholder) RTL upload page
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
| `make serve` | Valkey + inference API + GPU worker |
| `make bot` | Telegram bot (needs `ST_TELEGRAM_TOKEN`) |
| `make up` / `make down` | docker compose |

On Windows without GNU make, use `.\make.cmd <target>` — it runs the same `tools/tasks.py` code.

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

1. Bot handlers + end-to-end flow against the local mock Bot API server (no Telegram token needed).
2. `--speed-only` re-measurement of all models on an idle machine, then write `eval/analysis.md`.
3. Persian listing generator (Qwen3.5-4B or Gemma-4-E4B GGUF via llama.cpp, both Apache-2.0).
4. ONNX export + TensorRT trial; load test with real numbers → `docs/loadtest.md`.
5. `docs/economics.md` from the measured throughput.

## Ground rules this repo follows

- No model on the DO-NOT-SHIP list ships, even temporarily — enforced in code, not by convention.
- No scraping of Torob/Digikala/Basalam: interfaces and stubs only, pending a ToS review (H-003).
- No real payment gateway, no account registration, no money spent.
- No fabricated numbers. Anything not measured says so.

© 2026. All rights reserved. Third-party model licences are catalogued in `docs/licenses.md`.
