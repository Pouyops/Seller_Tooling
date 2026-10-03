# Handoff

For whoever picks this up next, human or agent. Written 2026-10-03, at commit `e6eb31a`
(20 commits, 129 files, 319 tests passing).

---

## 1. What this is, in one paragraph

Image and listing tooling for Iranian e-commerce sellers: send a product photo, get a background-free
cutout and a Persian listing. Everything is open-weight and self-hosted — no OpenAI/Anthropic/Google
API, nothing that can be geo-blocked at runtime. Target deployment is one 24 GB consumer GPU. The
foundations are built and measured: licence audit, benchmark harness, inference service, Telegram bot,
listing generator, economics model. **Nothing has met a real seller yet**, and every accuracy number so
far comes from a *synthetic* benchmark.

## 2. Read these first, in this order

1. **`HUMAN_NEEDED.md`** — 14 open decisions, each with options, a recommendation, and what it blocks.
   Do not re-litigate these silently; if you decide one, edit the entry and say so.
2. **`README.md`** — what exists, how to run it, measured results.
3. **`docs/decisions.md`** — 13 ADRs. Every non-obvious choice is there with its reasoning.
4. **`eval/analysis.md`** — the model comparison read in prose. The tables above it are generated.
5. This file, section 6 — the traps. They cost hours to find and they are not obvious from the code.

## 3. This machine

| | |
|---|---|
| GPU | **GTX 1650, 4 GB** — not the 24 GB target (H-001). fp16 is broken on it (see §6). |
| Python / venv | `D:\seller-tooling\venv` (Python 3.11). Repo is on `C:`, which has <7 GB free. |
| `ST_DATA_DIR` | `D:\seller-tooling` — weights, datasets, blobs, ONNX, LLM GGUFs. Never in git. |
| Queue | Valkey in Docker (`make up`). Docker Desktop is often *not* running; code falls back where it can. |
| Real GPU benchmarks | Kaggle T4 via `tools/kaggle_bench.py` (§5). The laptop GPU is for smoke tests only. |
| Live bot | **@BizBizAI_bot**, token in `.env` (gitignored). |

## 4. Run it

```bash
# one process: API + GPU worker + web page (+ bot with --bot). Easiest way to see it work.
python tools/serve_demo.py --bot          # http://127.0.0.1:8000/

make up            # Valkey (durable queue). Without it, serve_demo falls back to in-process.
make test          # 319 tests, no GPU/weights/network needed
make serve         # the real layout: separate API + worker processes
make bench         # full model comparison -> eval/results.md (GPU, resumable)
make economics     # recompute docs/economics.md from the measured runs
python tools/listing_demo.py --image <photo> --preset carpet   # needs llm_server running
python -m st_inference.listing.llm_server                       # llama.cpp + Qwen3.5-2B
```

On Windows without GNU make: `.\make.cmd <target>` — same `tools/tasks.py` code.

## 5. Benchmarking on Kaggle (the only real GPU available)

```bash
python tools/kaggle_bench.py push --models birefnet_lite,inspyrenet_base
python tools/kaggle_bench.py status --wait 90
python tools/kaggle_bench.py fetch --out eval/runs/<name>
python -m st_eval.report eval/runs/<name> --out eval/results.md
```

It pushes a **private** kernel that clones the public GitHub repo at your current HEAD — so
**commit and push before running**, or you benchmark old code (the tool checks).

## 6. Traps — read before debugging anything

**Hardware / precision**
- **fp16 returns all-NaN on GTX 16xx.** Caught by `device.fp16_unreliable` plus a numeric probe in
  `MattingModel.load` (ADR-008). The first benchmark scored every BiRefNet row 0.000 because of this.
- **InSPyReNet is fp32-only.** `.half()` → `expected scalar type Float but found Half`; autocast → NaN.
  Both caught automatically. Don't "fix" it by removing the probe.
- **transformers ≥ 5 keeps the stored weight dtype.** Asking for fp32 does nothing unless you call
  `.float()` explicitly — we do, in `MattingModel.load`.
- **Windows/WDDM pages VRAM into system RAM instead of raising OOM.** A model that "fits" in 4 GB may
  be 50× slower. The benchmark detects spill by comparing allocator peak against physical VRAM.

**Measurement hygiene**
- **Never run tests or other heavy work while benchmarking.** An early run reported 0.033 img/s because
  pytest was competing for memory; the whole run had to be redone.
- **The API deduplicates by content hash.** Load tests must salt each image or they measure the cache
  (~100× faster) and nothing else. `unique_images(..., salt=...)` exists for this.
- **Kaggle T4 throughput varies ±17% run to run.** Don't chase smaller differences.

**Platform**
- **`.env` must not have trailing comments.** `KEY=value  # note` stores the comment *in the value*.
  This silently broke the live bot (`ST_TELEGRAM_API_BASE` became a comment string).
- **HuggingFace cache symlinks fail on Windows** without Developer Mode. Weights download with
  `local_dir=` into `$ST_DATA_DIR/models/<org>__<repo>` (ADR-007).
- **`pip install -e` hangs** on this box from build isolation fetching setuptools; use
  `--no-build-isolation`.
- **Kaggle from Iran:** `api.kaggle.com` is 403-blocked (Google frontend), so the modern `kaggle` CLI
  and its `KGAT_` tokens cannot authenticate. The **legacy REST API at `www.kaggle.com/api/v1` works**
  with a classic username+key. Kernel *output files* are also CDN-blocked, so results come home as
  gzipped base64 chunks in the kernel **log**. Kaggle derives the kernel slug from the **title**, not
  the slug field; the real slug is recorded in `.kaggle-run.json`.

**Python/async**
- On 3.11, `asyncio.wait_for` can swallow a cancellation and leave a task alive — it hung the whole test
  suite at event-loop teardown. Use `async with asyncio.timeout(...)`.
- Debug hangs with `pytest -o faulthandler_timeout=30`; it dumps the stuck stack.
- Don't let `torch.cuda.empty_cache()` run in a process that hasn't initialised CUDA — it creates a
  context and can take a minute.

**Persian / LLM**
- **Persian costs ~2 tokens per character** in Qwen's tokenizer. A 700-token cap truncated mid-JSON and
  looked like malformed output (ADR-012). Default is 1500.
- Small models **copy a few-shot example verbatim** when it shares the product's category; the example in
  `prompt.py` is deliberately an unrelated category.
- They also pad descriptions with a repeated sentence — detected, one rewrite requested, then shipped
  with the defect recorded in `issues`.

## 7. State of each deliverable

| | state | evidence |
|---|---|---|
| Repo scaffold, Docker, Makefile | done | `make up` verified against a healthy Valkey |
| Licence audit | done | `docs/licenses.md`; DO-NOT-SHIP enforced in `st_common.licensing` + tests |
| Benchmark set (200 synthetic) | done | deterministic from seed; `docs/dataset-gap.md` says what real data is still needed |
| Eval harness | done | `eval/results.md` — 9 configs on a T4, resumable, spill/OOM aware |
| Inference service | done | `docs/loadtest.md` — 4 scenarios, no 5xx through a worker outage |
| Telegram bot | **live** | 15 e2e tests + @BizBizAI_bot running on a real queue |
| Listing generator | done | real Persian output verified; quality needs H-013 |
| Economics | done | `docs/economics.md` from measured runs; prices are assumptions (H-014) |
| ONNX / TensorRT | **not viable** on this toolchain, documented | `docs/onnx-tensorrt.md`, ADR-013 |
| Web page | done | RTL, served at `/` |

**Not verified anywhere:** real seller photos, the 24 GB target GPU, a real payment gateway, marketplace
field limits, and whether the Persian reads well to a native speaker.

## 8. What I'd do next

1. **Watch the live bot with real photos.** Every accuracy claim rests on synthetic data (H-005). The
   bot has a quota and rate limit; let real sellers hit it and collect failures.
2. **Pipeline CPU encode/store behind GPU inference.** Measured as worth more than a 3× faster GPU
   (`docs/economics.md` §4): inference is 64% of the pipeline, encode+store is most of the rest.
3. **Verify economics inputs** (H-014) and re-run `make economics`.
4. **Native Persian review of ~50 listings** (H-013) before the listing feature ships to anyone.
5. **Sanctions/export review** (H-002) before any production deployment decision.

## 9. Conventions to keep

- **Never report a number you didn't measure.** If a run didn't happen, the doc says so. `docs/` files
  are generated from run JSON; hand-written interpretation lives in a separate `*-analysis.md` that the
  renderer appends, so regenerating never destroys reasoning.
- **Measured vs assumed is always explicit** (see `docs/economics.md` §1 vs §2).
- **DO-NOT-SHIP models are blocked in code**, not by convention, and a test checks the code against the
  licence doc. Don't add a model without a licence row.
- **Marketplace scraping stays stubbed** until H-003 is answered. `st_common.marketplace` adapters raise
  `SourceBlocked` and make no requests.
- **No real payment gateway, no account registration, no spending.**
- When something fails, write the failure down with its exact error — `docs/onnx-tensorrt.md` is more
  useful as a record of what doesn't work than it would have been as a success story.

## 10. Live right now

- **@BizBizAI_bot** — polling, via `tools/serve_demo.py --bot` in a terminal tab. Ctrl-C stops it.
- **Valkey** — `docker compose -f infra/docker-compose.yml` container, AOF on.
- **Kaggle kernels** — `seller-tooling-bench`, `seller-tooling-fp16`, `seller-tooling-amp`,
  `st-smoke-test` (all private, on the `pooyaebrahimi` account).
- `.env` holds a real Telegram token and a Kaggle token. It is gitignored. The Telegram token has
  appeared in a terminal error message locally; rotate it if that scrollback is ever shared.
