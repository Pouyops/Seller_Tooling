# HUMAN_NEEDED

Decisions I should not make alone. Each entry: the decision, options with tradeoffs,
my recommendation, and what it blocks. Newest entries at the bottom. Status is one of
`OPEN`, `WORKAROUND-IN-PLACE` (I picked a reversible default so work could continue), or `CLOSED`.

---

## H-001 — Benchmarks were run on a GTX 1650 (4 GB), not the target RTX 3090/4090 (24 GB)
**Status:** WORKAROUND-IN-PLACE

The only GPU on the build machine is a GeForce GTX 1650 laptop GPU (Turing TU117, 4 GB,
no tensor cores, 50 W cap). Every latency/VRAM/throughput number in `eval/results.md`
and `docs/economics.md` is measured on that card and labelled as such. Numbers on a
24 GB Ampere/Ada card will be materially better (FP16 tensor cores, more memory for
batching); nothing here extrapolates them.

- **Option A:** Rent/borrow a 3090/4090 for one afternoon and run `make bench` there.
  The harness writes hardware info into every run, so results are directly comparable. Cost: small.
- **Option B:** Make decisions from GTX 1650 numbers. Accuracy ranking transfers (same
  weights, same inputs); speed ranking mostly transfers; absolute cost/image does not.

**Recommendation:** A, before committing to pricing. Accuracy-based model choice can proceed now.
**Blocks:** final numbers in `docs/economics.md`; batch-size tuning for production.

---

## H-002 — Sanctions / export-control review of the whole stack
**Status:** OPEN

The product serves Iranian sellers. Licences (MIT/Apache/BSD) do not restrict *who* may
use the weights, but the *hosting* of weights and dependencies does: Hugging Face,
GitHub, PyPI, Docker Hub and NVIDIA (drivers, CUDA, TensorRT) are US-based and apply
US sanctions/geo-blocking to Iranian IPs/accounts to varying degrees. NVIDIA's TensorRT/CUDA
EULAs also contain export-compliance clauses. None of this is a licence question I can answer.

- **Option A:** Mirror every artifact (wheels, weights, container images) into
  infrastructure the company controls, pinned by hash (`infra/mirror-manifest` — to be created),
  so production never pulls from US-hosted services at runtime. Get counsel to confirm
  that mirroring/usage is permissible for the company's legal situation.
- **Option B:** Pull from upstream at deploy time. Fragile (geo-blocks) and legally unreviewed.

**Recommendation:** A, plus a legal opinion. Everything is already designed to run offline
once weights are on disk (no runtime calls to any hosted API).
**Blocks:** production deployment inside Iran; choice of TensorRT (NVIDIA EULA) vs pure ONNX Runtime.

---

## H-003 — Scraping Torob / Digikala / Basalam
**Status:** OPEN — stub only, per instructions

I wrote the `MarketplaceSource` interface and adapter stubs
(`common/st_common/marketplace/`). Every adapter raises `NotImplementedError`; nothing fetches.

The ToS question: all three sites' terms (as generally published — not re-verified by me,
and I did not visit them to avoid anything resembling scraping) restrict automated collection.
Uses we'd want: (1) category/attribute taxonomies for the listing generator,
(2) real product photos for the benchmark, (3) price data.

- **Option A:** Official partner/seller APIs or data agreements (Digikala and Basalam have
  seller-side programmes). Slow, clean.
- **Option B:** Sellers export their *own* listings and opt in to share them. Clean, small, biased to our users.
- **Option C:** Scrape. Fast, legal/ToS risk, IP-block risk, reputational risk with the exact platforms we depend on.

**Recommendation:** B now (onboarding checkbox in the bot), A in parallel. Not C.
**Blocks:** real benchmark images (see H-005), taxonomy-aware attributes in the listing generator.

---

## H-004 — Telegram bot token, and Telegram as the primary channel
**Status:** WORKAROUND-IN-PLACE (bot verified end-to-end against a local mock Bot API server)

I must not register accounts, so I have no bot token. The bot is tested end-to-end against
a local mock of the Telegram Bot API (`bot/tests/mock_telegram.py`) using the real bot code, real
inference API and real model.

Separately, Telegram is filtered in Iran; users reach it through VPNs. That affects reachability,
upload speed (unreliable network assumption), and payment UX.

- **Option A:** Telegram only (brief as given).
- **Option B:** Keep the core transport-agnostic (already done: handlers call `st_bot.core`) and add
  a Bale (بله) adapter — Bale's bot API is intentionally close to Telegram's. Eitaa/Rubika are alternatives.

**Recommendation:** Create the Telegram bot now (a human runs BotFather, puts the token in `.env`),
and green-light a Bale adapter as the next channel.
**Blocks:** live bot on real Telegram.

---

## H-009 — Benchmarking on Kaggle GPUs (US-hosted compute)
**Status:** WORKAROUND-IN-PLACE — working, but it is the same risk family as H-002

The dev laptop has a GTX 1650 (4 GB). Kaggle gives 2× Tesla T4 (14.6 GB each, Linux), which is close
enough to the 24 GB target to produce meaningful batch sweeps and a real fp16 answer.
`tools/kaggle_bench.py` pushes a **private** kernel that clones the public repo at a pinned commit,
rebuilds the benchmark set from its seed, and returns the result JSON.

Two findings worth recording:
- **`api.kaggle.com` is blocked from here** (403 from Google's frontend), so the current `kaggle` CLI
  cannot authenticate at all — including the newer `KGAT_…` tokens, which it validates against that
  host. The **legacy REST API at `www.kaggle.com/api/v1` works** with a classic username + key, so
  the tool speaks to that directly and does not depend on the CLI.
- Measurements still come from US-hosted infrastructure under a Google-owned ToS, with an account
  that could be restricted at any time.

- **Option A:** Keep Kaggle for development benchmarking only; never in the production path.
- **Option B:** Rent a real 3090/4090 (RunPod, Vast.ai) for one afternoon and close H-001 properly.
  Needs a payment method that works from Iran.
- **Option C:** Buy/borrow the target card.

**Recommendation:** A now (free, works today), B before pricing is finalised, since Kaggle's T4 still
isn't the target card. Keep production inference on hardware the company controls.
**Blocks:** nothing immediately; H-001 stays open until B or C.

---

## H-010 — Payment gateway for the paid tier
**Status:** OPEN — stub only, per the brief

`bot/st_bot/payments.py` defines the `PaymentProvider` seam and a stub that creates invoices and
never takes money. Nothing real is integrated and no account was registered.

- **Option A:** An Iranian PSP (Zarinpal, Zibal, IDPay, NextPay). Works with Iranian bank cards
  (شتاب), needs a registered business, and each has its own verification callback.
- **Option B:** Card-to-card with manual confirmation. Zero integration, awful at scale, common
  in small Iranian shops.
- **Option C:** Telegram Payments. Not usable here — its providers don't serve Iranian cards.

**Recommendation:** A, choosing the PSP by which one onboards your business entity fastest; keep the
stub's "credit exactly once per invoice id" semantics, since PSP callbacks retry.
**Blocks:** charging anyone. The free tier works without it.

---

## H-011 — Price points for the credit packs
**Status:** OPEN — placeholder numbers in code

`PACKS` currently reads 49k/149k/490k toman for 50/200/1000 images. Those are placeholders I invented
to exercise the UI, not a pricing recommendation. `docs/economics.md` (next deliverable) will give the
cost floor from measured throughput; the price above it is a business decision about what Iranian
sellers will pay, which I can't derive from benchmarks.
**Blocks:** nothing technically; the numbers must not reach real users as-is.

---

## H-012 — Marketplace field limits and category taxonomies
**Status:** OPEN — conservative guesses in code

`inference/st_inference/listing/schema.py` caps titles at 70 characters, descriptions at 900 and
keywords at 12. Those are my conservative guesses. Digikala, Basalam and Torob each have their own
limits, required attributes per category, and title conventions; getting them wrong means rejected
listings.

- **Option A:** Read the limits from each seller panel (needs a seller account on each) and encode
  them per marketplace.
- **Option B:** Ask early users to paste a rejection message when one happens, and learn the rules.

**Recommendation:** A for the two that matter (Digikala, Basalam) — an afternoon of someone with
accounts. It also feeds the per-marketplace prompt.
**Blocks:** confident "ready to paste into Digikala" claims.

---

## H-013 — Which LLM, and who checks the Persian
**Status:** WORKAROUND-IN-PLACE (Qwen3.5-2B on the dev box)

The listing generator runs Qwen3.5-2B (Apache-2.0) because the dev GPU has 4 GB. It produces valid,
normalized Persian, but measurably weak writing: it padded a description with a repeated sentence
(now caught and retried), invented a pseudo-scientific claim for saffron ("جاذبه شیمیایی"), and copied
a same-category few-shot example almost verbatim until I changed the example's category.

- **Option A:** Qwen3.5-9B or Gemma-4-12B (both Apache-2.0) on the 24 GB target. Roughly 5–6 GB at
  4-bit, leaving room beside the matting model.
- **Option B:** Keep a small model and lean harder on templates.

**Recommendation:** A, and then **a native Persian speaker reviews ~50 generated listings** before any
of this reaches sellers. I can measure that the text is valid Persian; I cannot judge whether it reads
like a competent Iranian seller wrote it, and no automated metric settles that.
**Blocks:** turning the listing generator on for real users.

---

## H-014 — The price inputs behind `docs/economics.md`
**Status:** OPEN — the model is built, the business prices in it are guesses

`tools/economics.py` reads throughput, payload sizes and pipeline timings from the actual benchmark
and load-test runs, so the engineering half is evidence. The money half is not: exchange rate, GPU
purchase price in Iran, commercial electricity tariff, and object-storage/egress quotes are all
placeholders I invented (§2 of the report lists each one).

Sensitivity says only two of them matter: **utilisation** (35% → 15% costs +73%) and the
**toman/USD rate** (+50% costs +47%). Electricity is noise at Iranian tariffs.

- **Option A:** Get one real quote for each — a supplier price for a 4090, a recent commercial
  electricity bill, and an ArvanCloud/Parspack price list — then re-run `make economics`.
- **Option B:** Ship with placeholders and never quote a cost externally.

**Recommendation:** A; it is an hour of phone calls and it moves the headline number by up to 50%.
Note the conclusion is unlikely to change: compute is ~0.1% of a 50k subscription, so pricing is a
market question (H-011), not a cost-plus one.
**Blocks:** any external claim about margins or cost per image.

---

## H-005 — Real benchmark images (and the rights to use them)
**Status:** WORKAROUND-IN-PLACE (synthetic proxy set; see `docs/dataset-gap.md`)

I did not download product photos from the web (copyright, and no marketplace scraping — H-003).
The benchmark is procedurally generated with exact ground-truth alpha. It stresses the right failure
modes (fringe, thin chains, glass, low contrast) but is not photographic.

- **Option A:** Commission ~200 photos of the target categories and have them labelled
  (matte/trimap). Clean rights, costs money and 2–3 weeks.
- **Option B:** Sellers opt in to donate images (bot checkbox) + in-house labelling of a held-out 200.
- **Option C:** Academic sets (DIS5K, HRSOD, P3M). Research-licensed; fine for internal
  sanity checks per their terms *only if counsel agrees*; no Iranian categories.

**Recommendation:** B + a small A for categories sellers won't donate early (gold jewellery).
C only for internal sanity checks, after a licence read.
**Blocks:** final model choice with real-world confidence; eval numbers are proxy-only until then.

---

## H-006 — Docker Desktop's disk image is on C: (6.5 GB free)
**Status:** WORKAROUND-IN-PLACE

Building the CUDA inference image (~8–10 GB) would fill the system drive. I run Valkey and
Prometheus (small images) in Docker and the GPU services natively in the venv on D:. The
Dockerfiles are written but **the GPU image build was not run on this machine**.

- **Option A:** Move Docker Desktop's disk image to D: (Settings → Resources → Disk image location).
  One click, but it changes system configuration, so I didn't do it.
- **Option B:** Build images in CI or on the GPU server only.

**Recommendation:** A for local dev, B for release images.
**Blocks:** verifying `docker compose --profile gpu up` locally.

---

## H-007 — Training-data provenance of MIT-licensed matting weights
**Status:** OPEN

BiRefNet, InSPyReNet and BEN2-Base weights are MIT, but they were trained on academic datasets
(DIS5K, HRSOD/UHRSD-style saliency sets, and for BEN2 an undisclosed proprietary set) whose own terms
are often research-oriented. The weight licence is the author's grant. Whether dataset terms
reach trained weights is legally unsettled.

- **Option A:** Accept the risk (industry norm; rembg/BiRefNet are used commercially at scale).
- **Option B:** Fine-tune/retrain on commercially licensed + seller-consented data (H-005) and ship
  our own weights. Weeks of work, needs data first; reduces risk and improves Iranian categories anyway.

**Recommendation:** A now, B as the natural outcome of the data flywheel.
**Blocks:** nothing technical; this is a risk sign-off.

---

## H-008 — Stable Diffusion 3.5: trade-control clause, registration, revenue cap
**Status:** OPEN (SD3.5 is not used by any code path)

The Stability AI Community License requires registering with Stability for commercial use,
terminates above US$1M annual revenue, and requires compliance with "Trade Control Laws". For a
company serving Iran, that clause may make the licence unusable regardless of revenue.
SDXL-inpaint (OpenRAIL++) and FLUX.2-klein-4B / FLUX.1-schnell (Apache-2.0) cover the same needs.

**Recommendation:** Don't use SD3.5. Use SDXL-inpaint now and evaluate FLUX.2-klein-4B for background generation.
**Blocks:** nothing (alternatives exist).

---
