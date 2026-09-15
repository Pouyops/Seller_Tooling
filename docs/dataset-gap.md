# Dataset gap: what the synthetic benchmark covers, and what real data is still needed

**Status (2026-09-15):** every accuracy number in `eval/results.md` comes from **`synth-v1`**, a
procedurally rendered proxy set of 200 images. I did not source real product photos: no
licensed or seller-consented photo set exists yet, marketplace scraping is off the table
(HUMAN_NEEDED H-003), and downloading arbitrary web photos would create copyright problems in a
commercial benchmark (H-005).

## 1. What `synth-v1` is

| | |
|---|---|
| Generator | `eval/st_eval/synth/` (deterministic: same seed → byte-identical set) |
| Size | 200 images: carpet 40, gold jewelry 40, saffron & packaged 30, glassware 30, handicrafts 30, clothing on hangers 30 |
| Resolution | 1080×1350, 1200×1200, 1280×960, 960×1280 (typical seller uploads), JPEG q72–95 |
| Ground truth | rendered at 2× and area-downsampled: `alpha/` physical opacity, `mask/` product silhouette |
| Rebuild | `python -m st_eval.synth --n 200 --seed 1403` (≈5 min on 4 CPU workers) |

**Failure modes it deliberately stresses** (each image is tagged; `eval/results.md` §3 reports per tag):

| Category | Hard cases rendered |
|---|---|
| Carpet | fringe threads 1–2 px wide with gaps, tassel bundles, beige fringe on beige backgrounds, dense field patterns |
| Gold jewelry | link chains with see-through links, pendants with holes, glints, mirror reflections on marble, gold-on-gold satin |
| Saffron & packaged | loose saffron threads scattered around the pack (foreground), zigzag pouch seals, clear windows, glass jars |
| Glassware | tea glasses (استکان) with saucers, bottles, fresnel-style partial alpha, background refraction, clear glass on white |
| Handicrafts | minakari plates with scalloped rims, khatam micro-mosaic, vases with handle holes, engraved copper on wood, patterned tablecloths |
| Clothing | wire hooks crossing a rail, sleeve–body gaps, lace hems with holes, sheer scarves with fringe, fuzzy knit edges |

**Ground-truth conventions** (a real annotation guide must adopt these explicitly):
1. The **silhouette** includes transparent parts. A glass counts as foreground even where it's 10% opaque, because a seller expects the whole glass in the cutout. IoU/BF use `mask/`; alpha metrics use `alpha/`.
2. **Cast shadows and surface reflections are background.**
3. **Hangers are foreground** (sellers usually keep them); rails, stands and hands would be background.
4. **Loose saffron threads styled around a package are foreground.**

## 2. Why synthetic numbers are not enough

What `synth-v1` does not capture, roughly in order of how much it could change a model ranking:

1. **Photographic appearance.** Wool pile and real fringe fibres, gold's environment reflections, glass caustics, fabric sheen. Our shading is procedural. Models trained on real photos may do *better* on real images than on our renders (domain gap working against them), or worse on specific materials.
2. **Capture pipeline.** Low-end Android sensors (common among Iranian sellers), HDR/beautify processing, motion blur, and especially **Telegram and Instagram recompression**: a photo sent as a Telegram "photo" (not a file) gets downscaled to ~1280 px and re-encoded, and that destroys fringe detail.
3. **Real seller scenes.** Carpets photographed on other carpets, bazaar stalls, products held in a hand, mannequins and people wearing clothing or jewelry, several products per photo, watermarks, phone numbers and Instagram handles overlaid on the image.
4. **Ambiguity.** What counts as product (price tags, packaging, stands, a rug's backing turned up at a corner) is sometimes unclear, and only real data plus an annotation guide resolves it.
5. **Distribution.** The category mix above is my guess at what's hard, not measured demand. Real request logs will say which categories matter.

**How much to trust `synth-v1` today:** use it for regression testing and for excluding clearly weaker
models, and treat per-category gaps smaller than ~0.03 IoU as noise. Validate the final model choice on real data (§3).

## 3. Real data still needed

### 3.1 Held-out benchmark (blocking a production model decision)

| Item | Requirement |
|---|---|
| **Size** | ≥ 200 images minimum (the brief); **600 recommended** (100 per category) for per-category confidence intervals of about ±0.02 IoU |
| **Categories** | the six above, plus a "general/other" bucket sampled from real bot traffic once it exists |
| **Rights** | seller opt-in with a written licence covering benchmarking and model training (bot checkbox, H-005 option B) **or** commissioned shoots with a photographer release. No scraped marketplace photos. |
| **Capture strata** | ≥ 3 phone tiers (low-end Android, mid Android, iPhone); lighting (daylight, shop fluorescent, flash); background (plain sheet, floor/carpet, cluttered home, bazaar stall) |
| **Transport strata** | each image kept in **three forms**: original file, Telegram "photo" recompression, Instagram-style 1080 px recompression |
| **Labels** | silhouette mask **and** alpha matte for fringe, chains, glass and sheer fabric (trimap + matting tool + manual touch-up); binary mask is enough for opaque boxy products |
| **Label QA** | 10% double-annotated; accept at silhouette IoU ≥ 0.97 and BF@3px ≥ 0.90 between annotators; disagreements go to the annotation guide |
| **Metadata** | category, sub-type, phone model, lighting, background type, recompression form, the same hard-case tags as `synth-v1` |
| **Split hygiene** | benchmark sellers must not appear in any fine-tuning data (split by seller, not by image) |

### 3.2 Fine-tuning data (needed for H-007 option B and for better Iranian-category quality)

- 5,000–10,000 real product photos with masks, same rights model, split by seller.
- Masks from a teacher model (e.g. the chosen BiRefNet variant) with human correction only on
  low-confidence images. Confidence = mean |p − 0.5| inside the boundary band.

### 3.3 Effort estimate (assumptions, not quotes)

- Alpha-matte annotation of hard images: **assumed** 20–40 min per image; binary masks: 3–8 min.
  Validate with a 20-image pilot before budgeting.
- 600-image benchmark with about 50% hard images → roughly 150–250 annotation hours under those assumptions.

## 4. What to do when real data arrives

1. Put it in the `synth-v1` layout (`images/ alpha/ mask/ meta/ manifest.jsonl dataset.json`), and
   `python -m st_eval.bench --dataset <real-set>` works unchanged.
2. Run every model on both sets and compute the **Spearman rank correlation** of Q between synthetic and real.
   If ρ ≥ 0.8, keep `synth-v1` as a cheap CI regression test; otherwise retire it or fix the generator
   where the rankings disagree.
3. Re-run `docs/economics.md` inputs if the chosen model changes.
