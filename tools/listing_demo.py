#!/usr/bin/env python
"""Generate a Persian listing from a real photo with the local LLM.

    python -m st_inference.listing.llm_server          # terminal 1: start llama-server
    python tools/listing_demo.py --image <photo.jpg>   # terminal 2

Prints the listing and says whether it came from the model or the deterministic fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):  # Persian output on a cp1252 console
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from st_inference.listing import ListingGenerator, SellerFields  # noqa: E402
from st_inference.listing.llm_client import LocalLLM  # noqa: E402
from st_inference.settings import get_settings  # noqa: E402

PRESETS = {
    "carpet": SellerFields(category="فرش دستباف", material="پشم", size="۱۲ متری", color="لاکی", origin="کاشان",
                           price_toman=85_000_000, notes="بدون لک و پارگی", marketplace="digikala"),
    "saffron": SellerFields(category="زعفران", brand="سحرخیز", weight="۴.۶ گرم", origin="قائنات",
                            price_toman=1_250_000, condition="نو", marketplace="basalam"),
    "jewelry": SellerFields(category="گردنبند طلا", material="طلای ۱۸ عیار", weight="۳.۲ گرم", color="طلایی",
                            price_toman=42_000_000, marketplace="instagram"),
    "glass": SellerFields(category="استکان کمرباریک", material="شیشه", size="۶ عددی", origin="ایران",
                          price_toman=850_000, marketplace="digikala"),
    "none": SellerFields(),
}


async def run(args) -> int:
    settings = get_settings()
    llm = LocalLLM(args.url or settings.llm_base_url, args.model or settings.llm_model, timeout_s=args.timeout)
    if not await llm.healthy():
        print(f"[demo] no LLM at {llm.http.base_url} — start it with: python -m st_inference.listing.llm_server")
        print("[demo] continuing anyway to show the fallback path\n")
    image = Path(args.image).read_bytes() if args.image else None
    fields = PRESETS[args.preset]
    print(f"[demo] image={args.image or 'none'}  preset={args.preset}  seller fields={fields.filled()}\n")

    result = await ListingGenerator(llm).generate(image, fields)
    await llm.aclose()

    listing = result.listing
    print(f"source={result.source}  attempts={result.attempts}  {result.latency_ms / 1000:.1f}s  issues={result.issues}\n")
    print("عنوان      :", listing.title)
    print("توضیحات   :", listing.description)
    print("ویژگی‌ها   :", json.dumps(listing.attributes, ensure_ascii=False))
    print("کلیدواژه‌ها:", "، ".join(listing.keywords))
    print("دسته‌بندی  :", listing.category_guess)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"listing": listing.model_dump(), "source": result.source, "issues": result.issues,
             "latency_ms": result.latency_ms, "model": result.model}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[demo] saved:", args.out)
    return 0 if result.ok else 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None)
    ap.add_argument("--preset", default="carpet", choices=sorted(PRESETS))
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--out", default=None)
    raise SystemExit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
