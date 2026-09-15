from __future__ import annotations

import csv
import json
from collections.abc import AsyncIterator
from pathlib import Path

from ..persian import normalize, parse_amount_toman
from .base import Category, MarketplaceSource, ProductRecord, SourceBlocked

_BLOCKED_MSG = (
    "{name}: automated access is blocked pending a Terms-of-Service review (HUMAN_NEEDED.md H-003). "
    "No request was made."
)


class _BlockedSource(MarketplaceSource):
    access_status = "blocked_tos_review"

    async def iter_products(self, *, category_id: str | None = None, limit: int = 100) -> AsyncIterator[ProductRecord]:
        raise SourceBlocked(_BLOCKED_MSG.format(name=self.name))
        yield  # pragma: no cover  (makes this an async generator)

    async def get_product(self, product_id: str) -> ProductRecord:
        raise SourceBlocked(_BLOCKED_MSG.format(name=self.name))

    async def get_categories(self) -> list[Category]:
        raise SourceBlocked(_BLOCKED_MSG.format(name=self.name))


class TorobSource(_BlockedSource):
    name = "torob"


class DigikalaSource(_BlockedSource):
    name = "digikala"


class BasalamSource(_BlockedSource):
    name = "basalam"


class SellerExportSource(MarketplaceSource):
    """Products a seller exported themselves and consented to share (CSV).

    Expected columns: ``product_id,title,category,price,attributes_json,images``, with
    ``images`` separated by ``|``. Text is normalized on the way in.
    """

    name = "seller_export"
    access_status = "cleared"

    def __init__(self, csv_path: Path | str, *, consent: str):
        self.csv_path = Path(csv_path)
        self.consent = consent

    def _rows(self):
        with self.csv_path.open(encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)

    def _to_record(self, row: dict[str, str]) -> ProductRecord:
        attrs = json.loads(row.get("attributes_json") or "{}")
        return ProductRecord(
            source=self.name,
            product_id=row["product_id"],
            title_fa=normalize(row.get("title", "")),
            category_path=[normalize(c) for c in (row.get("category") or "").split(">") if c.strip()],
            attributes={normalize(k): normalize(str(v)) for k, v in attrs.items()},
            price_toman=parse_amount_toman(row.get("price") or ""),
            image_refs=[p for p in (row.get("images") or "").split("|") if p],
            consent=self.consent,
        )

    async def iter_products(self, *, category_id: str | None = None, limit: int = 100) -> AsyncIterator[ProductRecord]:
        n = 0
        for row in self._rows():
            rec = self._to_record(row)
            if category_id and category_id not in rec.category_path:
                continue
            yield rec
            n += 1
            if n >= limit:
                return

    async def get_product(self, product_id: str) -> ProductRecord:
        for row in self._rows():
            if row["product_id"] == product_id:
                return self._to_record(row)
        raise KeyError(product_id)

    async def get_categories(self) -> list[Category]:
        seen: dict[str, Category] = {}
        for row in self._rows():
            parent = None
            for part in [normalize(c) for c in (row.get("category") or "").split(">") if c.strip()]:
                seen.setdefault(part, Category(self.name, part, part, parent))
                parent = part
        return list(seen.values())


_SOURCES = {"torob": TorobSource, "digikala": DigikalaSource, "basalam": BasalamSource}


def get_source(name: str) -> MarketplaceSource:
    try:
        return _SOURCES[name]()
    except KeyError:
        raise ValueError(f"unknown marketplace source {name!r}") from None
