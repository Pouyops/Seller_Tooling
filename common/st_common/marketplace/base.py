from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


class SourceBlocked(NotImplementedError):
    """The adapter exists but must not fetch anything until a human clears it."""


@dataclass
class Category:
    source: str
    category_id: str
    title_fa: str
    parent_id: str | None = None


@dataclass
class ProductRecord:
    source: str
    product_id: str
    title_fa: str
    category_path: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    price_toman: int | None = None
    image_refs: list[str] = field(default_factory=list)  # URLs or local paths
    url: str | None = None
    consent: str | None = None  # how we obtained the right to use this record


class MarketplaceSource(ABC):
    """Read-only interface every marketplace adapter implements."""

    name: str = "abstract"
    #: "cleared" | "blocked_tos_review" | "not_applicable"
    access_status: str = "blocked_tos_review"

    @abstractmethod
    async def iter_products(self, *, category_id: str | None = None, limit: int = 100) -> AsyncIterator[ProductRecord]:
        ...

    @abstractmethod
    async def get_product(self, product_id: str) -> ProductRecord:
        ...

    @abstractmethod
    async def get_categories(self) -> list[Category]:
        ...
