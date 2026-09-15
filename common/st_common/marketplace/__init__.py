"""Marketplace data sources.

Torob, Digikala and Basalam adapters are **stubs by design**: automated collection from those
sites is blocked on a Terms-of-Service review (HUMAN_NEEDED H-003). Only ``SellerExportSource``
(data a seller exports and explicitly shares with us) is implemented.
"""

from .base import Category, MarketplaceSource, ProductRecord, SourceBlocked
from .adapters import BasalamSource, DigikalaSource, SellerExportSource, TorobSource, get_source

__all__ = [
    "BasalamSource",
    "Category",
    "DigikalaSource",
    "MarketplaceSource",
    "ProductRecord",
    "SellerExportSource",
    "SourceBlocked",
    "TorobSource",
    "get_source",
]
