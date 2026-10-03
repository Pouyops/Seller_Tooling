"""Input and output shapes for listing generation, plus marketplace limits."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# Conservative caps. Real marketplace limits must be confirmed from each seller panel (H-012).
TITLE_MAX = 70
DESCRIPTION_MAX = 900
KEYWORDS_MIN, KEYWORDS_MAX = 5, 12
ATTRIBUTES_MAX = 12


@dataclass
class SellerFields:
    """The few things we ask a seller to type. Everything else comes from the photo."""

    category: str = ""
    brand: str = ""
    model: str = ""
    material: str = ""
    color: str = ""
    size: str = ""
    weight: str = ""
    origin: str = ""
    condition: str = ""
    price_toman: int | None = None
    notes: str = ""
    marketplace: str = "digikala"  # digikala | basalam | torob | instagram

    def filled(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in ("category", "brand", "model", "material", "color", "size", "weight", "origin", "condition", "notes"):
            value = (getattr(self, key) or "").strip()
            if value:
                out[key] = value
        if self.price_toman:
            out["price_toman"] = str(self.price_toman)
        return out


FIELD_LABELS_FA = {
    "category": "دسته‌بندی",
    "brand": "برند",
    "model": "مدل",
    "material": "جنس",
    "color": "رنگ",
    "size": "اندازه",
    "weight": "وزن",
    "origin": "ساخت",
    "condition": "وضعیت",
    "notes": "توضیحات فروشنده",
    "price_toman": "قیمت (تومان)",
}


class Listing(BaseModel):
    """What the LLM must return (also used as the JSON schema sent to the server)."""

    title: str = Field(description="عنوان کوتاه و دقیق محصول به فارسی")
    description: str = Field(description="توضیح دو تا چهار جمله‌ای به فارسی")
    attributes: dict[str, str] = Field(default_factory=dict, description="ویژگی‌های محصول")
    keywords: list[str] = Field(default_factory=list, description="کلمات کلیدی جستجو")
    category_guess: str = Field(default="", description="حدس دسته‌بندی")


def json_schema() -> dict:
    """A deliberately small schema: llama.cpp converts it to a grammar, and big schemas slow sampling."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "category_guess": {"type": "string"},
        },
        "required": ["title", "description", "attributes", "keywords"],
    }
