"""Turn a photo + seller fields into a validated Persian listing.

Pipeline: prompt -> local LLM -> parse JSON -> normalize every string -> validate as Persian ->
one repair round if needed -> deterministic template if the model still can't comply.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from st_common.logs import get_logger
from st_common.persian import normalize, search_key, to_persian_digits, validate_persian

from . import prompt as P
from .llm_client import LLMUnavailable, LocalLLM
from .schema import (
    ATTRIBUTES_MAX,
    DESCRIPTION_MAX,
    FIELD_LABELS_FA,
    KEYWORDS_MAX,
    KEYWORDS_MIN,
    TITLE_MAX,
    Listing,
    SellerFields,
    json_schema,
)

log = get_logger("st_inference.listing")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_OBJECT = re.compile(r"\{.*\}", re.S)
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff⬀-⯿️‍]+"
)
# Issues that make the output unusable: fall back to the template rather than show these.
FATAL = {"bad_json", "missing_fields", "empty", "mojibake", "low_persian_ratio", "foreign_script", "arabic_chars"}
# Worth one more attempt, but acceptable if the model can't do better than this.
REPAIRABLE = {"repetitive", "too_few_keywords", "title_too_long"}


@dataclass
class ListingResult:
    listing: Listing
    source: str  # model | model_repaired | template_fallback
    issues: list[str] = field(default_factory=list)
    attempts: int = 0
    latency_ms: float = 0.0
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.source != "template_fallback"


class ListingGenerator:
    def __init__(self, llm: LocalLLM, *, max_repairs: int = 1):
        self.llm = llm
        self.max_repairs = max_repairs

    # ---- public --------------------------------------------------------------------------------
    async def generate(self, image: bytes | None, fields: SellerFields) -> ListingResult:
        t0 = time.perf_counter()
        messages = P.build_messages(fields, image)
        raw, attempts, issues = "", 0, ["bad_json"]
        listing: Listing | None = None

        for attempt in range(self.max_repairs + 1):
            attempts = attempt + 1
            try:
                raw = await self.llm.chat(messages, schema=json_schema())
            except LLMUnavailable as e:
                log.warning("llm.unavailable", extra={"error": str(e)[:200]})
                return self._fallback(fields, ["llm_unavailable"], attempts, t0)
            data, parse_issues = self._parse(raw)
            if data is None:
                issues = parse_issues
            else:
                listing, issues = self._clean(data, fields)
            last_attempt = attempt == self.max_repairs
            blocking = set(issues) & (FATAL if last_attempt else FATAL | REPAIRABLE)
            if listing is not None and not blocking:
                source = "model" if attempt == 0 else "model_repaired"
                return ListingResult(listing, source, issues, attempts, (time.perf_counter() - t0) * 1000, self.llm.model)
            if attempt < self.max_repairs:
                messages = P.repair_messages(fields, image, raw, [P.ISSUE_FA.get(i, i) for i in issues])

        log.info("listing.fallback", extra={"issues": issues})
        return self._fallback(fields, issues, attempts, t0)

    # ---- parsing & cleaning ---------------------------------------------------------------------
    def _parse(self, raw: str) -> tuple[dict | None, list[str]]:
        text = (raw or "").strip()
        if not text:
            return None, ["empty"]
        fenced = _FENCE.search(text)
        if fenced:
            text = fenced.group(1).strip()
        else:
            obj = _OBJECT.search(text)
            if obj:
                text = obj.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None, ["bad_json"]
        if not isinstance(data, dict):
            return None, ["bad_json"]
        if not all(k in data for k in ("title", "description")):
            return None, ["missing_fields"]
        return data, []

    def _text(self, value: object) -> str:
        return normalize(_EMOJI.sub("", str(value or "")).strip())

    def _clean(self, data: dict, fields: SellerFields) -> tuple[Listing, list[str]]:
        issues: list[str] = []
        title = self._text(data.get("title"))
        description = self._text(data.get("description"))

        for name, value in (("title", title), ("description", description)):
            found = validate_persian(value, min_persian_ratio=0.5)
            issues.extend(i for i in found if i not in issues)
            if not value:
                issues.append("empty")
            del name

        if self._repetitive(description):
            issues.append("repetitive")
        if len(title) > TITLE_MAX:
            title = self._truncate(title, TITLE_MAX)
            issues.append("title_too_long")
        if len(description) > DESCRIPTION_MAX:
            description = self._truncate(description, DESCRIPTION_MAX)

        attributes: dict[str, str] = {}
        raw_attributes = data.get("attributes")
        if isinstance(raw_attributes, dict):
            for k, v in raw_attributes.items():
                key, value = self._text(k), self._text(v)
                if key and value and len(attributes) < ATTRIBUTES_MAX:
                    attributes[key] = value

        keywords: list[str] = []
        seen: set[str] = set()
        for item in data.get("keywords") or []:
            word = self._text(item)
            key = search_key(word)
            if word and key and key not in seen and len(keywords) < KEYWORDS_MAX:
                seen.add(key)
                keywords.append(word)
        if len(keywords) < KEYWORDS_MIN:
            for extra in self._fallback_keywords(fields, title):
                key = search_key(extra)
                if key not in seen and len(keywords) < KEYWORDS_MIN:
                    seen.add(key)
                    keywords.append(extra)
            issues.append("too_few_keywords")

        listing = Listing(title=title, description=description, attributes=attributes, keywords=keywords,
                          category_guess=self._text(data.get("category_guess")))
        return listing, issues

    @staticmethod
    def _repetitive(description: str, overlap: float = 0.75) -> bool:
        """Small models pad Persian descriptions by restating a sentence almost verbatim."""
        sentences = [s.strip() for s in re.split(r"[.!؟\n]+", description) if len(s.strip()) > 25]
        for i, a in enumerate(sentences):
            for b in sentences[i + 1:]:
                words_a, words_b = set(a.split()), set(b.split())
                if not words_a or not words_b:
                    continue
                if len(words_a & words_b) / min(len(words_a), len(words_b)) >= overlap:
                    return True
        return False

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        cut = text[:limit]
        space = cut.rfind(" ")
        return (cut[:space] if space > limit * 0.6 else cut).rstrip(" ،-")

    # ---- deterministic fallback -------------------------------------------------------------------
    @staticmethod
    def _fallback_keywords(fields: SellerFields, title: str = "") -> list[str]:
        out: list[str] = []
        for value in (fields.category, fields.brand, fields.material, fields.color, fields.size, fields.origin):
            value = normalize((value or "").strip())
            if value:
                out.append(value)
        if fields.category and fields.material:
            out.append(normalize(f"{fields.category} {fields.material}"))
        if fields.category and fields.color:
            out.append(normalize(f"{fields.category} {fields.color}"))
        for token in title.split():
            if len(token) > 3 and token not in out:
                out.append(token)
        return [k for k in out if k][:KEYWORDS_MAX]

    def _fallback(self, fields: SellerFields, issues: list[str], attempts: int, t0: float) -> ListingResult:
        parts = [fields.category, fields.brand, fields.model, fields.color, fields.size]
        title = normalize(" ".join(p.strip() for p in parts if p and p.strip())) or "محصول"
        title = self._truncate(title, TITLE_MAX)

        filled = fields.filled()
        described = "، ".join(f"{FIELD_LABELS_FA.get(k, k)}: {normalize(v)}"
                              for k, v in filled.items() if k not in ("notes", "price_toman"))
        sentences = [f"{title}."]
        if described:
            sentences.append(f"مشخصات: {described}.")
        if fields.notes:
            sentences.append(normalize(fields.notes.strip().rstrip(".")) + ".")
        description = to_persian_digits(normalize(" ".join(sentences)))

        attributes = {FIELD_LABELS_FA.get(k, k): to_persian_digits(normalize(v))
                      for k, v in filled.items() if k != "notes"}
        listing = Listing(title=to_persian_digits(title), description=description,
                          attributes=dict(list(attributes.items())[:ATTRIBUTES_MAX]),
                          keywords=self._fallback_keywords(fields, title),
                          category_guess=normalize(fields.category or ""))
        return ListingResult(listing, "template_fallback", issues, attempts, (time.perf_counter() - t0) * 1000,
                             self.llm.model)
