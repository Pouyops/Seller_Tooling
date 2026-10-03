"""Persian listing generation: product image + seller fields -> title, description, attributes, keywords.

Runs against a local OpenAI-compatible server (llama.cpp's ``llama-server``) — no hosted API.
Every string the seller sees passes through ``st_common.persian.normalize`` and is rejected if it
fails ``validate_persian``; if the model can't produce valid Persian twice, a deterministic
template built from the seller's own fields is returned instead, so a listing is never empty.
"""

from .generator import ListingGenerator, ListingResult
from .schema import Listing, SellerFields

__all__ = ["Listing", "ListingGenerator", "ListingResult", "SellerFields"]
