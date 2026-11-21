"""Metadata filtering over Wasalt listings (self-querying retrieval, step 2).

The model extracts filters from the question (app/query.py); this module applies
them to the listings' structured fields: city, purpose, property type, price, bedrooms.
"""
import re

from app.chunker import load_documents


def norm(text):
    """'Al-Khobar' / 'Al Khobar' / 'khobar' -> 'khobar' so names compare reliably."""
    return re.sub(r"[^a-z]", "", (text or "").lower()).removeprefix("al")


class ListingFilter:
    def __init__(self, docs=None):
        docs = docs if docs is not None else load_documents()
        self.listings = [d for d in docs if d.get("type") == "listing"]

    def matches(self, d, f):
        """True if listing d satisfies every filter that is set in f."""
        if f.get("city") and norm(f["city"]) not in norm(d.get("city")) + norm(d.get("location")):
            return False
        if f.get("purpose") and f["purpose"] != d.get("purpose"):
            return False
        if f.get("property_type") and f["property_type"].lower() != (d.get("property_type") or "").lower():
            return False
        price = d.get("price_sar")
        if f.get("min_price") and (price is None or price < f["min_price"]):
            return False
        if f.get("max_price") and (price is None or price > f["max_price"]):
            return False
        beds = d.get("bedrooms")
        if f.get("min_bedrooms") and (beds is None or beds < f["min_bedrooms"]):
            return False
        return True

    def search(self, filters, limit=8):
        """Return (up to `limit` unique matching listings in the requested order; total unique matches).

        Order: 'newest' (default), 'price_low' (cheapest first) or 'price_high' (most expensive first).
        The same property is often posted several times, so identical listings are counted once.
        """
        matching = [d for d in self.listings if self.matches(d, filters)]
        order = filters.get("sort") or "newest"
        if order == "price_low":
            matching.sort(key=lambda d: (d.get("price_sar") is None, d.get("price_sar") or 0))
        elif order == "price_high":
            matching.sort(key=lambda d: (d.get("price_sar") is None, -(d.get("price_sar") or 0)))
        else:
            matching.sort(key=lambda d: d.get("listed_on") or "", reverse=True)

        unique, seen = [], set()
        for d in matching:
            key = (d.get("purpose"), norm(d.get("location")), d.get("price_sar"),
                   d.get("area_sqm"), d.get("bedrooms"))
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique[:limit], len(unique)


def format_listing(d):
    """One compact line per listing for the prompt."""
    parts = [d["title"], d.get("location", ""), f"for {d.get('purpose')}"]
    if d.get("price_sar"):
        note = " (rent period not stated)" if d.get("purpose") == "rent" else ""
        parts.append(f"SAR {d['price_sar']:,.0f}{note}")
    if d.get("bedrooms"):
        parts.append(f"{d['bedrooms']} bedrooms")
    if d.get("area_sqm"):
        parts.append(f"{d['area_sqm']:,.0f} sqm")
    if d.get("listed_on"):
        parts.append(f"listed {d['listed_on']}")
    return " | ".join(parts)