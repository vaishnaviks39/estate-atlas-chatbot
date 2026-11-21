"""Scrape public property listings from wasalt.sa.

Each search page embeds schema.org JSON-LD:
  - one "Offer" block per listing (price, address, rooms, area)
  - one "FAQPage" block with Q&As about the area
We read these structured blocks instead of parsing visible HTML.
Seller/agent details are never stored.
"""
import json
from pathlib import Path

from scraper.common import get_soup

BASE = "https://wasalt.sa/en"
CITIES = ["riyadh", "jeddah", "dammam", "makkah", "madinah", "al-khobar"]
PURPOSES = ["sale", "rent"]
PROPERTY_KINDS = ["properties", "villas", "apartments"]
OUT_FILE = Path("data/wasalt.json")


def to_float(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def to_int(value):
    number = to_float(value)
    return int(number) if number is not None else None

TYPE_FIXES = {"شقة": "Apartment", "فيلا": "Villa", "أرض": "Land", "Rest": "Rest House"}
SKIP_WORDS = {"Furnished", "Semi-Furnished", "Unfurnished", "New", "Luxury"}


def clean_property_type(title):
    """'Furnished Apartment 120 SQM' -> 'Apartment'; 'شقة ...' -> 'Apartment'."""
    for word in title.split():
        if word in SKIP_WORDS:
            continue
        return TYPE_FIXES.get(word, word)
    return ""


def clean_price(price, purpose):
    """Drop obviously wrong prices (typos on the source site)."""
    if price is None or price <= 0:
        return None
    limit = 5_000_000 if purpose == "rent" else 2_000_000_000
    return price if price <= limit else None

def parse_offer(offer, purpose):
    """Turn one JSON-LD Offer into a clean, flat record."""
    item = offer.get("itemOffered") or {}
    address = item.get("address") or {}
    size = item.get("floorSize") or {}
    name = item.get("name", "")
    street = address.get("streetAddress", "")
    return {
        "source": "Wasalt",
        "type": "listing",
        "purpose": purpose,                                 # sale / rent
        "property_type": clean_property_type(name),   # Villa, Apartment, Floor, Land...
        "title": name,
        "district": street.split(",")[0].strip(),
        "region": address.get("addressRegion", ""),
        "city": address.get("addressLocality", ""),
        "location": street,
        "price_sar": clean_price(to_float(offer.get("price")), purpose),
        "price_note": "rent period not stated on listing" if purpose == "rent" else "",
        "bedrooms": to_int(item.get("numberOfRooms")),
        "bathrooms": to_int(item.get("numberOfBathroomsTotal")),
        "area_sqm": to_float(size.get("value")),
        "listed_on": offer.get("availabilityStarts", ""),
        "url": offer.get("url", ""),
    }


def parse_faq(faq_block, city, purpose, page_url):
    """Turn the FAQPage block into one record per question."""
    records = []
    for question in faq_block.get("mainEntity", []):
        records.append({
            "source": "Wasalt",
            "type": "faq",
            "city": city.title(),
            "purpose": purpose,
            "title": question.get("name", ""),
            "answer": (question.get("acceptedAnswer") or {}).get("text", ""),
            "url": page_url,
        })
    return records


def scrape_search_page(url, city, purpose, include_faq):
    soup = get_soup(url)
    if soup is None:
        return [], []
    listings, faqs = [], []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            block = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        if not isinstance(block, dict):
            continue
        if block.get("@type") == "Offer":
            listings.append(parse_offer(block, purpose))
        elif block.get("@type") == "FAQPage" and include_faq:
            faqs.extend(parse_faq(block, city, purpose, url))
    return listings, faqs


def main():
    all_records, seen_urls, seen_faqs = [], set(), set()
    for city in CITIES:
        for purpose in PURPOSES:
            for kind in PROPERTY_KINDS:
                url = f"{BASE}/{kind}-for-{purpose}-in-{city}"
                listings, faqs = scrape_search_page(url, city, purpose, include_faq=True)
                new = [d for d in listings if d["url"] and d["url"] not in seen_urls]
                seen_urls.update(d["url"] for d in new)
                new_faqs = [f for f in faqs if (f["city"], f["title"]) not in seen_faqs]
                seen_faqs.update((f["city"], f["title"]) for f in new_faqs)
                all_records.extend(new + new_faqs)
                print(f"{city:10} {purpose:4} {kind:10}: {len(new):2} new listings, {len(new_faqs)} FAQs")
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(all_records)} records to {OUT_FILE}")


if __name__ == "__main__":
    main()