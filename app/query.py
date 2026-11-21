"""Query understanding: rewrite the user's question and extract listing filters."""
import json
import re
import sys

from app.llm import chat

PROMPT = """You convert a user's message into a search request for a real-estate chatbot.
Available data: DarGlobal luxury projects (UAE, Saudi Arabia, Qatar, Oman, Spain, UK) and
Wasalt property listings in Saudi cities (Riyadh, Jeddah, Dammam, Makkah, Madinah, Al Khobar).

Use the conversation so far to resolve follow-ups (e.g. "what about cheaper ones?").
Return ONLY a JSON object with exactly these keys:
- "search_query": the request rewritten as a clear, standalone English search query
- "wants_listings": true if the user wants specific Wasalt properties to buy or rent, otherwise false
- "city": one of "Riyadh", "Jeddah", "Dammam", "Makkah", "Madinah", "Al Khobar", or null
- "purpose": "sale", "rent", or null
- "property_type": one of "Villa", "Apartment", "Land", "Building", "Floor", or null
- "min_price": number in SAR, or null
- "max_price": number in SAR, or null
- "min_bedrooms": number, or null
- "sort": how to order listings:
    "price_low"  if the user asks for the cheapest, lowest price, most affordable or least expensive
    "price_high" if the user asks for the most expensive, highest price or most luxurious
    "newest"     otherwise

Examples:
"cheapest villa in Makkah" -> "sort": "price_low"
"most expensive apartment for rent in Riyadh" -> "sort": "price_high"
"villas for sale in Jeddah" -> "sort": "newest\""""

DEFAULTS = {"search_query": "", "wants_listings": False, "city": None, "purpose": None,
            "property_type": None, "min_price": None, "max_price": None, "min_bedrooms": None,
            "sort": "newest"}
NUMBER_KEYS = ("min_price", "max_price", "min_bedrooms")


def to_number(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def analyze(question, history=None):
    """Ask the model to rewrite the question and extract filters. Never raises."""
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in (history or [])[-4:])
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Conversation so far:\n{convo or '(none)'}\n\nNew message: {question}"},
    ]
    data = {}
    try:
        raw = chat(messages, temperature=0, max_tokens=1500)
        match = re.search(r"\{.*\}", raw, re.S)          # take the JSON even if wrapped in text
        data = json.loads(match.group(0)) if match else {}
    except Exception as e:                               # model busy, bad JSON, etc.
        print("query analysis failed, using the raw question:", e)

    result = {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
    for key in NUMBER_KEYS:
        result[key] = to_number(result[key])
    result["wants_listings"] = bool(result["wants_listings"])
    result["search_query"] = result["search_query"] or question
    if result["sort"] not in ("newest", "price_low", "price_high"):
        result["sort"] = "newest"
    return result


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "villas for sale in Riyadh under 3 million"
    print(json.dumps(analyze(question), indent=2, ensure_ascii=False))