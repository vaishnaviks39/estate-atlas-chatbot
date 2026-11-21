"""Parse DarGlobal project data from pages saved with a normal browser.

DarGlobal uses Imperva bot protection, which blocks scripts. Rather than
bypass it, pages are saved manually (Cmd+S, "Webpage, HTML Only") into
raw/darglobal/, and this script reads the JSON the site embeds in each
page (__NEXT_DATA__).

    python -m scraper.darglobal --list   # which project pages still need saving
    python -m scraper.darglobal --open   # open the missing pages in Chrome
    python -m scraper.darglobal          # parse saved pages -> data/darglobal.json
"""
import json
import subprocess
import sys
from pathlib import Path
import re

from bs4 import BeautifulSoup

BASE = "https://darglobal.co.uk"
RAW_DIR = Path("raw/darglobal")
LISTING_PAGE = RAW_DIR / "projects.html"
OUT_FILE = Path("data/darglobal.json")


# ---------- helpers ----------

def load_next_data(path):
    """Return the __NEXT_DATA__ JSON embedded in a saved page, or None."""
    html = path.read_text(encoding="utf-8")
    tag = BeautifulSoup(html, "lxml").find("script", id="__NEXT_DATA__")
    return json.loads(tag.string) if tag and tag.string else None


def walk(node):
    """Yield every dict inside a nested JSON structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def rich_text(node):
    """Join all 'text' pieces of a rich-text block (used by FAQ answers)."""
    return " ".join(d["text"].strip() for d in walk(node)
                    if isinstance(d.get("text"), str) and d["text"].strip())

def clean(text):
    """Collapse newlines and repeated spaces into single spaces."""
    return " ".join(str(text).split()) if text else text

def file_name_for(link):
    """'/rayana/trump-mansions' -> 'rayana__trump-mansions.html'"""
    return link.strip("/").replace("/", "__") + ".html"


# ---------- /projects page ----------

def get_projects(listing_data):
    """All project cards on /projects, keyed by link (38 unique)."""
    projects = {}
    for d in walk(listing_data):
        if "ProjectTitle" in d and d.get("Link"):
            projects.setdefault(d["Link"], d)
    return projects


def get_guides(listing_data):
    """Long buying-guide texts (per country / property type) on /projects."""
    guides, seen = [], set()
    for d in walk(listing_data):
        html = d.get("seoContent")
        if not (isinstance(html, str) and len(html) > 500):
            continue
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)
        if text in seen:
            continue
        seen.add(text)
        heading = soup.find(["h1", "h2"])
        guides.append({
            "source": "DarGlobal",
            "type": "guide",
            "title": heading.get_text(strip=True) if heading else "DarGlobal buying guide",
            "description": text,
            "url": BASE + "/projects",
        })
    return guides


def base_record(card):
    """Fields available for every project from the /projects page."""
    return {
        "source": "DarGlobal",
        "type": "project",
        "title": card.get("ProjectTitle"),
        "location": card.get("Location"),
        "property_type": card.get("ProjectType"),
        "bedrooms": card.get("Bedrooms") if any(ch.isdigit() for ch in (card.get("Bedrooms") or "")) else None,
        "completion": card.get("completionDate"),
        "description": card.get("ProjectDescription"),
        "price": "Not published (contact DarGlobal sales)",
        "url": BASE + card["Link"],
    }


# ---------- individual project pages ----------

SKIP_KEYS = {"hash", "mime", "ext", "provider", "url", "href", "alt", "alternativeText",
             "createdAt", "updatedAt", "publishedAt", "vuid", "versionComment", "slug",
             "locale", "ButtonTheme", "label", "lat", "long"}


def page_text(node):
    """All readable paragraphs inside a page's own data block, de-duplicated."""
    seen, parts = set(), []
    for d in walk(node):
        for key, value in d.items():
            if (key in SKIP_KEYS or not isinstance(value, str) or len(value) < 25
                    or " " not in value or value.startswith(("http", "/"))):
                continue
            text = BeautifulSoup(value, "lxml").get_text(" ") if "<" in value else value
            text = clean(text)
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
    return "\n".join(parts)

def parse_detail_page(path, title, url):
    """Return (extra fields for the project, list of FAQ records)."""
    data = load_next_data(path)
    if data is None:
        return {}, []
    props = data["props"]["pageProps"]
    # Each page layout keeps its own content in a different block
    section = (props.get("projectDetailsData") or props.get("oneOfOneData")
               or props.get("pageData") or {})

        # One-of-one pages contain ALL units; keep only the unit this page is about
    if props.get("oneOfOneData"):
        units = (((props["oneOfOneData"].get("attributes") or {})
                  .get("one_of_one_collections") or {}).get("data") or [])
        match = next((u for u in units
                      if (u.get("attributes") or {}).get("slug") == props.get("slug")), None)
        if match:
            unit = {k: v for k, v in (match.get("attributes") or {}).items()
                    if k != "localizations"}          # skip other-language copies
            section = {"attributes": unit}
        else:
            section = {}

    # Structured fields (standard project pages)
    attrs = section.get("attributes") or {} if isinstance(section, dict) else {}
    pd = attrs.get("ProjectDetails") or {}
    about = pd.get("AboutProject") or {}
    features = pd.get("KeyFeatures") or {}
    amenities = pd.get("Amenities") or {}
    location = pd.get("loaction") or pd.get("location") or {}

    extra = {
        "about": about.get("description"),
        "overview": (pd.get("PhotoGallery") or {}).get("description"),
        "summary": (pd.get("SEO") or {}).get("metaDescription"),
        "features_intro": features.get("description"),
        "key_features": "; ".join(
            f"{f.get('Title')}: {f.get('description')}"
            for f in features.get("keyfeaturelist") or [] if f.get("Title")),
        "amenities_intro": amenities.get("description"),
        "amenities": "; ".join(
            a["name"] for a in amenities.get("amenitiesList") or [] if a.get("name")),
        "location_details": location.get("description"),
    }
    for fact in about.get("AboutDetailsType") or []:
        if fact.get("name") and fact.get("value"):
            key = re.sub(r"[^a-z0-9]+", "_", fact["name"].lower()).strip("_")
            extra[key] = fact["value"]

    # Fallback for other layouts: keep all readable text from the page's data block
    if not extra["about"]:
        extra["page_text"] = page_text(section)

    # FAQs: any question/answer pair anywhere in the page's data block
    faqs, seen_q = [], set()
    for d in walk(section):
        q, a = d.get("question"), d.get("answer")
        if not q or not a or q in seen_q:
            continue
        answer = clean(a if isinstance(a, str) else rich_text(a))
        if "<" in answer:
            answer = clean(BeautifulSoup(answer, "lxml").get_text(" "))
        if answer:
            seen_q.add(q)
            faqs.append({"source": "DarGlobal", "type": "faq", "project": title,
                         "title": clean(q), "answer": answer, "url": url})

    return {k: clean(v) for k, v in extra.items() if v}, faqs


# ---------- commands ----------

def list_pages(projects):
    saved = 0
    for i, link in enumerate(projects, 1):
        name = file_name_for(link)
        if (RAW_DIR / name).exists():
            saved += 1
            print(f"{i:2}. saved    {BASE}{link}")
        else:
            print(f"{i:2}. MISSING  {BASE}{link}")
            print(f"              save as: {name}")
    print(f"\n{saved}/{len(projects)} project pages saved")


def open_missing(projects, batch_size=5):
    """Open the next few not-yet-saved project pages in Chrome (save each with Cmd+S)."""
    missing = [BASE + link for link in projects if not (RAW_DIR / file_name_for(link)).exists()]
    batch = missing[:batch_size]
    if batch:
        subprocess.run(["open", "-a", "Google Chrome", *batch])
    print(f"Opened {len(batch)} pages ({len(missing) - len(batch)} more after these)")


def main():
    listing = load_next_data(LISTING_PAGE)
    if listing is None:
        sys.exit(f"Could not read __NEXT_DATA__ from {LISTING_PAGE}")
    projects = get_projects(listing)

    if "--list" in sys.argv:
        list_pages(projects)
        return
    if "--open" in sys.argv:
        open_missing(projects)
        return

    records, faqs, detailed = [], [], 0
    for link, card in projects.items():
        record = base_record(card)
        page = RAW_DIR / file_name_for(link)
        if page.exists():
            extra, page_faqs = parse_detail_page(page, record["title"], record["url"])
            record.update(extra)
            faqs.extend(page_faqs)
            detailed += bool(extra)
        records.append(record)

    guides = get_guides(listing)
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(records + faqs + guides, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"{len(records)} projects ({detailed} with full details), "
          f"{len(faqs)} FAQs, {len(guides)} guides -> {OUT_FILE}")


if __name__ == "__main__":
    main()