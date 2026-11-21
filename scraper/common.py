import time

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StudentAssignmentBot/1.0)",
    "Accept-Language": "en",
}
session = requests.Session()
session.headers.update(HEADERS)


def get_soup(url, delay=1.5):
    """Download a page politely (pause first) and return parsed HTML, or None."""
    time.sleep(delay)
    try:
        r = session.get(url, timeout=30)
    except requests.RequestException as e:
        print("  error:", e, url)
        return None
    if r.status_code != 200:
        print("  skip", r.status_code, url)
        return None
    return BeautifulSoup(r.text, "lxml")