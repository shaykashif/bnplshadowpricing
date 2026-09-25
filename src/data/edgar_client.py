"""Minimal SEC EDGAR client with polite rate limiting and retry on 503s."""
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "ShadowPricing research bot admin@shadowpricing.local"}
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch(url: str, retries: int = 6, pause: float = 4.0, **params) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params or None, timeout=300)
        except requests.exceptions.RequestException:
            time.sleep(pause * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(0.2)
            return r
        time.sleep(pause * (attempt + 1))
    r.raise_for_status()
    return r


def full_text_search(query: str, forms: str = "", start: str = "2015-01-01", end: str = "2026-12-31") -> list:
    params = {"q": query, "dateRange": "custom", "startdt": start, "enddt": end}
    if forms:
        params["forms"] = forms
    return fetch("https://efts.sec.gov/LATEST/search-index", **params).json()["hits"]["hits"]


def download(url: str, name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if not path.exists():
        path.write_bytes(fetch(url).content)
    return path
