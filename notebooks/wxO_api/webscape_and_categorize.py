#!/usr/bin/env python3
# webscape_and_categorize.py
"""
Match outage-incident dates between **local XML feeds** and **public forum pages**
for Bell, TELUS and Rogers.  
Prints any overlapping dates in a banner format.

The scrapers use only the public forum “home” pages you provided:

* Rogers https://communityforums.rogers.com/t5/Rogers-Community-Forums/ct-p/EnglishCommunity?profile.language=en  
* Bell   https://forum.bell.ca/t5/Bell-Community-Forum/ct-p/bell_en?profile.language=en&nobounce=  
* TELUS  https://forum.telus.com/t5/Home/ct-p/EN  

They look for anchor text containing outage-related keywords and grab the
nearest `<time datetime="…">` tag as the post date.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
import feedparser  # only used if you later decide to switch Bell back to RSS

from datetime import datetime, timedelta
from dateutil import parser
# ---------------------------------------------------------------------------
# New helper for date extraction
# ---------------------------------------------------------------------------

def extract_timestamp(article_tag):
    """
    Convert the <time> element inside a TELUS Neighbourhood article tile
    to a proper `datetime` instance.

    Works with:
    * absolute values in the datetime="" attribute (e.g. "05-17-2025 09:21 AM")
    * relative strings such as "yesterday", "3 hours ago", or "Friday"
    """
    time_tag = article_tag.find("time")
    if not time_tag:            # no <time> element found
        return None

    raw_attr = (time_tag.get("datetime") or "").strip()
    if raw_attr:
        try:
            # absolute timestamp already present → easy
            return parser.parse(raw_attr)
        except (ValueError, TypeError):
            pass    # fall back to human‑readable text

    human = (time_tag.get_text(strip=True) or "").lower()

    if human == "yesterday":
        return datetime.now() - timedelta(days=1)

    if human.endswith("hours ago"):
        hrs = int(human.split()[0])
        return datetime.now() - timedelta(hours=hrs)

    if human.endswith("minutes ago"):
        mins = int(human.split()[0])
        return datetime.now() - timedelta(minutes=mins)

    if human.endswith("days ago"):
        days = int(human.split()[0])
        return datetime.now() - timedelta(days=days)

    # Weekday names or month/day strings, e.g. "Friday" or "March"
    try:
        return parser.parse(human, fuzzy=True)
    except (ValueError, TypeError):
        return None

# Optional — bypass some Cloudflare sites if installed
try:
    import cloudscraper
except ImportError:
    cloudscraper = None

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36 IncidentBot/1.0"
    )
}

from urllib.parse import urljoin


def fetch(
    url: str,
    *,
    timeout: int = 20,
    json_: bool = False,
    extra_headers: dict | None = None,
):
    """
    GET helper with the polite UA header. If the first request raises or
    returns 403/503, fall back to `cloudscraper` (if installed).

    * `json_=True`  → return parsed JSON (dict) or None on failure
    * `json_=False` → return `requests.Response` or None on failure
    """
    headers = REQ_HEADERS.copy()
    if extra_headers:
        headers.update(extra_headers)

    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code not in (200, 404):
            print(f"[fetch] {url} → HTTP {r.status_code}", file=sys.stderr)
        if r.status_code in (200, 404):
            return r.json() if json_ else r
        raise RuntimeError(f"HTTP {r.status_code}")
    except Exception as e:
        if cloudscraper:
            try:
                s = cloudscraper.create_scraper()
                r = s.get(url, timeout=timeout)
                return r.json() if json_ else r
            except Exception as e2:  # noqa: F841
                return None
        return None


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def normalize_date(dt: datetime) -> datetime.date:
    """Return an Eastern-time calendar date (EDT/EST) for comparisons."""
    return dt.astimezone(timezone(timedelta(hours=-4))).date()


def cluster_dates(
    sorted_dates: List[datetime.date],
) -> List[Tuple[datetime.date, datetime.date]]:
    """Collapse consecutive calendar dates into (start, end) runs."""
    if not sorted_dates:
        return []
    runs: list[tuple[datetime.date, datetime.date]] = []
    run_start = prev = sorted_dates[0]
    for cur in sorted_dates[1:]:
        if (cur - prev).days == 1:
            prev = cur
        else:
            runs.append((run_start, prev))
            run_start = prev = cur
    runs.append((run_start, prev))
    return runs


def format_range(start: datetime.date, end: datetime.date) -> str:
    """Pretty banner label like  MAY 12-14 2025."""
    if start == end:
        return start.strftime("%b %d %Y").upper()
    if start.year == end.year and start.month == end.month:
        return f"{start.strftime('%b %d')}-{end.strftime('%d %Y')}".upper()
    if start.year == end.year:
        return f"{start.strftime('%b %d')}-{end.strftime('%b %d %Y')}".upper()
    return f"{start.strftime('%b %d %Y')}-{end.strftime('%b %d %Y')}".upper()


# ---------------------------------------------------------------------------
# XML loader
# ---------------------------------------------------------------------------


def parse_xml(file_path: pathlib.Path) -> List[Dict]:
    """
    Parse an RSS-like XML file; expect <item><pubDate>  elements.
    Returns list of {'date', 'title', 'description'} dicts.
    """
    import xml.etree.ElementTree as ET

    incidents: list[dict] = []
    root = ET.parse(file_path).getroot()
    for item in root.findall(".//item"):
        pub_date_raw = item.findtext("pubDate", "")
        if not pub_date_raw:
            continue
        try:
            date = normalize_date(dtparser.parse(pub_date_raw))
        except Exception:
            continue
        title = (item.findtext("title") or "").strip()
        descr = (item.findtext("description") or "").strip()
        incidents.append({"date": date, "title": title, "description": descr})
    return incidents


# ---------------------------------------------------------------------------
# Generic forum-thread collector
# ---------------------------------------------------------------------------


def _collect_threads(root_soup: BeautifulSoup) -> list[Tuple[str, str]]:
    """Return [(url, text)] whose anchor text contains outage-related keywords."""
    keywords = (
        "outage",
        "interruption",
        "interrup",
        "service",
        "down",
        "incident",
        "slow",
        "issue",
        "problem",
    )
    threads: list[tuple[str, str]] = []
    for a in root_soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        if not text:
            continue
        if any(k in text.lower() for k in keywords):
            threads.append((a["href"], text))
    return threads


# ---------------------------------------------------------------------------
# New helper for date extraction
# ---------------------------------------------------------------------------

def _extract_date_from_thread(t_soup: BeautifulSoup) -> datetime.date | None:
    """
    Return a normalized date from a thread page.

    Order of precedence:
    1. <span itemprop="dateCreated" content="ISO-8601">
    2. <meta itemprop="datePublished" content="ISO-8601">
    3. Text inside <span class="DateTime"> (forum-locale string)

    Returns None if no parseable date is found.
    """
    iso = t_soup.find(attrs={"itemprop": "dateCreated"})
    if iso and iso.get("content"):
        try:
            return normalize_date(dtparser.parse(iso["content"]))
        except Exception:
            pass

    meta = t_soup.find("meta", attrs={"itemprop": "datePublished"})
    if meta and meta.get("content"):
        try:
            return normalize_date(dtparser.parse(meta["content"]))
        except Exception:
            pass

    span = t_soup.find("span", class_="DateTime")
    if span:
        try:
            return normalize_date(dtparser.parse(span.get_text(" ", strip=True)))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rogers forum threads: nested <span class="local-date"> and
    # <span class="local-time"> inside a parent <span class="DateTime">.
    # Example HTML:
    #   <span class="DateTime">
    #       <span class="local-date">02-28-2023</span>
    #       <span class="local-time">01:57 PM</span>
    #   </span>
    # ------------------------------------------------------------------
    ld = t_soup.find("span", class_="local-date")
    lt = t_soup.find("span", class_="local-time")
    if ld and lt:
        try:
            combined = f"{ld.get_text(strip=True)} {lt.get_text(strip=True)}"
            return normalize_date(dtparser.parse(combined))
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------


def scrape_rogers() -> List[Dict]:
    """
    Scrape recent Rogers Community Forum threads.

    The landing page at
    https://communityforums.rogers.com/t5/Rogers-Community-Forums/ct-p/EnglishCommunity
    populates its message list with client‑side JavaScript, so the static
    HTML we download does not contain any real thread links.  Instead we
    use the server‑rendered *recent posts* feed which returns normal HTML
    that BeautifulSoup can parse:

        https://communityforums.rogers.com/t5/forums/recentpostspage/post-type/thread/page/1

    We then:
    1. Collect all anchors whose text contains outage‑related keywords
       (delegated to the existing `_collect_threads` helper).
    2. Try to extract the post timestamp directly from the list‑row
       (each row already contains a `<span class="DateTime">` subtree).
       If that fails, fall back to fetching the thread page itself.
    3. Normalise the timestamp to an Eastern‑time calendar date and keep
       only items for which we have both a date and a title.
    """
    root = "https://communityforums.rogers.com"
    recent_url = f"{root}/t5/forums/recentpostspage/post-type/thread/page/1"
    resp = fetch(recent_url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Re‑use the generic anchor/keyword matcher on the recent‑posts feed
    candidate_links = _collect_threads(soup)
    print("[DEBUG] ROGERS anchors matching keywords:",
          len(candidate_links), file=sys.stderr)

    incidents: list[dict] = []
    for href, title in candidate_links:
        full_url = urljoin(root, href)

        # Attempt to grab the timestamp from the list entry itself
        dt = None
        a_tag = soup.find("a", href=href)
        if a_tag:
            li_row = a_tag.find_parent("li")
            if li_row:
                dt = _extract_date_from_thread(li_row)

        # Fallback: open the thread page if we still have no timestamp
        if dt is None:
            page = fetch(full_url)
            if page:
                t_soup = BeautifulSoup(page.text, "html.parser")
                dt = _extract_date_from_thread(t_soup)

        if dt is None:
            continue  # skip if we could not determine a date

        # `dt` may already be a normalized `datetime.date` if the helper
        # returned one; avoid re‑normalizing to prevent attribute errors.
        incidents.append({
            "date": normalize_date(dt) if isinstance(dt, datetime) else dt,
            "title": title,
            "url": full_url,
        })

    return incidents


def scrape_bell() -> List[Dict]:
    """
    Collect recent Bell Community Forum threads from the landing page and
    pull the timestamp shown in the `<time datetime="MM-dd-yyyy hh:mm a">`
    element that accompanies every article‑tile.

    The function returns a list of dicts with *normalized* `date`
    (calendar day in Eastern time) so the downstream intersection logic
    works exactly like for TELUS / Rogers.
    """
    root = "https://forum.bell.ca"
    home_url = f"{root}/t5/Bell-Community-Forum/ct-p/bell_en?profile.language=en&nobounce="
    resp = fetch(home_url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    print(
        "[DEBUG] BELL article tiles found:",
        len(soup.select("article")),
        file=sys.stderr,
    )

    incidents: list[dict] = []
    for art in soup.select("article"):
        a = art.find("a", href=True)
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        href = urljoin(root, a["href"])

        published_dt = extract_timestamp(art)
        if not published_dt:
            continue

        incidents.append(
            {
                "title": title,
                "url": href,
                "published": published_dt,
                # save the calendar day right away for later set & map ops
                "date": normalize_date(published_dt),
            }
        )
    return incidents


def scrape_telus() -> List[Dict]:
    root = "https://forum.telus.com"
    home_url = f"{root}/t5/Home/ct-p/EN"
    resp = fetch(home_url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    print("[DEBUG] TELUS anchors matching keywords:",
          len(_collect_threads(soup)), file=sys.stderr)

    incidents: list[dict] = []
    # Find all article tiles and extract info
    for art in soup.select("article"):
        # title/href extraction (existing logic or fallback)
        a = art.find("a", href=True)
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        href = a["href"]
        published_dt = extract_timestamp(art)
        if not published_dt:
            continue  # skip tiles lacking a usable timestamp
        date = normalize_date(published_dt)
        incidents.append({
            "date": date,
            "title": title,
            "url": urljoin(root, href),
        })
    return incidents


SCRAPERS = {
    "ROGERS": scrape_rogers,
    "TELUS": scrape_telus,
    "BELL": scrape_bell,
}

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

XML_FILES = {
    "ROGERS": SCRIPT_DIR / "ROGERS XML.xml",
    "TELUS": SCRIPT_DIR / "TELUS XML.xml",
    "BELL": SCRIPT_DIR / "Bell Canada XML.xml",
}

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> None:
    for carrier in ("BELL", "TELUS", "ROGERS"):  # print in this order
        print(f"\n=== {carrier} ===", file=sys.stderr)
        # ---------------- XML ----------------
        if not XML_FILES[carrier].exists():
            print(f"  XML file not found → {XML_FILES[carrier]}", file=sys.stderr)
            continue
        xml_inc = parse_xml(XML_FILES[carrier])
        print(f"  XML rows:   {len(xml_inc)}", file=sys.stderr)

        # ---------------- Web ----------------
        try:
            www_inc = SCRAPERS[carrier]()
        except Exception as exc:
            print(f"  Scraper error: {exc}", file=sys.stderr)
            www_inc = []
        print(f"  Web rows:   {len(www_inc)}", file=sys.stderr)

        # --- Ensure each web incident has a `date` field -------------
        for inc in www_inc:
            if "date" not in inc and "published" in inc and isinstance(
                inc["published"], datetime
            ):
                inc["date"] = normalize_date(inc["published"])

        # ------------- Intersection ----------
        xml_days = {d["date"] for d in xml_inc}
        www_days = {d["date"] for d in www_inc}
        shared = sorted(xml_days & www_days)
        print(f"  Shared days: {len(shared)}", file=sys.stderr)
        if not shared:
            continue

        # Map date → lists for printing
        xml_by_day: defaultdict = defaultdict(list)
        www_by_day: defaultdict = defaultdict(list)
        for d in xml_inc:
            xml_by_day[d["date"]].append(d)
        for d in www_inc:
            www_by_day[d["date"]].append(d)

        for start, end in cluster_dates(shared):
            print(
                f"========= {carrier} INCIDENT {format_range(start, end)} ==========="  # noqa: E501
            )
            print("XML Incidents:")
            for day in (
                start + timedelta(days=i)
                for i in range((end - start).days + 1)
            ):
                for inc in xml_by_day.get(day, []):
                    print(f"    • {inc['title'] or '(no title)'}")

            print("\nForum Incidents:")
            for day in (
                start + timedelta(days=i)
                for i in range((end - start).days + 1)
            ):
                for inc in www_by_day.get(day, []):
                    print(f"    • {inc['title']}  ({inc.get('url', '')})")
            print("=============================================\n")


if __name__ == "__main__":
    main()