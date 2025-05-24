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

import re
from datetime import datetime, timedelta
from dateutil import parser

# ---------------------------------------------------------------------------
# LLM environment and setup (Watsonx Llama for complaint classification)
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx

# Load credentials and initialize LLM for complaint classification
load_dotenv()
_watsonx_url = os.getenv("WATSONX_URL")
_watsonx_apikey = os.getenv("WATSONX_API_KEY")
_watsonx_project = os.getenv("WATSONX_PROJECT_ID")

_model_id_llama = "meta-llama/llama-3-405b-instruct"
_llm_params = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42
}

llm_llama = ChatWatsonx(
    model_id=_model_id_llama,
    url=_watsonx_url,
    apikey=_watsonx_apikey,
    project_id=_watsonx_project,
    params=_llm_params
)

 # Complaint-related keywords for initial thread filtering
COMPLAINT_KEYWORDS = (
     "outage", "issue", "problem", "down", "slow", "interruption", "service", "interrupt"
 )

# Helper function to classify if a text is a complaint using llm_llama
def is_complaint(text: str) -> bool:
    """
    Use llm_llama to determine if the provided text is a user complaint.
    Returns True for complaints, False otherwise.
    """
    prompt = (
        "You are a classification assistant. "
        "Reply with 'YES' if the following forum post is a user complaint "
        "describing a service outage or problem; otherwise reply 'NO'.\n\n"
        "Post:\n\"\"\"\n" + text + "\n\"\"\"\n\nAnswer:"
    )
    resp = llm_llama.invoke(prompt)
    # Normalize LLM output
    reply = getattr(resp, "content", str(resp)).strip().upper()
    return reply.startswith("YES")

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
# Helper to scrape replies in a thread for outage-related complaints
# ---------------------------------------------------------------------------
def scrape_thread_replies(full_url: str, keywords: tuple[str, ...]) -> list[Dict]:
    """
    Fetch a forum thread page and return any replies whose text contains
    outage-related keywords, with their dates and full text.
    Only LLM-verified complaint posts are returned.
    """
    resp = fetch(full_url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    replies: list[dict] = []
    # Select each reply body in Khoros-based forums
    for content_div in soup.select("div.lia-message-body-content"):
        post = content_div.find_parent() or content_div
        text = content_div.get_text(" ", strip=True)
        if any(k in text.lower() for k in keywords):
            # Extract date from the surrounding message element
            dt = extract_timestamp(post) or _extract_date_from_thread(post)
            if not dt:
                continue
            # Use LLM to filter only actual complaints
            if not is_complaint(text):
                continue
            replies.append({
                "date": normalize_date(dt) if isinstance(dt, datetime) else dt,
                "title": text[:80] + ("..." if len(text) > 80 else ""),
                "url": full_url,
                "description": text,
            })
    return replies


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

        incidents.append({
            "date": normalize_date(dt) if isinstance(dt, datetime) else dt,
            "title": title,
            "url": full_url,
            "description": title,  # use title as fallback description
        })
        # Scrape replies in this thread for actual user complaints
        replies = scrape_thread_replies(full_url, (
            "outage", "issue", "problem", "down", "slow", "interrupt", "service"))
        incidents.extend(replies)

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
                "description": title,  # use title as fallback description
            }
        )
        # now capture replies
        replies = scrape_thread_replies(href, (
            "outage", "issue", "problem", "down", "slow", "interrupt", "service"))
        incidents.extend(replies)
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
        url = urljoin(root, href)
        incidents.append({
            "date": date,
            "title": title,
            "url": url,
            "description": title,  # use title as fallback description
        })
        replies = scrape_thread_replies(url, (
            "outage", "issue", "problem", "down", "slow", "interrupt", "service"))
        incidents.extend(replies)
    return incidents

# Preserve forum‐only scrapers to avoid recursion in scrape_all
FORUM_SCRAPERS = {
    "ROGERS": scrape_rogers,
    "TELUS":  scrape_telus,
    "BELL":   scrape_bell,
}

# ---------------------------------------------------------------------------
# Outage.Report and IsTheServiceDownCanada scrapers
# ---------------------------------------------------------------------------

def scrape_outage_report(provider: str) -> List[Dict]:
    """
    Scrape Outage.Report for a given provider.
    Returns list of {'date', 'source', 'count', 'summary'} dicts.
    """
    slug = provider.lower()
    url = f"https://outage.report/ca/{slug}"
    resp = fetch(url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    # find the first paragraph with "report"
    summary_tag = soup.find("p", string=re.compile(r"report", re.IGNORECASE))
    summary = summary_tag.text.strip() if summary_tag else ""
    # extract report count
    count = 0
    if summary:
        m = re.search(r"received reports? (?:in the past 24 hours)?(?:.*?)(\d+)", summary)
        if m:
            count = int(m.group(1))
    return [{
        "date": datetime.now().date(),
        "source": "outage_report",
        "count": count,
        "summary": summary,
    }]

def scrape_istheservicedown(provider: str) -> List[Dict]:
    """
    Scrape IsTheServiceDownCanada.com for a given provider.
    Returns list of {'date', 'source', 'status'} dicts.
    """
    url = "https://istheservicedowncanada.com/"
    resp = fetch(url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    # look for provider name in page text
    status = ""
    tag = soup.find(string=re.compile(provider, re.IGNORECASE))
    if tag:
        status = tag.strip()
    return [{
        "date": datetime.now().date(),
        "source": "istheservicedown",
        "status": status,
    }]

def scrape_all(provider: str) -> List[Dict]:
    """
    Combine forum-thread scraper with outage-report and service-down data.
    """
    # base incidents from forum
    base = FORUM_SCRAPERS.get(provider, lambda: [])()
    # augmentation from Outage.Report
    base += scrape_outage_report(provider)
    # augmentation from IsTheServiceDownCanada
    base += scrape_istheservicedown(provider)
    return base

# Use combined scraper for each provider
SCRAPERS = {
    "ROGERS": lambda: scrape_all("ROGERS"),
    "TELUS":  lambda: scrape_all("TELUS"),
    "BELL":   lambda: scrape_all("BELL"),
}

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------

# XML paths
import pathlib  # ensure pathlib is imported at top
XML_FILES = {
    "BELL":   pathlib.Path("/Users/denizaskin/telus-incident-mgmt-copilot/notebooks/wxO_api/TELUS_incident_management/xml_files/Bell_Canada_2025.xml"),
    "ROGERS": pathlib.Path("/Users/denizaskin/telus-incident-mgmt-copilot/notebooks/wxO_api/TELUS_incident_management/xml_files/Rogers_2025.xml"),
    "TELUS":  pathlib.Path("/Users/denizaskin/telus-incident-mgmt-copilot/notebooks/wxO_api/TELUS_incident_management/xml_files/TELUS_2025.xml"),
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
        # Fetch all raw forum entries (initial posts + replies)
        try:
            forum_all = FORUM_SCRAPERS[carrier]()
        except Exception as exc:
            print(f"  Forum scraper error: {exc}", file=sys.stderr)
            forum_all = []

        # Keep only LLM-verified complaints based on the full post text
        forum_inc = []
        for inc in forum_all:
            text = inc.get("description", "")
            if text and is_complaint(text):
                forum_inc.append(inc)
        print(f"  Forum rows (complaints only): {len(forum_inc)}", file=sys.stderr)
        # Print the scraped forum reply content, if available
        print("  Scraped forum post contents:", file=sys.stderr)
        for inc in forum_inc:
            date = inc.get("date", "(no date)")
            title = inc.get("title", "(no title)")
            desc = inc.get("description", "").strip()
            if desc:
                print(f"    • [{date}] {title}\n      {desc}\n", file=sys.stderr)
            else:
                # no detailed description → print thread title as fallback
                print(f"    • [{date}] {title}  (no specific complaint text)", file=sys.stderr)

        # Fetch Outage.Report data
        try:
            outage_inc = scrape_outage_report(carrier)
        except Exception as exc:
            print(f"  Outage.Report error: {exc}", file=sys.stderr)
            outage_inc = []
        print(f"  Outage.Report rows: {len(outage_inc)}", file=sys.stderr)

        # Fetch IsTheServiceDownCanada data
        try:
            down_inc = scrape_istheservicedown(carrier)
        except Exception as exc:
            print(f"  IsTheServiceDownCanada error: {exc}", file=sys.stderr)
            down_inc = []
        print(f"  IsTheServiceDownCanada rows: {len(down_inc)}", file=sys.stderr)

        # Combine all web incidents for matching
        www_inc = forum_inc + outage_inc + down_inc
        print(f"  Total web rows: {len(www_inc)}", file=sys.stderr)

        # --- Ensure each web incident has a `date` field -------------
        for inc in www_inc:
            if "date" not in inc and "published" in inc and isinstance(
                inc["published"], datetime
            ):
                inc["date"] = normalize_date(inc["published"])

        # ------------- Intersection (daily and weekly) ----------
        # Daily matching
        xml_days = {d["date"] for d in xml_inc}
        www_days = {d["date"] for d in www_inc}
        shared_days = sorted(xml_days & www_days)
        print(f"  Shared days count: {len(shared_days)}", file=sys.stderr)
        print(f"  Shared dates: {shared_days}", file=sys.stderr)

        # Weekly matching (ISO year-week)
        xml_weeks = {d["date"].isocalendar()[:2] for d in xml_inc}
        www_weeks = {d["date"].isocalendar()[:2] for d in www_inc}
        shared_weeks = sorted(xml_weeks & www_weeks)
        week_strs = [f"{year}-W{week:02d}" for year, week in shared_weeks]
        print(f"  Shared weeks count: {len(shared_weeks)}", file=sys.stderr)
        print(f"  Shared weeks: {week_strs}", file=sys.stderr)

        # Decide which window to use: switch to weekly if no daily matches
        if shared_days:
            match_key = "day"
            shared = shared_days
            xml_by_period = defaultdict(list)
            www_by_period = defaultdict(list)
            for d in xml_inc:
                xml_by_period[d["date"]].append(d)
            for d in www_inc:
                www_by_period[d["date"]].append(d)
            cluster_fn = cluster_dates
        elif shared_weeks:
            match_key = "week"
            shared = shared_weeks
            # build week-based maps
            xml_by_period = defaultdict(list)
            www_by_period = defaultdict(list)
            for inc in xml_inc:
                y, w = inc["date"].isocalendar()[:2]
                xml_by_period[(y, w)].append(inc)
            for inc in www_inc:
                y, w = inc["date"].isocalendar()[:2]
                www_by_period[(y, w)].append(inc)
            cluster_fn = lambda weeks: weeks  # no further clustering for weeks
        else:
            # no matches
            continue

        # Rendering loop based on match granularity
        if match_key == "day":
            for start, end in cluster_fn(shared):
                print(
                    f"========= {carrier} INCIDENT {format_range(start, end)} ==========="  # noqa: E501
                )
                print("XML Incidents:")
                for day in (
                    start + timedelta(days=i)
                    for i in range((end - start).days + 1)
                ):
                    for inc in xml_by_period.get(day, []):
                        print(f"    • {inc['title'] or '(no title)'}")

                print("\nForum Incidents:")
                for day in (
                    start + timedelta(days=i)
                    for i in range((end - start).days + 1)
                ):
                    for inc in www_by_period.get(day, []):
                        print(f"    • {inc['title']}  ({inc.get('url', '')})")
                print("=============================================\n")
        else:  # match_key == "week"
            for year, week in shared:
                print(f"========= {carrier} INCIDENT {year}-W{week:02d} ==========", file=sys.stderr)
                print("XML Incidents:")
                for inc in xml_by_period[(year, week)]:
                    print(f"    • {inc['title']}  [{inc['date']}]")
                print("\nForum Incidents:")
                for inc in www_by_period[(year, week)]:
                    print(f"    • {inc.get('title','(no title)')} ({inc.get('url','')})")
                print("=============================================", file=sys.stderr)


if __name__ == "__main__":
    main()