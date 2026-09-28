"""Download official Federal Reserve FOMC minutes for 2000-2024.

Outputs are written under:
  fed_put_ai_nlp_outputs/data_raw/fomc_minutes_official_2000_2024

The script stores raw HTML/PDF where available and writes a normalized TXT file
under Minutes/<year>/ so run_pipeline.py can ingest the downloaded minutes
without modifying the original local source files.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from project_paths import OUT_DIR

OUT_ROOT = (
    OUT_DIR
    / "data_raw"
    / "fomc_minutes_official_2000_2024"
)
RAW_HTML_DIR = OUT_ROOT / "raw_html"
RAW_PDF_DIR = OUT_ROOT / "raw_pdf"
TEXT_ROOT = OUT_ROOT / "Minutes"
MANIFEST_PATH = OUT_ROOT / "official_fomc_minutes_manifest.csv"
LOG_PATH = OUT_ROOT / "download_log.txt"

BASE_URL = "https://www.federalreserve.gov"
UA = "Mozilla/5.0 academic research downloader"


@dataclass
class MinuteLinks:
    date: str
    year: int
    html_urls: set[str] = field(default_factory=set)
    pdf_urls: set[str] = field(default_factory=set)
    source_pages: set[str] = field(default_factory=set)


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attrs_dict = dict(attrs)
            self._current_href = attrs_dict.get("href")
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            text = " ".join("".join(self._text_parts).split())
            self.links.append((text, self._current_href))
            self._current_href = None
            self._text_parts = []


class TextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.parts.append(stripped)


class ScopedTextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self, target_id: str) -> None:
        super().__init__()
        self.target_id = target_id
        self.parts: list[str] = []
        self._active = False
        self._active_depth = 0
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        tag = tag.lower()
        if not self._active and attrs_dict.get("id") == self.target_id:
            self._active = True
            self._active_depth = 1
        elif self._active:
            self._active_depth += 1

        if not self._active:
            return
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self._active:
            return
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"}:
            self.parts.append("\n")
        self._active_depth -= 1
        if self._active_depth <= 0:
            self._active = False

    def handle_data(self, data: str) -> None:
        if self._active and self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.parts.append(stripped)


def log(message: str) -> None:
    print(message)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_text(url: str) -> str:
    return fetch_bytes(url).decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_date_from_href(href: str) -> str | None:
    patterns = [
        r"fomcminutes((?:19|20)\d{6})",
        r"/minutes/((?:19|20)\d{6})",
    ]
    for pattern in patterns:
        match = re.search(pattern, href, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def discover_links() -> dict[str, MinuteLinks]:
    pages = [
        f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
        for year in range(2000, 2021)
    ]
    pages.append(f"{BASE_URL}/monetarypolicy/fomccalendars.htm")

    by_date: dict[str, MinuteLinks] = {}
    for page in pages:
        try:
            html = fetch_text(page)
        except Exception as exc:
            log(f"DISCOVERY_ERROR page={page} error={exc}")
            continue
        parser = LinkExtractor()
        parser.feed(html)
        found = 0
        for link_text, link_href in parser.links:
            href = urljoin(BASE_URL, link_href)
            lower_href = href.lower()
            text = " ".join(link_text.split()).lower()
            if (
                "minutes" not in text
                and "fomcminutes" not in lower_href
                and "/minutes/" not in lower_href
            ):
                continue
            date_key = extract_date_from_href(lower_href)
            if not date_key:
                continue
            year = int(date_key[:4])
            if not 2000 <= year <= 2024:
                continue
            links = by_date.setdefault(date_key, MinuteLinks(date=date_key, year=year))
            if lower_href.endswith(".pdf"):
                links.pdf_urls.add(href)
            elif lower_href.endswith(".htm") or lower_href.endswith(".html"):
                links.html_urls.add(href)
            links.source_pages.add(page)
            found += 1
        log(f"DISCOVERED page={page} minute_links={found}")
    merge_adjacent_pdf_only_links(by_date)
    return by_date


def merge_adjacent_pdf_only_links(by_date: dict[str, MinuteLinks]) -> None:
    """Merge Fed pages where PDF names use the first day of a two-day meeting.

    Example: October 23-24, 2012 has an HTML page keyed as 20121024 and a PDF
    file named fomcminutes20121023.pdf. Treat those as one meeting.
    """
    for date_key in sorted(list(by_date)):
        links = by_date.get(date_key)
        if not links or links.html_urls or not links.pdf_urls:
            continue
        try:
            next_key = (datetime.strptime(date_key, "%Y%m%d") + timedelta(days=1)).strftime(
                "%Y%m%d"
            )
        except ValueError:
            continue
        target = by_date.get(next_key)
        if target and target.html_urls and not target.pdf_urls:
            target.pdf_urls.update(links.pdf_urls)
            target.source_pages.update(links.source_pages)
            del by_date[date_key]


def extract_html_minutes_text(html: str) -> str:
    for target_id in ["article", "content"]:
        scoped_parser = ScopedTextExtractor(target_id)
        scoped_parser.feed(html)
        scoped_text = unescape(" ".join(scoped_parser.parts))
        scoped_lines = [line.strip() for line in scoped_text.splitlines()]
        scoped_lines = [line for line in scoped_lines if line]
        cleaned = clean_text("\n".join(scoped_lines))
        if len(cleaned.split()) >= 500:
            return cleaned

    parser = TextExtractor()
    parser.feed(html)
    text = unescape(" ".join(parser.parts))
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    # Remove repeated site-navigation labels where possible.
    drop_prefixes = {
        "board of governors of the federal reserve system",
        "federal reserve",
        "monetary policy",
        "about the fomc",
        "meeting calendars and information",
        "transcripts and other historical materials",
    }
    filtered = [line for line in lines if line.lower() not in drop_prefixes]
    return clean_text("\n".join(filtered))


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return clean_text(result.stdout)
    except FileNotFoundError:
        pass
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))
    except Exception as exc:
        log(f"PDF_TEXT_ERROR path={pdf_path} error={exc}")
        return ""


def download_one(links: MinuteLinks) -> dict[str, object]:
    year_dir = TEXT_ROOT / str(links.year)
    html_dir = RAW_HTML_DIR / str(links.year)
    pdf_dir = RAW_PDF_DIR / str(links.year)
    for d in [year_dir, html_dir, pdf_dir]:
        d.mkdir(parents=True, exist_ok=True)

    html_url = sorted(links.html_urls)[0] if links.html_urls else ""
    pdf_url = sorted(links.pdf_urls)[0] if links.pdf_urls else ""
    raw_html_path = ""
    raw_pdf_path = ""
    text = ""
    extraction_source = ""
    status = "ok"
    error = ""

    try:
        if html_url:
            html = fetch_text(html_url)
            raw_path = html_dir / f"fomcminutes{links.date}.htm"
            raw_path.write_text(html, encoding="utf-8")
            raw_html_path = str(raw_path)
            text = extract_html_minutes_text(html)
            extraction_source = "html"
            time.sleep(0.08)

        if pdf_url:
            pdf_bytes = fetch_bytes(pdf_url)
            pdf_path = pdf_dir / f"fomcminutes{links.date}.pdf"
            pdf_path.write_bytes(pdf_bytes)
            raw_pdf_path = str(pdf_path)
            if not text or len(text.split()) < 100:
                text = extract_pdf_text(pdf_path)
                extraction_source = "pdf"
            time.sleep(0.08)

        if not text or len(text.split()) < 50:
            status = "text_too_short"

    except Exception as exc:
        status = "error"
        error = str(exc)

    text_path = year_dir / f"fomcminutes{links.date}.txt"
    if text:
        header = (
            f"Source: Federal Reserve official FOMC minutes\n"
            f"Meeting date key: {links.date}\n"
            f"HTML URL: {html_url}\n"
            f"PDF URL: {pdf_url}\n\n"
        )
        text_path.write_text(header + text + "\n", encoding="utf-8")

    return {
        "date": f"{links.date[:4]}-{links.date[4:6]}-{links.date[6:]}",
        "date_key": links.date,
        "year": links.year,
        "status": status,
        "extraction_source": extraction_source,
        "word_count": len(text.split()) if text else 0,
        "html_url": html_url,
        "pdf_url": pdf_url,
        "text_path": str(text_path) if text_path.exists() else "",
        "raw_html_path": raw_html_path,
        "raw_pdf_path": raw_pdf_path,
        "source_pages": ";".join(sorted(links.source_pages)),
        "error": error,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [RAW_HTML_DIR, RAW_PDF_DIR, TEXT_ROOT]:
        if path.exists():
            shutil.rmtree(path)
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    links_by_date = discover_links()
    log(f"TOTAL_DISCOVERED unique_minutes={len(links_by_date)}")

    rows = []
    for date_key in sorted(links_by_date):
        row = download_one(links_by_date[date_key])
        rows.append(row)
        log(
            "DOWNLOADED "
            f"date={row['date']} status={row['status']} "
            f"words={row['word_count']} source={row['extraction_source']}"
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    short = sum(1 for r in rows if r["status"] == "text_too_short")
    err = sum(1 for r in rows if r["status"] == "error")
    log(f"SUMMARY ok={ok} text_too_short={short} error={err} manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
