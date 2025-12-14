# pipeline_sample/custom_scrapers.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, cast, Optional 

import feedparser
import requests
from bs4 import BeautifulSoup
import re
from ingest.crawler_dw import main as crawler_dw
from ingest.utils import is_urls_processed_already, fetch_and_extract
from lib.repositories.link_pool_repository import LinkPoolRepository

repo = LinkPoolRepository()
BLOOMBERG_RSS_FEEDS = {
    "markets": "https://feeds.bloomberg.com/markets/news.rss",
    "politics": "https://feeds.bloomberg.com/politics/news.rss",
    "technology": "https://feeds.bloomberg.com/technology/news.rss",
    "wealth": "https://feeds.bloomberg.com/wealth/news.rss",
}



LANACION_BASE_URL = "https://www.lanacion.com.ar/ultimas-noticias/"
LANACION_ARTICLE_RE = re.compile(
    r"^https?://(www\.)?lanacion\.com\.ar/.+-nid\d+/?$"
)


ELUNIVERSAL_BASE_URL = "https://www.eluniversal.com.mx/"
ELUNIVERSAL_ARTICLE_RE = re.compile(
    r"^https?://(www\.)?eluniversal\.com\.mx/.+/.+/?$",
    re.IGNORECASE,
)

BME_RSS_FEEDS = {
    "press_releases": "https://www.bolsasymercados.es/es/sala-de-comunicacion/notas-de-prensa.rss.xml",
    "news": "https://www.bolsasymercados.es/es/sala-de-comunicacion/noticias.rss.xml",
    "reports": "https://www.bolsasymercados.es/es/estudios-y-publicaciones/reportajes.rss.xml",
    "blog": "https://www.bolsasymercados.es/es/blog.rss.xml",
}

BME_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
}

def _fetch_rss_xml(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch RSS XML content via requests (more WAF-friendly than feedparser.parse(url))."""
    try:
        res = requests.get(url, headers=BME_HEADERS, timeout=timeout)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"[bme] Error fetching RSS {url}: {e}")
        return None
    
def scrape_bme_stream() -> Iterable[Dict]:
    seen_urls: set[str] = set()

    for feed_key, feed_url in BME_RSS_FEEDS.items():
        xml = _fetch_rss_xml(feed_url)
        if not xml:
            continue
        feed = feedparser.parse(xml)
        entries = getattr(feed, "entries", []) or []
        for entry in entries:
            url = (entry.get("link") or "").strip()
            title = (entry.get("title") or "").strip()
            if not url or not title:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if is_urls_processed_already(url):
                continue
            text = fetch_and_extract(url)
            if not text:
                # fallback: resumen RSS (si existe)
                text = (entry.get("summary") or "").strip()
            if not text:
                continue

            try:
                repo.insert_link({"url": url})
            except Exception as e:
                print(f"[bme] Warning inserting link: {e}")
            yield {
                "title": title,
                "url": url,
                "text": text,
                "source": f"bme-{feed_key}",
                "scraped_at": datetime.now(timezone.utc),
            }


def scrape_lanacion_stream() -> Iterable[Dict]:
    try:
        res = requests.get(LANACION_BASE_URL, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"[la-nacion] Error scraping homepage: {e}")
        return

    seen_urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # Normalizamos URLs relativas
        if href.startswith("/"):
            href = "https://www.lanacion.com.ar" + href

        # Nos quedamos solo con URLs que encajan con el patrón de artículo (acabado en nnddyyyy)
        if not LANACION_ARTICLE_RE.match(href):
            continue

        # Evitar duplicados en el propio scrapper
        if href in seen_urls:
            continue
        seen_urls.add(href)
        title = a.get_text(strip=True)
        if not title:
            continue
        if is_urls_processed_already(href):
            continue
        full_text = fetch_and_extract(href)
        if not full_text:
            continue
        repo.insert_link({"url": href})
        yield {
            "title": title,
            "url": href,
            "text": full_text,
            "source": "la-nacion-ar",
            "scraped_at": datetime.now(timezone.utc),
        }


def get_title_from_dw_url(url: str) -> str:
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        title_tag = soup.find("h1")
        if title_tag:
            return title_tag.get_text(strip=True)
    except Exception as e:
        print(f"Error fetching title from DW URL {url}: {e}")
    return "DW Article"



def scrape_eluniversal_stream() -> Iterable[Dict]:
    # para evitar que nos interprete la web como bot, añadimos headers de navegadores:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        res = requests.get(ELUNIVERSAL_BASE_URL, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"[el-universal] Error scraping homepage: {e}")
        return
    seen_urls: set[str] = set()
    # Secciones que nos podrían interesar
    VALID_PATH_PREFIXES = (
    "/nacion/",
    "/mundo/",
    "/metropoli/",
    "/estados/",
    "/politica/",
    "/cartera/",
    "/cultura/",
    "/deportes/",
    "/ciencia-y-salud/",
    "/techbit/",
    "/seguridad/",
    "/cdmx/",
)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Normalizar URLs relativas o protocolo-relativas
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = "https://www.eluniversal.com.mx" + href
        if not href.startswith("https://www.eluniversal.com.mx"):
            continue
        # Filtrar por rutas que parezcan notícias
        from urllib.parse import urlparse
        path = urlparse(href).path or ""
        if not any(path.startswith(prefix) for prefix in VALID_PATH_PREFIXES):
            continue
        if not ELUNIVERSAL_ARTICLE_RE.match(href):
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)
        title = a.get_text(strip=True)
        # A veces los enlaces tienen solo “Leer más” o cosas muy cortas
        if not title or len(title) < 8:
            continue
        if is_urls_processed_already(href):
            continue
        full_text = fetch_and_extract(href)
        if not full_text:
            continue
        try:
            repo.insert_link({"url": href})
        except Exception as e:
            print(f"[el-universal] Warning inserting link: {e}")
        yield {
            "title": title,
            "url": href,
            "text": full_text,
            "source": "el-universal-mx",
            "scraped_at": datetime.now(timezone.utc),
        }


def scrape_bbc_stream() -> Iterable[Dict]:
    """Yield BBC articles. No DB writes, no link_pool checks."""
    url_bbc = "https://www.bbc.com/news"
    try:
        res = requests.get(url_bbc, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"Error scraping BBC homepage: {e}")
        return

    for link in soup.select("a[href^='/news'] h2"):
        title = link.get_text(strip=True)
        parent = link.find_parent("a")
        href = parent.get("href") if parent else ""
        full_url = "https://www.bbc.com" + href if href.startswith("/") else href
        if not full_url:
            continue
        if is_urls_processed_already(full_url):
            continue
        full_text = fetch_and_extract(full_url)
        if not full_text:
            continue
        repo.insert_link({"url": full_url})
        yield {
            "title": title,
            "url": full_url,
            "text": full_text,
            "source": "bbc-news",
            "scraped_at": datetime.now(timezone.utc),
        }


def scrape_cnn_stream() -> Iterable[Dict]:
    url_cnn = "https://edition.cnn.com/world"
    try:
        res = requests.get(url_cnn, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        print(f"Error scraping CNN homepage: {e}")
        return

    for link in soup.select("a[data-link-type='article']"):
        href = link.get("href", "")
        if not href:
            continue
        full_url = "https://edition.cnn.com" + href if href.startswith("/") else href

        title_tag = link.select_one(".container__headline-text, [data-editable='headline']")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        if is_urls_processed_already(full_url):
            continue
        full_text = fetch_and_extract(full_url)
        if not full_text:
            continue
        repo.insert_link({"url": full_url})
        yield {
            "title": title,
            "url": full_url,
            "text": full_text,
            "source": "cnn",
            "scraped_at": datetime.now(timezone.utc),
        }


def scrape_wsj_stream() -> Iterable[Dict]:
    rss_url = "https://feeds.a.dj.com/rss/RSSWorldNews.xml"
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"Error parsing WSJ RSS feed: {e}")
        return

    for entry in feed.entries:
        url = entry.get("link")
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        if not url or not title or not summary:
            continue
        if is_urls_processed_already(url):
            continue
        repo.insert_link({"url": url})
        yield {
            "title": title,
            "url": url,
            "text": summary,
            "source": "the-wall-street-journal",
            "scraped_at": datetime.now(timezone.utc),
        }


def scrape_aljazeera() -> Iterable[Dict]:
    import feedparser
    from datetime import datetime, timezone
    feed = feedparser.parse("https://www.aljazeera.com/xml/rss/all.xml")
    for e in feed.entries:
        url = e.get("link")
        title = (e.get("title") or "").strip()
        if not url or not title:
            continue
        if is_urls_processed_already(url):
            continue
        text = fetch_and_extract(url)
        if not text:
            continue
        repo.insert_link({"url": url})
        yield {
            "title": title,
            "url": url,
            "text": text,
            "source": "aljazeera",
            "scraped_at": datetime.now(timezone.utc),
        }


def scrape_dw_stream() -> Iterable[Dict]:
    # crawler_dw was imported as a function (from crawler_dw import main as crawler_dw)
    # call it to get the iterable of links. Add defensive checks and logging.
    try:
        # call crawler_dw() if it's a function; otherwise use it as provided
        raw_links = crawler_dw() if callable(crawler_dw) else crawler_dw
        # help static type checkers by casting to an iterable of strings
        links_iterable = cast(Iterable[str], raw_links)
    except Exception as e:
        print(f"Error running crawler_dw: {e}")
        return

    if links_iterable is None:
        print("crawler_dw returned None; skipping DW scraping.")
        return

    if not links_iterable:
        print("crawler_dw returned no links; skipping DW scraping.")
        return

    for link in links_iterable:
        try:
            if is_urls_processed_already(link):
                continue
            full_text = fetch_and_extract(link)
            if not full_text:
                continue
            title = get_title_from_dw_url(link)
            try:
                repo.insert_link({"url": link})
            except Exception as e:
                print(f"Warning: failed to insert link into repo: {e}")
            yield {
                "title": title,
                "url": link,
                "text": full_text,
                "source": "dw",
                "scraped_at": datetime.now(timezone.utc),
            }
        except Exception as e:
            print(f"Error processing DW link {link}: {e}")
            continue
