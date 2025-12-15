# pipeline_sample/custom_scrapers.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, cast, Optional 
import json 
import feedparser
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta, timezone
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


DOGC_DATASET_ID = "n6hn-rmy7"  # Normativa del DOGC i Portal Jurídic (Open Data)
DOGC_RESOURCE_JSON = f"https://analisi.transparenciacatalunya.cat/resource/{DOGC_DATASET_ID}.json"
DOGC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}
DOGC_TITLE_KEYS = [
    "titol", "titulo", "title",
    "titol_norma", "titol_disposicio", "titol_disposicio_normativa",
    "denominacio", "denominacion",
]
DOGC_RANG_KEYS = ["rang_de_norma", "rang_norma", "tipus", "tipus_norma"]
DOGC_NUM_KEYS = ["numero_de_control", "n_mero_de_control", "num_control", "numero_control"]
DOGC_DATE_KEYS = [
    "data_publicacio", "data_publication",
    "data_disposicio", "data",
    "data_de_publicacio",
]
_DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}") 
DOGC_TITLE_BEST_KEYS = [
    "t_tol_de_la_norma",
    "titol_de_la_norma",
    "títol_de_la_norma",

    "descripcio", "descripcion", "description",
    "resum", "resumen", "summary",

    "titol", "titulo", "title",
]

def _looks_like_date(s: str) -> bool:
    return bool(_DATE_RX.match(s.strip()))

def detect_order_field(rows: list[dict]) -> Optional[str]:
    """
    Busca en las primeras filas un campo que parezca fecha (YYYY-MM-DD...).
    Devuelve el nombre de columna más probable.
    """
    scores: dict[str, int] = {}
    for row in rows[:10]:
        for k, v in row.items():
            s = _as_text(v)
            if s and _looks_like_date(s):
                scores[k] = scores.get(k, 0) + 1
    if not scores:
        return None
    return max(scores, key=scores.get)

def get_row_date(row: dict, date_field: str) -> Optional[datetime]:
    s = _as_text(row.get(date_field))
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None

def build_dogc_title(row: dict) -> str:
    t = _pick_first(row, DOGC_TITLE_BEST_KEYS)
    if t:
        return t
    rang = _pick_first(row, ["rang_de_norma", "rang_norma", "tipus", "tipus_norma"]) or "Norma"
    num = _pick_first(row, ["numero_de_control", "n_mero_de_control", "num_control", "numero_control"]) or ""
    date = _pick_first(row, ["data_del_document", "data_publicacio", "data_disposicio", "data"]) or ""
    title = " ".join(x for x in [rang, num, date] if x).strip()
    return title if title else "DOGC (normativa)"

def _normalize_dogc_url(v: str) -> Optional[str]:
    v = v.strip()
    if not v:
        return None
    if v.startswith("http://") or v.startswith("https://"):
        return v
    if v.startswith("eli/") or v.startswith("/eli/"):
        return "https://portaljuridic.gencat.cat/" + v.lstrip("/")
    if v.startswith("portaljuridic.gencat.cat/") or v.startswith("dogc.gencat.cat/"):
        return "https://" + v
    if "documentId=" in v and "dogc.gencat.cat" in v:
        return "https://" + v.lstrip("/")
    return None

def _find_any_url(row: dict) -> Optional[str]:
    for k, v in row.items():
        lk = str(k).lower()
        if any(tok in lk for tok in ["url", "enlla", "enllac", "link", "eli", "uri"]):
            s = _as_text(v) 
            if s:
                u = _normalize_dogc_url(s)
                if u:
                    return u
    for v in row.values():
        s = _as_text(v)
        if s and s.startswith(("http://", "https://", "eli/", "/eli/", "portaljuridic.gencat.cat/", "dogc.gencat.cat/")):
            u = _normalize_dogc_url(s)
            if u:
                return u
    doc_id = row.get("documentId") or row.get("documentid") or row.get("id_document")
    doc_s = _as_text(doc_id)
    if doc_s and doc_s.isdigit():
        return f"https://dogc.gencat.cat/ca/document-del-dogc/?documentId={doc_s}"
    return None

def fetch_rss_xml(url: str, headers: dict, timeout: int = 15, tag: str = "rss") -> Optional[str]:
    try:
        res = requests.get(url, headers=headers, timeout=timeout)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"[{tag}] Error fetching RSS {url}: {e}")
        return None

def scrape_bme_stream() -> Iterable[Dict]:
    seen_urls: set[str] = set()

    for feed_key, feed_url in BME_RSS_FEEDS.items():
        xml = fetch_rss_xml(feed_url, headers=BME_HEADERS, timeout=15, tag="bme")
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


def fetch_dogc_resource_json(limit: int = 50, offset: int = 0, timeout: int = 25,order_by: Optional[str] = None,) -> Optional[list[dict]]:
    params = {
        "$limit": str(limit),
        "$offset": str(offset),
    }
    if order_by:
        params["$order"] = f"{order_by} DESC"
    try:
        res = requests.get(DOGC_RESOURCE_JSON, headers=DOGC_HEADERS, params=params, timeout=timeout)
        res.raise_for_status()
        data = res.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[dogc] Error fetching resource json: {e}")
        return None

def _as_text(v) -> Optional[str]:
    """Normaliza valores devueltos por Socrata: str o dict tipo URL."""
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    # Socrata URL datatype: {"url": "...", "description": "..."}
    if isinstance(v, dict):
        u = v.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip()
        # otros posibles nombres por si acaso
        href = v.get("href")
        if isinstance(href, str) and href.strip():
            return href.strip()
    return None


def _pick_first(row: dict, candidates: list[str]) -> Optional[str]:
    for k in candidates:
        s = _as_text(row.get(k))
        if s:
            return s
    return None

def scrape_dogc_stream(days_back: int = 30) -> Iterable[Dict]:
    seen_urls: set[str] = set()
    page_size = 50
    offset = 0
    pages_without_yield = 0
    max_pages_without_yield = 5
    order_field: Optional[str] = None
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    while True:
        print(f"[dogc] fetching $limit={page_size} $offset={offset} order={order_field}", flush=True)
        rows = fetch_dogc_resource_json(limit=page_size, offset=offset, order_by=order_field)
        if not rows:
            break
        if order_field is None:
            order_field = detect_order_field(rows)
            if order_field:
                rows = fetch_dogc_resource_json(limit=page_size, offset=0, order_by=order_field) or rows
                offset = 0
                print(f"[dogc] detected order_field={order_field}", flush=True)
        yielded_before_page = 0
        for row in rows:
            if order_field:
                dt = get_row_date(row, order_field)
                if dt and dt < cutoff:
                    print(f"[dogc] stopping by cutoff ({days_back}d). row_date={dt.isoformat()}", flush=True)
                    return
            url = _find_any_url(row)
            if not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if is_urls_processed_already(url):
                continue
            title = build_dogc_title(row)
            lead = _pick_first(
                row,
                [
                    "t_tol_de_la_norma",
                    "resumen",
                    "resum",
                    "summary",
                    "descripcion",
                    "descripcio",
                ],
            ) or ""
            raw = json.dumps(row, ensure_ascii=False)
            text = (lead + "\n\n" + raw).strip()
            try:
                repo.insert_link({"url": url})
            except Exception as e:
                print(f"[dogc] Warning inserting link: {e}")
            yielded_before_page += 1
            yield {
                "title": title,
                "url": url,
                "text": text,
                "source": "dogc-open-data",
                "scraped_at": datetime.now(timezone.utc),
            }
        if yielded_before_page == 0:
            pages_without_yield += 1
            print(f"[dogc] page produced 0 items ({pages_without_yield}/{max_pages_without_yield})", flush=True)
            if pages_without_yield >= max_pages_without_yield:
                print("[dogc] stopping: no new items in consecutive pages", flush=True)
                break
        else:
            pages_without_yield = 0
        offset += page_size

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
