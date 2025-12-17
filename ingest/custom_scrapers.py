# pipeline_sample/custom_scrapers.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Iterable, cast

import feedparser
import requests
from bs4 import BeautifulSoup

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


def build_chrome_driver(headless: bool = True):
    """Create a Chrome driver with safe defaults for headless/systemd environments."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    chrome_options = Options()
    if headless:
        # "new" flag for modern versions; older Chrome ignores it.
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    # Persist user data to a writable location to avoid DevTools port issues.
    user_data_dir = os.getenv("CHROME_USER_DATA_DIR", "/tmp/chrome-user-data")
    chrome_options.add_argument(f"--user-data-dir={user_data_dir}")

    # Allow overriding the Chrome binary, driver path, or version via env.
    chrome_bin = os.getenv("CHROME_BIN")
    if chrome_bin:
        chrome_options.binary_location = chrome_bin

    driver_path = os.getenv("CHROMEDRIVER_PATH")
    if driver_path:
        service = Service(driver_path)
    else:
        driver_version = os.getenv("CHROMEDRIVER_VERSION")
        try:
            service = Service(ChromeDriverManager(version=driver_version).install())
        except TypeError:
            # Older webdriver_manager versions don't accept 'version'; ignore it.
            service = Service(ChromeDriverManager().install())

    return webdriver.Chrome(service=service, options=chrome_options)


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

def scrape_guardian_stream() -> Iterable[Dict]:
    """Scrape The Guardian using RSS feeds"""
    GUARDIAN_RSS_FEEDS = {
        "world": "https://www.theguardian.com/world/rss",
        "business": "https://www.theguardian.com/business/rss",
        "technology": "https://www.theguardian.com/technology/rss",
        "science": "https://www.theguardian.com/science/rss",
        "environment": "https://www.theguardian.com/environment/rss",
        "politics": "https://www.theguardian.com/politics/rss",
    }
    
    for category, rss_url in GUARDIAN_RSS_FEEDS.items():
        try:
            feed = feedparser.parse(rss_url)
        except Exception as e:
            print(f"Error parsing Guardian RSS feed ({category}): {e}")
            continue
        
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
                "source": "the-guardian",
                "scraped_at": datetime.now(timezone.utc),
            }


def scrape_reuters_stream() -> Iterable[Dict]:
    """Scrape Reuters using RSS feeds"""
    REUTERS_RSS_FEEDS = {
        "world": "https://www.reuters.com/rssFeed/worldNews",
        "business": "https://www.reuters.com/rssFeed/businessNews",
        "technology": "https://www.reuters.com/rssFeed/technologyNews",
        "sports": "https://www.reuters.com/rssFeed/sportsNews",
        "entertainment": "https://www.reuters.com/rssFeed/entertainmentNews",
    }
    
    for category, rss_url in REUTERS_RSS_FEEDS.items():
        try:
            feed = feedparser.parse(rss_url)
        except Exception as e:
            print(f"Error parsing Reuters RSS feed ({category}): {e}")
            continue
        
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
                "source": "reuters",
                "scraped_at": datetime.now(timezone.utc),
            }
def scrape_guardian_selenium_stream() -> Iterable[Dict]:
    """Scrape The Guardian using Selenium for dynamic content"""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    
    GUARDIAN_SECTIONS = {
        "world": "https://www.theguardian.com/world",
        "business": "https://www.theguardian.com/business",
        "technology": "https://www.theguardian.com/technology",
        "science": "https://www.theguardian.com/science",
        "environment": "https://www.theguardian.com/environment",
    }
    
    # Inicializar el driver
    try:
        driver = build_chrome_driver(headless=True)
    except Exception as e:
        print(f"Error initializing Selenium: {e}")
        return
    
    try:
        for category, section_url in GUARDIAN_SECTIONS.items():
            print(f"\n Scraping The Guardian - {category}...")
            
            try:
                # Cargar la página
                driver.get(section_url)
                time.sleep(3)  # Esperar a que cargue el contenido
                
                # Buscar enlaces de artículos
                # The Guardian usa diferentes selectores, probamos varios
                article_links = driver.find_elements(By.CSS_SELECTOR, "a[data-link-name='article']")
                
                # Si no encuentra con ese selector, intenta otro
                if not article_links:
                    article_links = driver.find_elements(By.CSS_SELECTOR, "div.fc-item__container a")
                
                print(f"   Found {len(article_links)} articles in {category}")
                
                # Procesar los primeros 10 artículos
                processed_urls = set()
                
                for link_element in article_links[:10]:
                    try:
                        article_url = link_element.get_attribute("href")
                        
                        # Evitar duplicados
                        if not article_url or article_url in processed_urls:
                            continue
                        
                        # Filtrar URLs que no sean artículos
                        if not article_url or "theguardian.com" not in article_url:
                            continue
                        
                        processed_urls.add(article_url)
                        
                        # Verificar si ya fue procesado en MongoDB
                        if is_urls_processed_already(article_url):
                            continue
                        
                        # Extraer contenido con trafilatura
                        text = fetch_and_extract(article_url)
                        
                        if not text or len(text) < 100:
                            print(f" Content insufficient: {article_url[:60]}...")
                            continue
                        
                        # Extract title from element or URL
                        try:
                            title = link_element.text.strip()
                            if not title:
                                # Intentar obtener de un hijo
                                title_element = link_element.find_element(By.CSS_SELECTOR, "span")
                                title = title_element.text.strip()
                        except:
                            # Usar parte de la URL como título
                            title = article_url.split("/")[-1].replace("-", " ").title()
                        
                        if not title:
                            title = "The Guardian Article"
                        
                        # Guardar en link_pool
                        repo.insert_link({"url": article_url})
                        
                        print(f"   {title[:60]}...")
                        
                        yield {
                            "title": title,
                            "url": article_url,
                            "text": text,
                            "source": "the-guardian",
                            "scraped_at": datetime.now(timezone.utc),
                        }
                        
                    except Exception as e:
                        print(f"    Error processing article: {e}")
                        continue
                
                time.sleep(2)  # Pause between categories
                
            except Exception as e:
                print(f"Error scrapping category {category}: {e}")
                continue
                
    finally:
        driver.quit()

def scrape_france24_selenium_stream() -> Iterable[Dict]:
    """Scrape France24 using Selenium"""
    from selenium.webdriver.common.by import By
    import time
    
    FRANCE24_SECTIONS = {
        "world": "https://www.france24.com/en/",
        "europe": "https://www.france24.com/en/europe/",
        "americas": "https://www.france24.com/en/americas/",
        "middle-east": "https://www.france24.com/en/middle-east/",
        "africa": "https://www.france24.com/en/africa/",
        "asia-pacific": "https://www.france24.com/en/asia-pacific/",
    }
    
    try:
        driver = build_chrome_driver(headless=True)
    except Exception as e:
        print(f"Error initializing Selenium: {e}")
        return
    
    try:
        for category, section_url in FRANCE24_SECTIONS.items():
            print(f"\n Scrapping France24 - {category}...")
            driver.get(section_url)
            time.sleep(3)
            
            # France24 usa estos selectores
            article_links = driver.find_elements(By.CSS_SELECTOR, "article a.article__title-link")
            if not article_links:
                article_links = driver.find_elements(By.CSS_SELECTOR, ".m-item-list-article a")
            if not article_links:
                article_links = driver.find_elements(By.CSS_SELECTOR, "h2 a")
            
            print(f"   Found {len(article_links)} articles in {category}")
            
            processed_urls = set()
            
            for link_element in article_links[:10]:
                try:
                    article_url = link_element.get_attribute("href")
                    
                    if not article_url or article_url in processed_urls:
                        continue
                    
                    # Construir URL completa si es relativa
                    if article_url.startswith("/"):
                        article_url = "https://www.france24.com" + article_url
                    
                    if "france24.com" not in article_url:
                        continue
                    
                    processed_urls.add(article_url)
                    
                    if is_urls_processed_already(article_url):
                        continue
                    
                    text = fetch_and_extract(article_url)
                    if not text or len(text) < 100:
                        print(f" Content insufficient: {article_url[:60]}...")
                        continue
                    
                    try:
                        title = link_element.text.strip()
                        if not title:
                            title = article_url.split("/")[-1].replace("-", " ").title()
                    except:
                        title = "France24 Article"
                    
                    if not title or len(title) < 10:
                        continue
                    
                    repo.insert_link({"url": article_url})
                    
                    print(f" {title[:60]}...")
                    
                    yield {
                        "title": title,
                        "url": article_url,
                        "text": text,
                        "source": "france24",
                        "scraped_at": datetime.now(timezone.utc),
                    }
                    
                except Exception as e:
                    print(f"  Error processing article: {e}")
                    continue
            
            time.sleep(2)
                
    finally:
        driver.quit()

def scrape_npr_selenium_stream() -> Iterable[Dict]:
    """Scrape NPR using Selenium"""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    NPR_SECTIONS = {
        "world": "https://www.npr.org/sections/world/",
        "business": "https://www.npr.org/sections/business/",
        "technology": "https://www.npr.org/sections/technology/",
    }
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    try:
        for category, section_url in NPR_SECTIONS.items():
            print(f"\n📻 Scrapping NPR - {category}...")
            driver.get(section_url)
            time.sleep(3)
            
            article_links = driver.find_elements(By.CSS_SELECTOR, "h2.title a")
            if not article_links:
                article_links = driver.find_elements(By.CSS_SELECTOR, "article h3 a")
            
            print(f"   Found {len(article_links)} articles in {category}")
            
            processed_urls = set()
            
            for link_element in article_links[:10]:
                try:
                    article_url = link_element.get_attribute("href")
                    
                    if not article_url or article_url in processed_urls:
                        continue
                    if "npr.org" not in article_url:
                        continue
                    
                    processed_urls.add(article_url)
                    
                    if is_urls_processed_already(article_url):
                        continue
                    
                    text = fetch_and_extract(article_url)
                    if not text or len(text) < 100:
                        continue
                    
                    try:
                        title = link_element.text.strip()
                        if not title:
                            title = article_url.split("/")[-2].replace("-", " ").title()
                    except:
                        title = "NPR Article"
                    
                    if not title or len(title) < 10:
                        continue
                    
                    repo.insert_link({"url": article_url})
                    
                    print(f"  {title[:60]}...")
                    
                    yield {
                        "title": title,
                        "url": article_url,
                        "text": text,
                        "source": "npr",
                        "scraped_at": datetime.now(timezone.utc),
                    }
                    
                except Exception as e:
                    print(f"   Error procesando artículo: {e}")
                    continue
            
            time.sleep(2)
                
    finally:
        driver.quit()
