#!/usr/bin/env python3
"""
selenium_dw_extract_links.py

- Requires: selenium, webdriver-manager
- Purpose: Open DW top-stories page, dismiss cookie modal, extract article links, save CSV/JSON.
"""

import os
import os
import shutil
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


DW_URL = "https://www.dw.com/en/top-stories/s-9097"


# Patterns to match article links we care about (DW english articles/videos/dossiers)
LINK_PATTERNS = [
    r"^https?://(www\.)?dw\.com/en/.+/(a|video|g)-\d+",
    r"^https?://(www\.)?dw\.com/en/.+/(a|video|g)-[A-Za-z0-9\-]+",  # catch other forms
    r"^/en/.+/(a|video|g)-\d+",
    r"^/en/.+/(a|video|g)-[A-Za-z0-9\-]+",
]

# Some localized accept texts to try (lowercase comparisons)
ACCEPT_TEXTS = [
    "accept", "accept all", "i accept", "i agree", "agree", "ok", "accept cookies","Yes,I Accept",
    "allow", "allow all",
    # Spanish / Portuguese
    "aceptar", "acepto", "aceitar",
    # German / French
    "zustimmen", "einverstanden", "akzeptieren", "accepter",
    # short variants
    "cookie settings", "cookies", "allow all",
]

# A more permissive CSS/XPath selection that often contains cookie accept buttons
COOKIE_BUTTON_XPATHS = [
    # typical buttons with text
    "//button[normalize-space()!='' and (contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'acept') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'aceitar') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'zustimmen') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'akzeptier') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accepter') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ok') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow') )]",
    # look for inputs
    "//input[@type='button' or @type='submit' and (contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') or contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree'))]",
    # aria labelled cookie/consent
    "//button[contains(@aria-label, 'cookie') or contains(@aria-label, 'consent') or contains(@title, 'cookie') or contains(@title, 'consent')]",
    # generic selectors
    "//button[contains(@class,'cookie') or contains(@class,'consent') or contains(@class,'accept') or contains(@id,'cookie') or contains(@id,'consent')]",
]


CHROME_BINARY_ENV = "CHROME_BINARY"
CHROMEDRIVER_ENV = "CHROMEDRIVER_PATH"


def _resolve_chrome_binary():
    """Return a Chrome binary path, honoring overrides for custom installations."""
    env_path = os.getenv(CHROME_BINARY_ENV)
    if env_path:
        env_path = env_path.strip()
        if os.path.exists(env_path):
            return env_path

    candidates = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium-browser"),
        shutil.which("chromium"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise RuntimeError(
        "Chrome/Chromium binary not found. Install google-chrome or chromium, "
        f"or set {CHROME_BINARY_ENV}=/full/path/to/chrome before running this crawler."
    )


def _resolve_chromedriver_path():
    """Return a chromedriver path, enabling offline reuse via an env override."""
    env_path = os.getenv(CHROMEDRIVER_ENV)
    if env_path:
        env_path = env_path.strip()
        if os.path.exists(env_path):
            return env_path
        raise RuntimeError(
            f"{CHROMEDRIVER_ENV} is set to '{env_path}' but the file does not exist."
        )
    # webdriver-manager caches drivers locally, so calling install is safe
    return ChromeDriverManager().install()

def build_driver(headless=True):
    options = webdriver.ChromeOptions()
    options.binary_location = _resolve_chrome_binary()
    if headless:
        # use new-headless mode when available
        options.add_argument("--headless=new")
    else:
        options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--single-process")
    # optional: mimic regular user agent
    options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
    driver_path = _resolve_chromedriver_path()
    driver = webdriver.Chrome(service=Service(driver_path), options=options)
    return driver

def try_click(element):
    try:
        element.click()
        return True
    except (ElementClickInterceptedException, StaleElementReferenceException):
        try:
            # fallback JS click
            element._parent.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False

def dismiss_cookie_modal(driver, timeout=10):
    wait = WebDriverWait(driver, timeout)
    # 1) Wait a bit for any modal to appear
    time.sleep(1.0)
    # 2) Try iframe-based cookie banners (common pattern)
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    for idx, iframe in enumerate(iframes):
        try:
            # only attempt if iframe is visible & likely related
            src = iframe.get_attribute("src") or ""
            title = iframe.get_attribute("title") or ""
            # Heuristics: cookie / consent / cookiebot / usercentrics often in src/title
            if any(k in src.lower() for k in ("cookie", "consent", "usercentrics", "cookiehub", "optanon")) or any(k in title.lower() for k in ("cookie", "consent")):
                try:
                    driver.switch_to.frame(iframe)
                    # try xpath buttons inside this frame
                    for xpath in COOKIE_BUTTON_XPATHS:
                        elems = driver.find_elements(By.XPATH, xpath)
                        for el in elems:
                            txt = (el.text or el.get_attribute("value") or "").strip().lower()
                            if any(tok in txt for tok in ACCEPT_TEXTS) or True:
                                if try_click(el):
                                    time.sleep(0.6)
                                    driver.switch_to.default_content()
                                    return True
                    driver.switch_to.default_content()
                except Exception:
                    driver.switch_to.default_content()
                    continue
        except Exception:
            continue

    # 3) Try common cookie banners in main DOM
    for xpath in COOKIE_BUTTON_XPATHS:
        try:
            candidates = driver.find_elements(By.XPATH, xpath)
            for el in candidates:
                txt = (el.text or el.get_attribute("value") or el.get_attribute("aria-label") or "").strip().lower()
                # click if text matches our accept words, or as fallback click the first visible candidate
                if txt:
                    if any(tok in txt for tok in ACCEPT_TEXTS):
                        if try_click(el):
                            time.sleep(0.6)
                            return True
                else:
                    # fallback: click if visible
                    if el.is_displayed():
                        if try_click(el):
                            time.sleep(0.6)
                            return True
        except Exception:
            continue

    # 4) Try dismissing banners via JS: remove elements that look like cookie banners
    try:
        script = r'''
        (() => {
          const keywords = ['cookie', 'consent', 'eu-cookie', 'cookie-banner', 'usercentrics', 'onetrust', 'optanon', 'cookiebot', 'cookieconsent'];
          let removed = 0;
          for (const el of Array.from(document.querySelectorAll('div,section,aside,dialog'))) {
            const cls = (el.className || '').toLowerCase();
            const id = (el.id || '').toLowerCase();
            const text = (el.innerText || '').toLowerCase().slice(0, 400);
            if (keywords.some(k => cls.includes(k) || id.includes(k) || text.includes(k))) {
              el.style.display = 'none';
              removed += 1;
            }
          }
          return removed;
        })();
        '''
        removed = driver.execute_script(script)
        if removed and int(removed) > 0:
            # small wait and return true (we removed banner)
            time.sleep(0.4)
            return True
    except Exception:
        pass

    # 5) If nothing worked, return False
    return False

def extract_links_from_page(driver):
    anchors = driver.find_elements(By.TAG_NAME, "a")
    links = set()
    for a in anchors:
        try:
            href = a.get_attribute("href")
            if not href:
                continue
            href = href.strip()
            # normalize relative links that start with /en/...
            if href.startswith("/"):
                href = "https://www.dw.com" + href
            # filter via regex patterns
            for pat in LINK_PATTERNS:
                if re.search(pat, href):
                    links.add(href)
                    break
        except Exception:
            continue
    # Also attempt to find links inside data attributes or JSON-LD (fallback)
    # (optional: not implemented here)
    return sorted(links)

def main(headless=True):
    driver = build_driver(headless=headless)
    try:
        print("[*] opening page:", DW_URL)
        driver.get(DW_URL)
        # let page load and JS run
        time.sleep(2.0)

        print("[*] attempting to dismiss cookie modal...")
        ok = dismiss_cookie_modal(driver, timeout=8)
        print("[*] cookie modal dismissed?" , ok)

        # Additional wait for dynamic content to fully load after dismissal
        time.sleep(1.0)

        # maybe load more content by scrolling
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        time.sleep(0.6)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.6)

        # extract
        links = extract_links_from_page(driver)
        print(f"[*] found {len(links)} matching links.")
        print(links)
        # Print sample
        for i, l in enumerate(links):
            if i >= 50:
                break
            print(l)
        return list(links)

    finally:
        driver.quit()


if __name__ == "__main__":
    # change headless=False for debugging with a browser window
    main(headless=True)
