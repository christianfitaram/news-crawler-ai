#!/usr/bin/env python3
"""
selenium_dw_extract_links.py

- Requires: selenium
- Purpose: Open DW top-stories page, dismiss cookie modal, extract article links.
"""

import os
import shutil
import time
import re
import tempfile

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
)
from selenium.webdriver.support.ui import WebDriverWait

from webdriver_manager.chrome import ChromeDriverManager


DW_URL = "https://www.dw.com/en/top-stories/s-9097"

# ----------------------------
# Link patterns
# ----------------------------
LINK_PATTERNS = [
    r"^https?://(www\.)?dw\.com/en/.+/(a|video|g)-\d+",
    r"^https?://(www\.)?dw\.com/en/.+/(a|video|g)-[A-Za-z0-9\-]+",
    r"^/en/.+/(a|video|g)-\d+",
    r"^/en/.+/(a|video|g)-[A-Za-z0-9\-]+",
]

# ----------------------------
# Cookie handling
# ----------------------------
ACCEPT_TEXTS = [
    "accept", "accept all", "agree", "ok", "allow",
    "aceptar", "aceitar",
    "zustimmen", "akzeptieren", "accepter",
]

COOKIE_BUTTON_XPATHS = [
    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
    "//button[contains(@class,'cookie') or contains(@id,'cookie')]",
]

CHROME_BINARY_ENV = "CHROME_BINARY"
CHROMEDRIVER_ENV = "CHROMEDRIVER_PATH"


# ----------------------------
# Chrome resolution
# ----------------------------
def _resolve_chrome_binary():
    env_path = os.getenv(CHROME_BINARY_ENV)
    if env_path and os.path.exists(env_path):
        print(f"[chrome] Using env CHROME_BINARY={env_path}")
        return env_path

    candidates = [
        "/snap/bin/chromium",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            print(f"[chrome] Using detected browser binary: {c}")
            return c

    raise RuntimeError("Chrome/Chromium binary not found")


def _resolve_chromedriver_path():
    env_path = os.getenv(CHROMEDRIVER_ENV)
    if env_path and os.path.exists(env_path):
        print(f"[chromedriver] Using env CHROMEDRIVER_PATH={env_path}")
        return env_path

    for c in ["/usr/bin/chromedriver", shutil.which("chromedriver")]:
        if c and os.path.exists(c):
            print(f"[chromedriver] Using system chromedriver at: {c}")
            return c

    print("[chromedriver] Falling back to webdriver-manager")
    return ChromeDriverManager().install()


# ----------------------------
# Profile dir (snap-safe)
# ----------------------------
def _make_profile_dir():
    base = os.path.expanduser("~/snap/chromium/common/selenium-profiles")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="dw-", dir=base)


# ----------------------------
# Driver builder (FIXED)
# ----------------------------
def build_driver(headless=True):
    options = webdriver.ChromeOptions()
    options.binary_location = _resolve_chrome_binary()

    if headless:
        options.add_argument("--headless=new")
    else:
        options.add_argument("--start-maximized")

    # REQUIRED for servers / snap
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--window-size=1920,1080")

    # ❌ DO NOT USE --single-process (causes DevToolsActivePort crashes)

    profile_dir = _make_profile_dir()
    options.add_argument(f"--user-data-dir={profile_dir}")

    options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143 Safari/537.36"
    )

    driver_path = _resolve_chromedriver_path()
    print(
        f"[driver] Starting Chrome with binary={options.binary_location} "
        f"chromedriver={driver_path} profile={profile_dir}"
    )

    try:
        driver = webdriver.Chrome(service=Service(driver_path), options=options)
        driver.set_page_load_timeout(30)
        driver.set_script_timeout(30)
        return driver, profile_dir
    except Exception:
        shutil.rmtree(profile_dir, ignore_errors=True)
        raise


# ----------------------------
# Cookie dismissal
# ----------------------------
def try_click(el):
    try:
        el.click()
        return True
    except (ElementClickInterceptedException, StaleElementReferenceException):
        try:
            el._parent.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            return False


def dismiss_cookie_modal(driver):
    time.sleep(1)
    for xpath in COOKIE_BUTTON_XPATHS:
        for el in driver.find_elements(By.XPATH, xpath):
            txt = (el.text or "").lower()
            if any(k in txt for k in ACCEPT_TEXTS) and el.is_displayed():
                if try_click(el):
                    time.sleep(0.5)
                    return True
    return False


# ----------------------------
# Link extraction
# ----------------------------
def extract_links_from_page(driver):
    links = set()
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = a.get_attribute("href")
        if not href:
            continue
        href = href.strip()
        if href.startswith("/"):
            href = "https://www.dw.com" + href
        for pat in LINK_PATTERNS:
            if re.search(pat, href):
                links.add(href)
                break
    return sorted(links)


# ----------------------------
# Main
# ----------------------------
def main(headless=True):
    try:
        driver, profile_dir = build_driver(headless=headless)
    except Exception as e:
        print(f"[crawler_dw][fatal] Cannot start Selenium driver: {e}")
        return []

    try:
        print("[*] Opening:", DW_URL)
        driver.get(DW_URL)
        time.sleep(2)

        print("[*] Dismissing cookie modal...")
        dismiss_cookie_modal(driver)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        links = extract_links_from_page(driver)
        print(f"[*] Found {len(links)} links")

        for l in links[:30]:
            print(l)

        return links

    finally:
        try:
            driver.quit()
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)


if __name__ == "__main__":
    main(headless=True)
