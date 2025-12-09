# Selenium Configuration and Troubleshooting

This guide shows how to configure Selenium locally and fix common path-related errors when running scrapers or smoke tests.

## Installation
- Install dependencies: `pip install selenium webdriver-manager` (the helper auto-downloads matching drivers).
- Ensure a supported browser is present (Chrome, Firefox, or Edge). On macOS with Homebrew: `brew install --cask google-chrome`.

## Driver setup
- Preferred: rely on `webdriver-manager` to avoid manual driver downloads. Example for Chrome:
  ```python
  from selenium import webdriver
  from selenium.webdriver.chrome.service import Service
  from selenium.webdriver.chrome.options import Options
  from webdriver_manager.chrome import ChromeDriverManager

  opts = Options()
  opts.add_argument("--headless=new")  # drop if you need a visible browser

  driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
  driver.get("https://example.com")
  print(driver.title)
  driver.quit()
  ```
- Manual driver path: download the driver that matches your browser version, make it executable, and set an env var in `.env`:
  ```bash
  CHROMEDRIVER_PATH=/usr/local/bin/chromedriver
  GECKODRIVER_PATH=/usr/local/bin/geckodriver
  EDGE_DRIVER_PATH=/usr/local/bin/msedgedriver
  ```
  Then use `Service(os.getenv("CHROMEDRIVER_PATH"))` when constructing the driver.

## Common path errors and fixes
- "`chromedriver` executable needs to be in PATH": add the driver directory to `PATH` (`export PATH="$PATH:/usr/local/bin"`) or provide the full path via `Service("/full/path/chromedriver")`.
- "cannot find Chrome binary": install the browser or point Selenium to it (`opts.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`).
- Permission denied: ensure the driver is executable (`chmod +x /path/to/chromedriver`).
- Version mismatch: use `webdriver-manager` so the driver version matches the installed browser; alternatively download the correct version manually.
- Headless issues in CI: add `--no-sandbox --disable-dev-shm-usage` to Chrome options to avoid sandbox/share-memory errors.

## Quick sanity check
- Verify paths: `which chromedriver` or `ls -l /usr/local/bin/chromedriver`.
- Minimal run to confirm everything is wired:
  ```bash
  python - <<'PY'
  from selenium import webdriver
  from selenium.webdriver.chrome.service import Service
  from selenium.webdriver.chrome.options import Options
  from webdriver_manager.chrome import ChromeDriverManager

  opts = Options(); opts.add_argument("--headless=new")
  driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
  driver.get("https://example.com")
  print(driver.title)
  driver.quit()
  PY
  ```

Keep these snippets close to your scraper so misconfigurations are caught early during local runs.
