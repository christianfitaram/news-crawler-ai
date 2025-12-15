## Selenium ChromeDriver Version Mismatch (Chromium 143 vs. Driver 114)

**Symptom**
- Service logs show: `session not created: This version of ChromeDriver only supports Chrome version 114`
- Current browser version: `Chromium 143.0.7499.40` (`/snap/bin/chromium`)
- Follow-up errors like `DevToolsActivePort file doesn't exist` after driver mismatch.

**Root Cause**
- ChromeDriver bundled/installed (114) does not match the running Chromium version (143), so Selenium cannot start a session.

**Resolution**
1) Install unzip (if missing):
   ```bash
   sudo apt-get update && sudo apt-get install -y unzip
   ```
2) Download the matching driver (use the exact browser version):
   ```bash
   VER=143.0.7499.40
   URL=https://storage.googleapis.com/chrome-for-testing-public/${VER}/linux64/chromedriver-linux64.zip
   wget -O /tmp/chromedriver.zip "$URL"
   unzip -o /tmp/chromedriver.zip -d /tmp/chromedriver-${VER}
   sudo mv /tmp/chromedriver-${VER}/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver-${VER}
   sudo chmod +x /usr/local/bin/chromedriver-${VER}
   ```
3) Point Selenium to the correct binaries (in systemd unit or shell):
   ```bash
   export CHROME_BIN=/snap/bin/chromium
   export CHROMEDRIVER_PATH=/usr/local/bin/chromedriver-143.0.7499.40
   ```
   If using systemd, add to the service unit:
   ```
   Environment=CHROME_BIN=/snap/bin/chromium
   Environment=CHROMEDRIVER_PATH=/usr/local/bin/chromedriver-143.0.7499.40
   ```
4) Reload and restart the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart news-crawler-ai.service
   ```

**Notes**
- Always match ChromeDriver to the exact browser version. For snap-based Chromium, use the version from `/snap/bin/chromium --version` and download the corresponding driver from `chrome-for-testing-public`.
- If using `webdriver_manager`, pin a matching driver or supply `CHROMEDRIVER_PATH` to bypass automatic downloads.
