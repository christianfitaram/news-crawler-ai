import json
import os
from typing import Any, Dict, Iterable, Optional
import hashlib
import hmac
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# Separate credentials for fetching article data and signing webhook calls.
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "eyJAdminK3y-2025!zXt9fGHEMPLq4RsVm7DwuJXeb6u")
WEBHOOK_SIGNATURE = os.getenv("WEBHOOK_SIGNATURE", NEWSAPI_KEY)

DEFAULT_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", 60))
FETCH_TIMEOUT = float(os.getenv("NEWS_FETCH_TIMEOUT", 20))


def _build_session(total_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Create a shared requests session with retry/backoff to harden network calls."""
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = _build_session()


def _validate_payload(payload: Dict[str, Any], required_fields: Iterable[str]) -> Optional[str]:
    missing = [f for f in required_fields if payload.get(f) in (None, "")]
    if missing:
        return f"Payload missing required fields {missing}; aborting webhook call."
    return None


def _log_outgoing(target_url: str, headers: Dict[str, Any], payload: Dict[str, Any]) -> None:
    try:
        redacted_headers = {**headers, "X-Signature": "***redacted***"} if "X-Signature" in headers else headers
        print(f"Sending webhook POST to: {target_url}")
        # print(f"Headers: {redacted_headers}")
        # print("Payload:", json.dumps(payload, ensure_ascii=False))
    except Exception:
        print("Payload (repr):", repr(payload))


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Optional[Dict[str, Any]]:
    _log_outgoing(url, headers, payload)
    response = SESSION.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        response.raise_for_status()
        print(f"Webhook POST succeeded: {response.status_code} Body: {response.text}")
        try:
            return response.json()
        except Exception:
            return None
    except requests.HTTPError as http_err:
        print(f"Error sending to webhook: {http_err} Status: {response.status_code} Body: {response.text}")
        return None


def send_to_webhook_to_embedding(insert_id, webhook_url=None):
    try:
        payload = get_news_data(insert_id)
        if not payload:
            print("No data found to send to webhook.")
            return None
        print(f"Fetched payload: {payload}")
        required_fields = ["article_id", "url", "title", "text", "topic", "source", "sentiment", "scraped_at"]
        validation_error = _validate_payload(payload, required_fields)
        if validation_error:
            print(validation_error)
            return None

        body = json.dumps(payload, separators=(',', ':'))
        print(f"Payload for webhook: {body}")
        signature = hmac.new(
        WEBHOOK_SIGNATURE.encode(),
        body.encode(),
        hashlib.sha256).hexdigest()
        
        headers = {
            "X-Signature": f"sha256={signature}",
            "Content-Type": "application/json"    
        }
        print(f"Headers for webhook: {headers}")
        target_url = webhook_url or os.getenv("WEBHOOK_URL", "http://localhost:8080/webhook/news")
        return _post_json(target_url, body, headers, timeout=DEFAULT_TIMEOUT)

    except requests.exceptions.RequestException as e:
        # Network-level errors, timeouts, DNS, etc.
        print(f"Error sending to webhook (network): {e}")
        return None


def send_to_all_webhooks(insert_id, webhook_url=None):
    """Send to both embedding and thread-event webhooks; returns a result map."""
    webhook_urls = webhook_url or {}
    embedding_url = webhook_urls.get("embedding") if isinstance(webhook_urls, dict) else webhook_urls
    thread_url = webhook_urls.get("thread_events") if isinstance(webhook_urls, dict) else None

    embedding_resp = send_to_webhook_to_embedding(insert_id, webhook_url=embedding_url)
    thread_resp = send_to_webhook_thread_events(insert_id, webhook_url=thread_url)

    return {
        "embedding": embedding_resp,
        "thread_events": thread_resp,
    }


def send_to_webhook_thread_events(insert_id, webhook_url=None):
    data = get_news_data(insert_id)
    if not data:
        print("No data found to send to webhooks.")
        return None
    try:
        required_fields = ["article_id", "source", "scraped_at"]
        validation_error = _validate_payload(data, required_fields)
        if validation_error:
            print(validation_error)
            return None
        payload = {
            "article_id": data.get("article_id"),
            "source": data.get("source"),
            "scraped_at": data.get("scraped_at"),
        }
        target_url = webhook_url or os.getenv("WEBHOOK_URL_THREAD_EVENTS",
                                              "http://localhost:8000/webhooks/article-vectorized")
        headers = {"Content-Type": "application/json"}
        return _post_json(target_url, payload, headers, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.RequestException as e:
        # Network-level errors, timeouts, DNS, etc.
        print(f"Error sending to webhook (network): {e}")
        return None


def get_news_data(insert_id: str, timeout: float = FETCH_TIMEOUT) -> Optional[Dict[str, Any]]:
    base_url = f"https://newsapi.one/v1/news/{insert_id}?apiKey={NEWSAPI_KEY}"
    try:
        response = SESSION.get(base_url, timeout=timeout)
        response.raise_for_status()
        data_raw = response.json()
        data = data_raw.get("data", {})
        if not data:
            print(f"No news data found for ID: {insert_id}")
            return None
        data_to_return: Dict[str, Any] = {
            "article_id": data.get("id"),
            "url": data.get("url"),
            "title": data.get("title"),
            "text": data.get("text"),
            "topic": data.get("topic"),
            "source": data.get("source"),
            "sentiment": data.get("sentiment"),
            "scraped_at": data.get("scraped_at"),
        }
        return data_to_return
    except (requests.exceptions.RequestException, ValueError) as e:
        # ValueError catches JSON decode errors
        print(f"Error fetching news data: {e}")
        return None
