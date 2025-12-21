from __future__ import annotations

from typing import Optional, Dict, Any, List
import os
import json
import requests


# -----------------------------
# Helpers
# -----------------------------
def _normalize_entity_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("'s", "")
    key = key.replace('"', "").replace("'", "")
    key = key.replace(",", "")
    key = " ".join(key.split())
    return key


def normalize_entity_lists(entities: dict) -> dict:
    """Normalize entity lists into [{name, key}, ...] with case-insensitive dedupe.

    Accepts either:
      - list[str]
      - list[dict] with "name" and optional "key"
    """
    normalized: Dict[str, List[Dict[str, str]]] = {}
    for field in ("locations", "organizations", "persons"):
        values = entities.get(field, [])
        items: List[Dict[str, str]] = []
        seen_keys = set()

        if not isinstance(values, list):
            normalized[field] = items
            continue

        for value in values:
            if isinstance(value, dict):
                name = str(value.get("name", "")).strip()
                if not name:
                    continue
                key = str(value.get("key", "")).strip() or _normalize_entity_key(name)
            else:
                name = str(value).strip()
                if not name:
                    continue
                key = _normalize_entity_key(name)

            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            items.append({"name": name, "key": key})

        normalized[field] = items
    return normalized


def _parse_gpt_json(raw_response: Any) -> Optional[dict]:
    """Parse a JSON object from either a dict response or a raw string.

    Ollama sometimes returns extra text; we try to locate the outermost JSON object.
    """
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        return None

    raw_text = raw_response.strip()
    if not raw_text:
        return None

    # First, try direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...last} block
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _ollama_generate(
    prompt: str,
    *,
    api_url: str,
    model: str,
    timeout: int,
    options: Optional[dict] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    # Ollama /api/generate uses {response: "..."}
    # but some proxies return already-parsed JSON or other shapes.
    return data.get("response", "") if isinstance(data, dict) else ""


# -----------------------------
# 2-pass pipeline
# -----------------------------
def _build_cleaner_prompt(raw_text: str) -> str:
    return f"""You are a STRICT text cleaner.

Output MUST be ONLY valid JSON.
Return a single JSON object with EXACTLY this key:
{{
  \"cleaned_text\": string
}}

CLEANING RULES (DELETE-ONLY):
- You may ONLY remove content; do NOT paraphrase, summarize, translate, or reorder.
- If a sentence (or clause) is kept, it MUST be copied VERBATIM from the input (same words, same order).
- Allowed micro-edits ONLY:
  * remove URLs, navigation/cookie text, share/subscribe prompts, bylines, author blocks, publisher/outlet mentions
  * remove image captions/layout fragments
  * remove duplicated lines/fragments
  * fix line-break hyphenation caused by wraps (e.g., "inter-\nference" -> "interference") and normalize whitespace
- Discard malformed/incomplete fragments.

Output requirements:
- cleaned_text must be composed only of verbatim kept text from the input.
- Do not add any words not present in the input.

Input text:
<<<
{raw_text}
>>>
""".strip()


def _build_extractor_prompt(cleaned_text: str) -> str:
    return f"""You are a STRICT named-entity extraction engine.

Output MUST be ONLY valid JSON.
Return a single JSON object with EXACTLY these keys:
{{
  \"locations\": string[],
  \"organizations\": string[],
  \"persons\": string[]
}}

EXTRACTION RULES (HIGH RECALL, NO INFERENCE):
- Extract entities ONLY from the provided text (below).
- Every entity MUST appear EXACTLY as a substring in the text (verbatim match).
- Do NOT infer, normalize, expand, or correct names.
- Include all explicitly-mentioned proper names for people, organizations, and locations.
- Do NOT include generic groups (e.g., "police", "protesters") unless a proper name is explicitly present.
- Prefer full names as written (e.g., "Fedor Gorst" not "Gorst" if both appear).
- Deduplicate case-insensitively while preserving the first-seen original casing.
- If no entities exist for a category, return [].

Self-check BEFORE output:
- For each item in locations/organizations/persons, confirm it appears verbatim in the text.

Text:
<<<
{cleaned_text}
>>>
""".strip()


def call_to_gpt_api(
    prompt: str,
    timeout: int = 120,
    *,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """2-pass: (1) delete-only cleaning, (2) entity extraction from cleaned text.

    Returns:
      {
        "cleaned_text": str,
        "locations": [{name,key},...],
        "organizations": [{name,key},...],
        "persons": [{name,key},...],
      }
    """
    api_url = api_url or os.getenv("OLLAMA_GENERATE_URL") or "http://localhost:11434/api/generate"
    model = model or os.getenv("OLLAMA_MODEL") or "gpt-oss:20b"

    # Conservative decoding to reduce paraphrase drift
    options = {
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.9")),
    }

    # -----------------
    # Pass 1: cleaning
    # -----------------
    cleaned_text = prompt
    try:
        cleaner_prompt = _build_cleaner_prompt(prompt)
        raw = _ollama_generate(cleaner_prompt, api_url=api_url, model=model, timeout=timeout, options=options)
        parsed = _parse_gpt_json(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("cleaned_text"), str):
            cleaned_text = parsed["cleaned_text"].strip() or prompt
    except requests.exceptions.Timeout:
        print(f"GPT/Ollama cleaner timeout after {timeout}s; using original text")
        cleaned_text = prompt
    except requests.exceptions.RequestException as e:
        print(f"GPT/Ollama cleaner error: {e}; using original text")
        cleaned_text = prompt
    except Exception as e:
        print(f"Cleaner unexpected error: {e}; using original text")
        cleaned_text = prompt

    # -----------------
    # Pass 2: extraction
    # -----------------
    try:
        extractor_prompt = _build_extractor_prompt(cleaned_text)
        raw = _ollama_generate(extractor_prompt, api_url=api_url, model=model, timeout=timeout, options=options)
        parsed = _parse_gpt_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Extractor did not return JSON object")

        entities = normalize_entity_lists(parsed)
        return {
            "cleaned_text": cleaned_text,
            "locations": entities.get("locations", []),
            "organizations": entities.get("organizations", []),
            "persons": entities.get("persons", []),
        }
    except requests.exceptions.Timeout:
        print(f"GPT/Ollama extractor timeout after {timeout}s; returning empty entities")
    except requests.exceptions.RequestException as e:
        print(f"GPT/Ollama extractor error: {e}; returning empty entities")
    except Exception as e:
        print(f"Extractor unexpected error: {e}; returning empty entities")

    return {
        "cleaned_text": cleaned_text,
        "locations": [],
        "organizations": [],
        "persons": [],
    }


if __name__ == "__main__":
    sample = "Deal to avoid drug tariffs has no un...ck. Several officials said those meetings were ‘very positive’"
    print(call_to_gpt_api(sample, timeout=60))
