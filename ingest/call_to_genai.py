from __future__ import annotations

from typing import Optional, Dict, Any, List, Iterable
import os
import json
import re
import requests
from google import genai
from google.genai import types


# ==================================================
# Gemini client
# ==================================================


class GeminiClient:
    """Manages a single instance of the GenAI client."""
    _instance: Optional[genai.Client] = None

    @classmethod
    def get_client(cls) -> genai.Client:
        if cls._instance is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY")
            
            # Initialize the client once
            cls._instance = genai.Client(api_key=api_key)
        return cls._instance


# -----------------------------
# Helpers
# -----------------------------
def _normalize_entity_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("'s", "")
    key = key.replace('"', "").replace("'", "")
    key = re.sub(r"\s+", " ", key)
    return key


def _dedupe_preserve_first(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        vv = v.strip()
        if not vv:
            continue
        k = _normalize_entity_key(vv)
        if k in seen:
            continue
        seen.add(k)
        out.append(vv)
    return out


def normalize_entity_lists(obj: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Strict normalizer: expects exact keys locations/organizations/persons.
    """
    out: Dict[str, List[str]] = {}
    for k in ("locations", "organizations", "persons"):
        v = obj.get(k, [])
        if v is None:
            v = []
        if not isinstance(v, list):
            v = []
        out[k] = _dedupe_preserve_first([x for x in v if isinstance(x, str)])
    return out


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Chunk text by paragraph boundaries when possible, otherwise hard-split.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append("\n\n".join(cur).strip())
            cur = []
            cur_len = 0

    for p in paras:
        add_len = len(p) + (2 if cur else 0)
        if cur_len + add_len <= max_chars:
            cur.append(p)
            cur_len += add_len
            continue

        flush()

        if len(p) > max_chars:
            for i in range(0, len(p), max_chars):
                part = p[i : i + max_chars].strip()
                if part:
                    chunks.append(part)
        else:
            cur.append(p)
            cur_len = len(p)

    flush()
    return [c for c in chunks if c]


def _simple_clean_fallback(raw_text: str) -> str:
    """
    Deterministic fallback if LLM cleaner times out.
    Delete common scraping noise; no paraphrasing.
    """
    t = raw_text or ""
    t = re.sub(r"https?://\S+", "", t)
    junk_patterns = [
        r"Follow .*? news on .*?(?:\.|$)",
        r"Share this.*?(?:\.|$)",
        r"Sign up.*?(?:\.|$)",
        r"Subscribe.*?(?:\.|$)",
        r"Cookie (?:policy|preferences).*?(?:\.|$)",
    ]
    for pat in junk_patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# -----------------------------
# Ollama call (cleaning stays on Ollama)
# -----------------------------
def call_ollama_generate(
    api_url: str,
    model: str,
    prompt: str,
    timeout: int,
    options: Optional[dict] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # force JSON mode
    }
    if options:
        payload["options"] = options

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "") if isinstance(data, dict) else ""

# -----------------------------
# Prompts
# -----------------------------

def _build_genai_extractor_prompt(cleaned_text: str) -> str:
    # NOTE: We rely on response_schema for structure, not "ONLY JSON" prompting.
    return f"""Extract named entities from the text.

Rules:
- Extract entities ONLY from the text below.
- Every entity MUST appear EXACTLY as a substring in the text (verbatim match).
- Do NOT infer, normalize, expand, translate, or correct names.
- Deduplicate case-insensitively, keep first-seen casing.
- Prefer full names as written (e.g., "Fedor Gorst" over "Gorst" if both appear).
- If none exist for a category, return an empty array for that category.

Text:
<<<
{cleaned_text}
>>>
""".strip()

def _build_genai_cleaner_prompt(raw_text: str) -> str:
    # NOTE: We rely on response_schema for structure, not "ONLY JSON" prompting.
    return f"""You are a STRICT text cleaner.
    LEANING RULES (DELETE-ONLY):
- You may ONLY remove content; do NOT paraphrase, summarize, translate, or reorder.
- If a sentence (or clause) is kept, it MUST be copied VERBATIM from the input (same words, same order).
- Allowed micro-edits ONLY:
  * remove URLs, navigation/cookie text, share/subscribe prompts, bylines, author blocks, publisher/outlet mentions
  * remove image captions/layout fragments
  * remove duplicated lines/fragments
  * fix line-break hyphenation caused by wraps (e.g., "inter-\\nference" -> "interference") and normalize whitespace
- Discard malformed/incomplete fragments.
- Do not add any words not present in the input.

Input text:
<<<
{raw_text}
>>>
""".strip()

def _build_genai_extractor_retry_prompt(cleaned_text: str) -> str:
    return f"""You previously returned empty arrays or missed entities.

Hard rules:
- Every item MUST appear verbatim as a substring in the text.
- If there is ANY proper name (person/org/place) in the text, you MUST include it.
- Only return empty arrays if there are truly none in the text.

Text:
<<<
{cleaned_text}
>>>
""".strip()


# -----------------------------
# JSON parsing / validation
# -----------------------------
def _safe_json_loads(s: str) -> Optional[dict]:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _validate_extractor_schema(d: Dict[str, Any]) -> None:
    required = ("locations", "organizations", "persons")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Extractor JSON missing keys {missing}. Got keys: {list(d.keys())}")
    for k in required:
        v = d.get(k)
        if v is None:
            continue
        if not isinstance(v, list):
            raise ValueError(f"Extractor key '{k}' must be a list, got {type(v)}")

# -----------------------------
# Pass 1: Clean with GenAI (Gemini)
# -----------------------------
def _clean_text_with_genai(
    raw_text: str,
    model: str,
) -> str:
    
    client = GeminiClient.get_client()
    schema = _genai_cleaner_schema()

    cleaned_parts: List[str] = []

    prompt = _build_genai_cleaner_prompt(raw_text)

    cfg = types.GenerateContentConfig(
            temperature=float(os.getenv("GENAI_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("GENAI_TOP_P", "1.0")),
            response_mime_type="application/json",
            response_schema=schema,
        )
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        parsed = _safe_json_loads(getattr(resp, "text", "") or "")
        if not isinstance(parsed, dict) or "cleaned_text" not in parsed:
            raise ValueError("GenAI cleaner did not return JSON with cleaned_text")
        cleaned_parts.append(str(parsed.get("cleaned_text", "")).strip())
    except Exception as e:
        print(f"GenAI cleaner error: {e}; using deterministic fallback")
        cleaned_parts.append(_simple_clean_fallback(raw_text))
    
    joined = "\n\n".join([p for p in cleaned_parts if p.strip()]).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()

# -----------------------------
# Pass 2: Extract with GenAI (Gemini) (chunked)
# -----------------------------
def _genai_entity_schema() -> Dict[str, Any]:
    # JSON Schema for structured output
    return {
        "type": "object",
        "properties": {
            "locations": {"type": "array", "items": {"type": "string"}},
            "organizations": {"type": "array", "items": {"type": "string"}},
            "persons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["locations", "organizations", "persons"],
    }

def _genai_cleaner_schema() -> Dict[str, Any]:
    # JSON Schema for structured output
    return{
        "type": "object",
        "properties": {
            "cleaned_text": {"type": "string"},
        },
        "required": ["cleaned_text"],
    }

def _extract_entities_with_genai(
    cleaned_text: str,
    model: str,
    max_chunk_chars: int,
) -> Dict[str, List[str]]:
    chunks = _chunk_text(cleaned_text, max_chunk_chars)
    if not chunks:
        return {"locations": [], "organizations": [], "persons": []}

    client = GeminiClient.get_client()
    schema = _genai_entity_schema()

    locs: List[str] = []
    orgs: List[str] = []
    pers: List[str] = []

    for i, ch in enumerate(chunks, 1):
        prompt = _build_genai_extractor_prompt(ch)

        cfg = types.GenerateContentConfig(
            temperature=float(os.getenv("GENAI_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("GENAI_TOP_P", "1.0")),
            response_mime_type="application/json",
            response_schema=schema,
        )

        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            # With structured output, .text should already be JSON
            parsed = _safe_json_loads(getattr(resp, "text", "") or "")
            if not isinstance(parsed, dict):
                raise ValueError(f"GenAI extractor did not return JSON object. Raw: {(getattr(resp,'text','') or '')[:300]}")

            _validate_extractor_schema(parsed)
            norm = normalize_entity_lists(parsed)

            # Retry once per chunk if empty (still GenAI-only)
            if not (norm["locations"] or norm["organizations"] or norm["persons"]):
                retry_prompt = _build_genai_extractor_retry_prompt(ch)
                resp2 = client.models.generate_content(model=model, contents=retry_prompt, config=cfg)
                parsed2 = _safe_json_loads(getattr(resp2, "text", "") or "")
                if isinstance(parsed2, dict):
                    _validate_extractor_schema(parsed2)
                    norm2 = normalize_entity_lists(parsed2)
                    norm = norm2

            locs.extend(norm["locations"])
            orgs.extend(norm["organizations"])
            pers.extend(norm["persons"])

        except Exception as e:
            print(f"GenAI extractor error on chunk {i}/{len(chunks)}: {e}; skipping this chunk")

    return {
        "locations": _dedupe_preserve_first(locs),
        "organizations": _dedupe_preserve_first(orgs),
        "persons": _dedupe_preserve_first(pers),
    }

# -----------------------------
# SDK call entry point
# -----------------------------
def call_to_genai_sdk(text: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Full GenAI (Gemini) cleaning + entity extraction pipeline.
    """
    # --- GenAI extraction (use full cleaned text, do NOT cap before extraction) ---
    genai_model = os.getenv("GENAI_MODEL", "gemini-2.0-flash")

    cleaned = _clean_text_with_genai(
        raw_text=text,
        model=genai_model,
    )

    try:
        genai_chunk = int(os.getenv("GENAI_MAX_CHUNK_CHARS", "12000"))
    except Exception:
        genai_chunk = 12000

    entities = _extract_entities_with_genai(
        cleaned_text=cleaned,
        model=genai_model,
        max_chunk_chars=genai_chunk,
    )

    return {
        "cleaned_text": cleaned,
        "locations": entities.get("locations", []),
        "organizations": entities.get("organizations", []),
        "persons": entities.get("persons", []),
    }
