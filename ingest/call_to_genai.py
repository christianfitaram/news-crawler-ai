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

# -----------------------------
# Pass 1: Clean with GenAI (Gemini)
# -----------------------------
def _clean_text_with_genai(raw_text: str, model: str) -> str:
    client = GeminiClient.get_client()
    
    # SYSTEM INSTRUCTION: Sets the permanent behavior/rules
    sys_instr = (
        "Act as a VERBATIM text cleaner. Output ONLY JSON. "
        "Rules: DELETE-ONLY. No paraphrasing, summarizing, or reordering. "
        "Keep text exactly as provided. "
        "REMOVE: URLs, nav/cookie text, ads, bylines, author/publisher info, "
        "image captions, and layout fragments. "
        "FIX: Normalize whitespace and join hyphenated line-breaks."
    )

    cfg = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=_genai_cleaner_schema(),
    )

    try:
        # The user content now ONLY contains the variable data
        resp = client.models.generate_content(
            model=model, 
            contents=f"TEXT TO CLEAN:\n{raw_text}", 
            config=cfg
        )
        parsed = _safe_json_loads(getattr(resp, "text", "") or "")
        return str(parsed.get("cleaned_text", "")).strip()
    except Exception as e:
        print(f"GenAI cleaner error: {e}")
        return _simple_clean_fallback(raw_text)
    
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
    client = GeminiClient.get_client()

    # SYSTEM INSTRUCTION: Permanent rules for extraction
    sys_instr = (
        "Extract named entities ONLY from the provided text. "
        "STRICT RULE: Every entity MUST be a verbatim substring from the input. "
        "Do NOT infer, translate, or correct names. "
        "Deduplicate case-insensitively, keeping the first-seen casing. "
        "Return empty arrays if no entities are found."
    )

    cfg = types.GenerateContentConfig(
        system_instruction=sys_instr,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=_genai_entity_schema(),
    )

    locs, orgs, pers = [], [], []

    for ch in chunks:
        try:
            resp = client.models.generate_content(
                model=model, 
                contents=f"EXTRACT FROM THIS TEXT:\n{ch}", 
                config=cfg
            )
            parsed = _safe_json_loads(getattr(resp, "text", "") or "")
            if isinstance(parsed, dict):
                norm = normalize_entity_lists(parsed)
                locs.extend(norm["locations"])
                orgs.extend(norm["organizations"])
                pers.extend(norm["persons"])
        except Exception as e:
            print(f"Extraction error: {e}")

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
    genai_model = "gemini-2.5-flash-lite"
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
