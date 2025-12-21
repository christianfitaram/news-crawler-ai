from __future__ import annotations

from typing import Optional, Dict, Any, List, Iterable, Tuple
import os
import json
import re
import requests


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
    text = text or ""
    text = text.strip()
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

        # paragraph too big or would overflow; flush current
        flush()

        # if paragraph still too big, hard split
        if len(p) > max_chars:
            for i in range(0, len(p), max_chars):
                part = p[i:i + max_chars].strip()
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
    # remove URLs
    t = re.sub(r"https?://\S+", "", t)
    # remove common footer/promos
    junk_patterns = [
        r"Follow .*? news on .*?(?:\.|$)",
        r"Share this.*?(?:\.|$)",
        r"Sign up.*?(?:\.|$)",
        r"Subscribe.*?(?:\.|$)",
        r"Cookie (?:policy|preferences).*?(?:\.|$)",
    ]
    for pat in junk_patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE | re.DOTALL)
    # collapse excessive whitespace
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _regex_entity_fallback(cleaned_text: str) -> Dict[str, List[str]]:
    """
    Very lightweight regex fallback to avoid empty outputs.
    Conservative: collect multi-token Title Case spans.
    """
    text = cleaned_text or ""
    # Capture sequences like "Steve Reed", "Disneyland Paris", "Universal Studios"
    candidates = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b", text)
    cands = _dedupe_preserve_first(candidates)

    persons: List[str] = []
    orgs: List[str] = []
    locs: List[str] = []

    # Heuristics: if preceded by "Mr/Ms/Dr/Secretary/President" treat as person
    for c in cands:
        if re.search(rf"(Mr|Ms|Mrs|Dr|Secretary|President)\s+{re.escape(c)}", text):
            persons.append(c)

    # Common org/location suffix hints
    org_suffix = ("Ltd", "Inc", "Company", "University", "Ministry", "Council", "BBC", "Forbes", "TikTok", "YouTube")
    loc_suffix = ("Paris", "Europe", "UK", "England", "Scotland", "Wales", "Ireland", "Bedford", "Berkshire", "Surrey", "Staffordshire", "Atlanta", "Georgia")

    for c in cands:
        if c in persons:
            continue
        if any(s in c for s in org_suffix):
            orgs.append(c)
            continue
        if any(s == c.split()[-1] for s in loc_suffix):
            locs.append(c)
            continue

    return {
        "persons": _dedupe_preserve_first(persons),
        "organizations": _dedupe_preserve_first(orgs),
        "locations": _dedupe_preserve_first(locs),
    }


# -----------------------------
# Ollama call
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
    }
    if options:
        payload["options"] = options

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "") if isinstance(data, dict) else ""


def _default_ollama_options() -> Dict[str, Any]:
    """
    Sensible defaults for extraction/cleaning stability.
    You can override via env vars.
    """
    def _env_float(key: str, default: float) -> float:
        try:
            return float(os.getenv(key, str(default)))
        except Exception:
            return default

    def _env_int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except Exception:
            return default

    opts: Dict[str, Any] = {
        "temperature": _env_float("OLLAMA_TEMPERATURE", 0.0),
        "top_p": _env_float("OLLAMA_TOP_P", 0.9),
    }

    # Bigger context helps avoid truncation inside Ollama (if your model supports it)
    opts["num_ctx"] = _env_int("OLLAMA_NUM_CTX", 4096)

    # Cap generation so the model doesn't ramble (and speeds up)
    opts["num_predict"] = _env_int("OLLAMA_NUM_PREDICT", 800)

    return opts


# -----------------------------
# Prompts
# -----------------------------
def _build_cleaner_prompt(raw_text: str) -> str:
    return f"""You are a STRICT text cleaner.

Output MUST be ONLY valid JSON.
Return a single JSON object with EXACTLY this key:
{{
  "cleaned_text": string
}}

CLEANING RULES (DELETE-ONLY):
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


def _build_extractor_prompt(cleaned_text: str) -> str:
    return f"""You are a STRICT named-entity extraction engine.

Output MUST be ONLY valid JSON.
Return a single JSON object with EXACTLY these keys:
{{
  "locations": string[],
  "organizations": string[],
  "persons": string[]
}}

Rules:
- Extract entities ONLY from cleaned_text below.
- Every entity MUST appear EXACTLY as a substring in cleaned_text (verbatim match).
- Do NOT infer, normalize, expand, translate, or correct names.
- Include ALL proper names that appear (high recall).
- Do NOT include generic groups (e.g., "police", "protesters") unless a proper name is explicitly present.
- Deduplicate case-insensitively while preserving the first-seen original casing.
- Prefer full names as written (e.g., "Fedor Gorst" not "Gorst" if both appear).
- If none exist for a category, return [].
- Self-check: each array item must appear verbatim in cleaned_text.

cleaned_text:
<<<
{cleaned_text}
>>>
""".strip()


# -----------------------------
# 2-pass pipeline (chunked)
# -----------------------------
def _safe_json_loads(s: str) -> Optional[dict]:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        # Try to salvage JSON object inside (common LLM glitch)
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _clean_text_with_llm(
    raw_text: str,
    api_url: str,
    model: str,
    timeout: int,
    options: Dict[str, Any],
    max_chunk_chars: int,
) -> str:
    chunks = _chunk_text(raw_text, max_chunk_chars)
    if not chunks:
        return ""

    cleaned_parts: List[str] = []
    for i, ch in enumerate(chunks, 1):
        prompt = _build_cleaner_prompt(ch)
        try:
            out = call_ollama_generate(api_url, model, prompt, timeout=timeout, options=options)
            parsed = _safe_json_loads(out)
            if not isinstance(parsed, dict) or "cleaned_text" not in parsed:
                raise ValueError("Cleaner did not return JSON with cleaned_text")
            cleaned_parts.append(str(parsed.get("cleaned_text", "")).strip())
        except requests.exceptions.Timeout:
            print(f"GPT/Ollama cleaner timeout after {timeout}s on chunk {i}/{len(chunks)}; using deterministic fallback for this chunk")
            cleaned_parts.append(_simple_clean_fallback(ch))
        except Exception as e:
            print(f"GPT/Ollama cleaner error on chunk {i}/{len(chunks)}: {e}; using deterministic fallback for this chunk")
            cleaned_parts.append(_simple_clean_fallback(ch))

    # Join and lightly normalize whitespace
    joined = "\n\n".join([p for p in cleaned_parts if p.strip()]).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def _extract_entities_with_llm(
    cleaned_text: str,
    api_url: str,
    model: str,
    timeout: int,
    options: Dict[str, Any],
    max_chunk_chars: int,
) -> Dict[str, List[str]]:
    chunks = _chunk_text(cleaned_text, max_chunk_chars)
    if not chunks:
        return {"locations": [], "organizations": [], "persons": []}

    locs: List[str] = []
    orgs: List[str] = []
    pers: List[str] = []

    for i, ch in enumerate(chunks, 1):
        prompt = _build_extractor_prompt(ch)
        try:
            out = call_ollama_generate(api_url, model, prompt, timeout=timeout, options=options)
            parsed = _safe_json_loads(out)
            if not isinstance(parsed, dict):
                raise ValueError("Extractor did not return JSON object")
            norm = normalize_entity_lists(parsed)
            locs.extend(norm.get("locations", []))
            orgs.extend(norm.get("organizations", []))
            pers.extend(norm.get("persons", []))
        except requests.exceptions.Timeout:
            print(f"GPT/Ollama extractor timeout after {timeout}s on chunk {i}/{len(chunks)}; skipping this chunk")
        except Exception as e:
            print(f"GPT/Ollama extractor error on chunk {i}/{len(chunks)}: {e}; skipping this chunk")

    merged = {
        "locations": _dedupe_preserve_first(locs),
        "organizations": _dedupe_preserve_first(orgs),
        "persons": _dedupe_preserve_first(pers),
    }

    # Optional regex fallback if everything is empty
    if os.getenv("USE_REGEX_FALLBACK", "1").strip() in ("1", "true", "True") and not any(merged.values()):
        fb = _regex_entity_fallback(cleaned_text)
        merged = {
            "locations": fb.get("locations", []),
            "organizations": fb.get("organizations", []),
            "persons": fb.get("persons", []),
        }

    return merged


def _cap_cleaned_text_for_downstream(cleaned_text: str) -> str:
    """
    Prevent downstream transformer models with small max_length (e.g. 1024 tokens) from crashing.
    Default cap is conservative and configurable via env var.
    """
    try:
        max_chars = int(os.getenv("MAX_CLEANED_TEXT_CHARS", "4500"))
    except Exception:
        max_chars = 4500

    t = (cleaned_text or "").strip()
    if len(t) <= max_chars:
        return t

    # Keep start + end to preserve both headline entities and later details.
    head = t[: int(max_chars * 0.7)].rstrip()
    tail = t[-int(max_chars * 0.3):].lstrip()
    return (head + "\n...\n" + tail).strip()


def call_to_gpt_api(text: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper:
    - Cleans (pass 1)
    - Extracts entities (pass 2)
    - Returns dict with keys: cleaned_text, locations, organizations, persons
      where arrays are list[{"name": ..., "key": ...}] to match your pipeline.
    """
    api_url = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

    # Timeouts per chunk (override with env if you want)
    try:
        cleaner_timeout = int(os.getenv("CLEANER_TIMEOUT", str(timeout)))
    except Exception:
        cleaner_timeout = timeout
    try:
        extractor_timeout = int(os.getenv("EXTRACTOR_TIMEOUT", str(timeout)))
    except Exception:
        extractor_timeout = timeout

    opts = _default_ollama_options()

    try:
        cleaner_chunk = int(os.getenv("CLEANER_MAX_CHUNK_CHARS", "7000"))
    except Exception:
        cleaner_chunk = 7000

    try:
        extractor_chunk = int(os.getenv("EXTRACTOR_MAX_CHUNK_CHARS", "3000"))
    except Exception:
        extractor_chunk = 3000

    cleaned = _clean_text_with_llm(
        raw_text=text,
        api_url=api_url,
        model=model,
        timeout=cleaner_timeout,
        options=opts,
        max_chunk_chars=cleaner_chunk,
    )

    cleaned_capped = _cap_cleaned_text_for_downstream(cleaned)

    entities = _extract_entities_with_llm(
        cleaned_text=cleaned_capped,
        api_url=api_url,
        model=model,
        timeout=extractor_timeout,
        options=opts,
        max_chunk_chars=extractor_chunk,
    )

    # Convert to the previous "array of {name,key}" format if needed by the pipeline
    def _to_name_key_list(values: List[str]) -> List[Dict[str, str]]:
        out = []
        for v in values:
            k = _normalize_entity_key(v)
            if not k:
                continue
            out.append({"name": v, "key": k})
        return out

    return {
        "cleaned_text": cleaned_capped,
        "locations": entities.get("locations", []),
        "organizations": entities.get("organizations", []),
        "persons": entities.get("persons", []),
    }
