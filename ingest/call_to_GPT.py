from typing import Optional
import requests
import json

def _normalize_entity_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("'s", "")
    key = key.replace('"', "").replace("'", "")
    key = key.replace(",", "")
    key = " ".join(key.split())
    return key

def normalize_entity_lists(entities: dict) -> dict:
    normalized = {}
    for field in ("locations", "organizations", "persons"):
        values = entities.get(field, [])
        items = []
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

def _normalize_gpt_payload(payload: dict) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    cleaned_text = payload.get("cleaned_text")
    if not isinstance(cleaned_text, str):
        return None

    entities = normalize_entity_lists(payload)

    return {
        "cleaned_text": cleaned_text.strip(),
        "locations": entities.get("locations", []),
        "organizations": entities.get("organizations", []),
        "persons": entities.get("persons", []),
    }

def _parse_gpt_json(raw_response) -> Optional[dict]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        return None
    raw_text = raw_response.strip()
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw_text[start:end + 1])
        except json.JSONDecodeError:
            return None


def call_to_gpt_api(prompt: str, timeout: int = 60) -> dict:
    prompt_final = f"""
You are a professional text cleaner and entity extraction engine.
Output must be ONLY valid JSON.

Return a single JSON object with EXACTLY these keys:
{{
  "cleaned_text": string,
  "locations": string[],
  "organizations": string[],
  "persons": string[]
}}

Tasks:
1. Clean the input text.
2. Extract named entities (locations, organizations, persons) from the cleaned text.

Rules:
- Output JSON ONLY. No markdown, no explanation, no extra keys, no trailing commas.
- Do NOT add new information or commentary.
- Entities must appear explicitly in the text. Do NOT infer or normalize beyond simple cleanup.
- Remove noise and artifacts, including:
  * news outlets, publishers, authors, bylines
  * URLs, navigation text, cookie banners, image captions, layout fragments
- Discard malformed, incomplete, duplicated, or irrelevant fragments.
- Discard generic groups (e.g., "protesters", "police", "organizers", "resident") unless a proper name is given.
- Deduplicate entities (case-insensitive) while preserving original casing.
- Prefer full names as written in the text.
- If no entities exist for a category, return an empty array.
- The cleaned_text must be coherent, readable, and derived only from the original content.

Input text:
<<<
{prompt}
>>>
""".strip()


    api_url = "http://localhost:11434/api/generate"

    payload = {
        "model": "gpt-oss:20b",
        "prompt": prompt_final,
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        data = response.json()
        parsed = _parse_gpt_json(data.get("response", ""))
        if parsed:
            cleaned_text = parsed.get("cleaned_text", prompt)
            if not isinstance(cleaned_text, str):
                cleaned_text = prompt
            locations = parsed.get("locations", [])
            organizations = parsed.get("organizations", [])
            persons = parsed.get("persons", [])
            return {
                "cleaned_text": cleaned_text.strip(),
                "locations": locations if isinstance(locations, list) else [],
                "organizations": organizations if isinstance(organizations, list) else [],
                "persons": persons if isinstance(persons, list) else [],
            }
        print("GPT API response did not include valid JSON, using original text")
        return {
            "cleaned_text": prompt,
            "locations": [],
            "organizations": [],
            "persons": [],
        }
    except (json.JSONDecodeError, ValueError):
        print("GPT API returned invalid JSON, using original text")
        return {
            "cleaned_text": prompt,
            "locations": [],
            "organizations": [],
            "persons": [],
        }
    except requests.exceptions.Timeout:
        print(f"GPT API timeout after {timeout}s, using original text")
        return {
            "cleaned_text": prompt,
            "locations": [],
            "organizations": [],
            "persons": [],
        }
    except requests.exceptions.RequestException as e:
        print(f"GPT API error: {e}, using original text")
        return {
            "cleaned_text": prompt,
            "locations": [],
            "organizations": [],
            "persons": [],
        }

if __name__ == "__main__":
    data = call_to_gpt_api("Deal to avoid drug tariffs has no underlying text beyond limited headline terms. Ministers and senior MPs have warned that the agreements with Donald Trump are ‘built on sand’ Concerns over the basis of the agreement have been heightened by Washington’s decision to suspend the £31bn ‘tech prosperity deal’ The deal was paused after the US claimed a lack of progress from the UK in lowering trade barriers in other areas.\nGovernment figures downplayed the chances of the US reneging on the pharma deal, which took weeks longer to finalise than expected. One source said the US pharmaceutical industry had been pushing for the agreement as they wanted certainty on imports and drug prices, while by comparison the tech prosperity deal ‘was always quite abstract’ Another said this instability was the ‘new normal now in our relationship across the pond’\nUS-UK tariff deal agreed last May still not formally approved. Quota on beef exports that were due to kick in next month still not approved. US informed the UK it was pausing the tech prosperity deal over wider trade disagreements last week before Peter Kyle, the trade secretary, held meetings with senior US officials in Washington, including commerce secretary Howard Lutnick. Several officials said those meetings were ‘very positive’")
    print(data)
