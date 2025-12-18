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


def call_to_gpt_api(prompt: str, timeout: int = 60) -> dict:
    prompt_final = """You are a professional text cleaner and entity extractor.
Return a strict JSON object with only these keys:
- cleaned_text (string)
- locations (array of strings)
- organizations (array of strings)
- persons (array of strings)
Rules:
- Remove references to news outlets, authors, publication names, URLs, or web layout artifacts.
- Discard malformed, incomplete, or irrelevant fragments.
- Do not add new information or commentary.
- Locate and extract all principal named entities (locations, organizations, persons) mentioned in the text do not add any new entities that are not present.
- Ensure the JSON is well-formed and parsable.
- If there are no entities, return empty arrays.
Text to rewrite:
""" + prompt

    api_url = "http://35.204.248.56:11434/api/generate"

    payload = {
        "model": "gpt-oss:20b",
        "prompt": prompt_final,
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        data = response.json()
        parsed = _parse_gpt_json(data.get("response", "").strip())
        normalized = _normalize_gpt_payload(parsed) if parsed else None
        if normalized:
            return normalized
        print("GPT API response did not include valid JSON, using original text")
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

def _parse_gpt_json(raw_text: str) -> Optional[dict]:
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

if __name__ == "__main__":
    data = call_to_gpt_api("Deal to avoid drug tariffs has no underlying text beyond limited headline terms. Ministers and senior MPs have warned that the agreements with Donald Trump are ‘built on sand’ Concerns over the basis of the agreement have been heightened by Washington’s decision to suspend the £31bn ‘tech prosperity deal’ The deal was paused after the US claimed a lack of progress from the UK in lowering trade barriers in other areas.\nGovernment figures downplayed the chances of the US reneging on the pharma deal, which took weeks longer to finalise than expected. One source said the US pharmaceutical industry had been pushing for the agreement as they wanted certainty on imports and drug prices, while by comparison the tech prosperity deal ‘was always quite abstract’ Another said this instability was the ‘new normal now in our relationship across the pond’\nUS-UK tariff deal agreed last May still not formally approved. Quota on beef exports that were due to kick in next month still not approved. US informed the UK it was pausing the tech prosperity deal over wider trade disagreements last week before Peter Kyle, the trade secretary, held meetings with senior US officials in Washington, including commerce secretary Howard Lutnick. Several officials said those meetings were ‘very positive’")
    print(data)