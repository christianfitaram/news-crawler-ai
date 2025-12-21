# Entity Extraction Troubleshooting

## Issue: Entities missing or empty

Symptoms:
- `locations`, `organizations`, and `persons` are empty even when the text contains names.
- The log prints `gpt_result` but the entity lists are missing or always empty.

Root cause:
- The endpoint returns a JSON envelope where `response` is a *string* containing the LLM's JSON.
- `call_to_gpt_api` treated `response` like a dict and called `.get(...)`, which fails and falls back to empty lists.
- The code never parsed the JSON string in `response`, so valid entity data was ignored.
- `entities_enriched` could be referenced before assignment in the classifier path.

Fix:
- Parse `data["response"]` as JSON (with a simple brace-slice fallback) before reading `cleaned_text` and entity lists.
- Validate types and fall back safely to empty arrays when the payload is malformed.
- Initialize `entities_enriched` before it is used.

Files updated:
- `ingest/call_to_GPT.py` (add `_parse_gpt_json`, parse `response` string, validate lists)
- `ingest/classifier.py` (initialize `entities_enriched`)

Quick verification:
- Run `python ingest/call_to_GPT.py` and confirm the printed `gpt_result` includes non-empty arrays when the text contains entities.
- Re-run ingestion and confirm the stored articles contain entity lists.
