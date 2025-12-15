"""
Minimal spaCy demo using the `en_core_web_trf` model to print detected entities.

Run:
    poetry run python -m enrich_news_article.spacy_demo
"""

from __future__ import annotations

import spacy
from typing import List

_GEO_ALIASES = {
    "u s": "united states",
    "u.s.": "united states",
    "u.s": "united states",
    "us": "united states",
    "united states": "united states",
    "the united states": "united states",
    "The United States": "united states",
    "america": "united states",
    "the United States": "united states",
    "the US": "united states",
    "the U.S.": "united states",
    "the U.S": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "u.k": "united kingdom",
    "united kingdom": "united kingdom",
    "the united kingdom": "united kingdom",
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    "u.a.e": "united arab emirates",
    "the uae": "united arab emirates",
    "The UAE": "united arab emirates",
    "the United Arab Emirates": "united arab emirates",
    "the united arab emirates": "united arab emirates",
    "The United Arab Emirates": "united arab emirates",
    "eu": "european union",
    "e.u.": "european union",
    "e.u": "european union",
    "european union": "european union",
    "the european union": "european union",
    "the EU": "european union",
    "the E.U.": "european union",
    "the E.U": "european union",
    "the European Union": "european union",
    "ONU": "united nations",
    "O.N.U.": "united nations",
    "O.N.U": "united nations",
    "united nations": "united nations",
    "the united nations": "united nations",
    "the UN": "united nations",
    "the U.N.": "united nations",
    "the U.N": "united nations",
    "the United Nations": "united nations",
}

def load_model():
    """Load the spaCy model."""
    return spacy.load("en_core_web_trf")


def process(text: str, nlp_model: spacy.language.Language) -> spacy.tokens.Doc:
    """Process the input text using the provided spaCy model."""
    return nlp_model(text)


def delete_possessive_marks(text: str) -> str:
    """Remove possessive marks from the text."""
    return text.replace("'s", "")



def deduplicate_contained_names(names: List[str]) -> List[str]:
    """Deduplicate names that are substrings of other names."""
    sorted_names = sorted(names, key=lambda x: len(x), reverse=True)

    kept = []

    for name in sorted_names:
        name_lc = name.lower()

        # Check if this name is contained in any already kept name
        if any(name_lc in kept_name.lower() for kept_name in kept):
            continue

        kept.append(name)

    # Optional: restore original input order
    original_order = {name: i for i, name in enumerate(names)}
    kept.sort(key=lambda x: original_order[x])

    return kept

def create_key(name: str) -> str:
    """Create a unique key for an entity based on its name and label."""
    name = delete_possessive_marks(name.strip().lower())

    # split and join to remove extra spaces and quotes
    name = " ".join(name.replace('"', '').replace("'", "").split())

    if(name in _GEO_ALIASES):
        return _GEO_ALIASES[name]
    return name.lower()

def contruct_response_object(data: spacy.tokens.Doc) -> dict:
    """Bucket entities into simple deduped lists (dedupe by normalized key)."""
    buckets = {
        "locations": {},
        "persons": {},
        "organizations": {},
    }

    for ent in data.ents:
        key = create_key(ent.text)
        if ent.label_ == "GPE":
            buckets["locations"].setdefault(key, {"name": ent.text, "key": key})
        elif ent.label_ == "PERSON":
            buckets["persons"].setdefault(key, {"name": ent.text, "key": key})
        elif ent.label_ == "ORG":
            buckets["organizations"].setdefault(key, {"name": ent.text, "key": key})

    # Convert dicts to lists for JSON-friendly output
    return {k: list(v.values()) for k, v in buckets.items()}


def main(text: str | None = None) -> dict:
    """Run the spaCy demo."""
    payload = text

    doc = process(payload, load_model())
    response = contruct_response_object(doc)
    return response

