"""NLP processing module using spaCy to extract entities, concepts, and calculate alignment."""

import subprocess
import sys
from typing import Dict, List, Set

import spacy
from app.backend.exceptions import NLPProcessingError

# Global variable to cache the loaded spaCy model
_nlp_model = None

def get_nlp():
    """Load and cache the spaCy model. Download it if not already available."""
    global _nlp_model  # pylint: disable=global-statement
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            try:
                # Attempt to download the model programmatically
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                _nlp_model = spacy.load("en_core_web_sm")
            except Exception as e:
                raise NLPProcessingError(
                    f"Failed to load or download spaCy model 'en_core_web_sm': {str(e)}"
                ) from e
    return _nlp_model

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract Named Entities relevant to pitch assessment (Money, Orgs, Percentages, Dates)."""
    try:
        nlp = get_nlp()
        doc = nlp(text[:100000])  # Cap length to avoid massive memory usage on huge inputs

        entities = {
            "financials": [],        # MONEY
            "organizations": [],     # ORG
            "percentages": [],       # PERCENT
            "dates_milestones": []   # DATE
        }

        label_map = {
            "MONEY": "financials",
            "ORG": "organizations",
            "PERCENT": "percentages",
            "DATE": "dates_milestones"
        }

        for ent in doc.ents:
            val = ent.text.strip()
            if not val or len(val) < 2:
                continue

            category = label_map.get(ent.label_)
            if category and val not in entities[category]:
                entities[category].append(val)

        # Limit to top 15 each for cleanliness
        for k in entities:
            entities[k] = entities[k][:15]

        return entities
    except Exception as e:
        if isinstance(e, NLPProcessingError):
            raise e
        raise NLPProcessingError(f"Error extracting entities: {str(e)}") from e

def extract_key_concepts(text: str) -> Dict[str, List[str]]:
    """Analyze text for key business concepts based on vocabulary matching of noun chunks."""
    try:
        nlp = get_nlp()
        doc = nlp(text.lower()[:100000])

        # Categorized vocabulary to map noun chunks/phrases
        categories = {
            "market_fit": {
                "market", "customer", "user", "audience", "competitor", "competition",
                "acquisition", "cac", "ltv", "demographic", "growth", "segment",
                "problem", "solution", "demand", "tam", "sam", "som", "product-market fit"
            },
            "financial_strategy": {
                "revenue", "profit", "loss", "margin", "pricing", "subscription",
                "saas", "arr", "mrr", "cost", "expense", "funding", "seed", "series",
                "round", "valuation", "ebitda", "capital", "cash flow", "monetization"
            },
            "execution_readiness": {
                "team", "founder", "advisor", "experience", "background", "timeline",
                "milestone", "roadmap", "launch", "pipeline", "strategy", "operation",
                "executing", "hiring", "developer", "track record", "scale", "phase"
            }
        }

        extracted = {
            "market_fit": set(),
            "financial_strategy": set(),
            "execution_readiness": set()
        }

        # Look for noun chunks that contain our keyword stems
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if len(chunk_text) < 3:
                continue

            # Classify chunks based on keyword matching
            for cat_name, keywords in categories.items():
                for kw in keywords:
                    if kw in chunk_text:
                        extracted[cat_name].add(chunk_text)
                        break

        # Convert sets to sorted lists to make them JSON-serializable
        return {k: sorted(list(v))[:15] for k, v in extracted.items()}
    except Exception as e:
        raise NLPProcessingError(f"Error extracting business concepts: {str(e)}") from e

def calculate_local_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity of lemmatized non-stopwords between two texts."""
    try:
        nlp = get_nlp()

        def get_tokens(text: str) -> Set[str]:
            doc = nlp(text.lower()[:50000])
            tokens = set()
            for token in doc:
                if not token.is_stop and not token.is_punct and token.is_alpha:
                    tokens.add(token.lemma_)
            return tokens

        set1 = get_tokens(text1)
        set2 = get_tokens(text2)

        if not set1 or not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union)
    except Exception as e:
        raise NLPProcessingError(f"Error calculating similarity: {str(e)}") from e
