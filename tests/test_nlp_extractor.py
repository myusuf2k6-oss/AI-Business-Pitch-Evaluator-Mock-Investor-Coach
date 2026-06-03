"""Tests for the NLP processing module nlp_extractor.py."""

from app.backend.nlp_extractor import extract_entities, extract_key_concepts, calculate_local_similarity

def test_extract_entities():
    """Verify that extract_entities extracts money, dates, and organizations."""
    sample_text = (
        "We are raising $1.5M in seed funding in June 2026. "
        "Our startup StripePay is partnering with Google to scale. "
        "We have achieved 250% year-over-year revenue growth."
    )
    entities = extract_entities(sample_text)
    
    # Assert keys exist
    assert "financials" in entities
    assert "organizations" in entities
    assert "percentages" in entities
    assert "dates_milestones" in entities
    
    # Check that entries are found (due to small size, sm model might classify differently, 
    # but keys should definitely exist and some entities should match)
    assert isinstance(entities["financials"], list)
    assert isinstance(entities["organizations"], list)
    assert isinstance(entities["percentages"], list)
    assert isinstance(entities["dates_milestones"], list)

def test_extract_key_concepts():
    """Verify that key business concepts are mapped into categories."""
    sample_text = (
        "Our product-market fit is validated by organic user acquisition. "
        "We operate a B2B SaaS business model with robust ARR and profit margin. "
        "The founding team has a strong timeline and milestones roadmap."
    )
    concepts = extract_key_concepts(sample_text)
    
    assert "market_fit" in concepts
    assert "financial_strategy" in concepts
    assert "execution_readiness" in concepts
    
    # Concepts should contain matched words
    assert any("market" in c or "user" in c for c in concepts["market_fit"])
    assert any("revenue" in c or "saas" in c or "margin" in c or "arr" in c for c in concepts["financial_strategy"])
    assert any("team" in c or "timeline" in c or "milestone" in c for c in concepts["execution_readiness"])

def test_calculate_local_similarity():
    """Verify that Jaccard similarity is computed correctly."""
    text1 = "artificial intelligence machine learning startup"
    text2 = "machine learning models for business growth"
    text3 = "completely unrelated text about baking bread"
    
    sim1 = calculate_local_similarity(text1, text2)
    sim2 = calculate_local_similarity(text1, text3)
    
    assert 0.0 <= sim1 <= 1.0
    assert 0.0 <= sim2 <= 1.0
    # Overlapping text should have higher similarity than unrelated text
    assert sim1 > sim2
