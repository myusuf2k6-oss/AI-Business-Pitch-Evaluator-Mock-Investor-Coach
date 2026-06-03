"""Tests for the Mock Investor Coach module coach.py."""

from unittest.mock import MagicMock, patch
import pytest
from app.backend.coach import (
    generate_questions,
    generate_questions_offline,
    evaluate_answer,
    evaluate_answer_offline,
    generate_performance_summary
)

def test_generate_questions_offline():
    """Verify that offline question generator returns exactly 5 questions with IDs and categories."""
    qs = generate_questions_offline("We are developing a new medical device technology for hospitals.")
    
    assert len(qs) == 5
    for q in qs:
        assert "id" in q
        assert "category" in q
        assert "question" in q
        assert q["category"] in ["Market Fit", "Financial Strategy", "Execution Readiness"]

def test_evaluate_answer_offline_short():
    """Verify short answers receive low scores and correct weakness feedback."""
    res = evaluate_answer_offline("What is your TAM?", "Market Fit", "It is big.")
    assert res["score"] < 50
    assert "Lacks specific market validation" in res["weaknesses"]

def test_evaluate_answer_offline_good_market():
    """Verify robust answers with keywords receive higher scores."""
    answer = (
        "We validated our customer demand through 50 user interviews showing a clear PMF problem. "
        "Our primary competitor lacks our proprietary data acquisition loop. "
        "We plan to acquire users through content marketing."
    )
    res = evaluate_answer_offline("Question", "Market Fit", answer)
    assert res["score"] >= 75
    assert "customer" in res["strengths"] or "user" in res["strengths"]

def test_evaluate_answer_offline_financial():
    """Verify robust financial answers receive higher scores."""
    answer = (
        "We are raising 1.5M in funding to give us 18 months of runway. "
        "Our pricing model will deliver 80% profit margins. "
        "Our revenue projections show scaling to 5M ARR."
    )
    res = evaluate_answer_offline("Question", "Financial Strategy", answer)
    assert res["score"] >= 80
    assert "financial" in res["strengths"] or "pricing" in res["strengths"]

def test_generate_performance_summary():
    """Verify scorecard averages and tier assignments."""
    evals_high = [
        {"score": 90}, {"score": 85}, {"score": 95}, {"score": 80}, {"score": 90}
    ]
    summary_high = generate_performance_summary(evals_high)
    assert summary_high["average_score"] == 88
    assert "Highly Recommended" in summary_high["tier"]
    
    evals_low = [
        {"score": 50}, {"score": 60}, {"score": 40}, {"score": 55}, {"score": 45}
    ]
    summary_low = generate_performance_summary(evals_low)
    assert summary_low["average_score"] == 50
    assert "Needs Work" in summary_low["tier"]

@patch("app.backend.coach.genai.GenerativeModel")
def test_generate_questions_gemini(mock_gen_model):
    """Verify that Gemini question generator executes correctly with mocks."""
    mock_model_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        '['
        '  {"id": 1, "category": "Market Fit", "question": "Q1"},'
        '  {"id": 2, "category": "Market Fit", "question": "Q2"},'
        '  {"id": 3, "category": "Financial Strategy", "question": "Q3"},'
        '  {"id": 4, "category": "Financial Strategy", "question": "Q4"},'
        '  {"id": 5, "category": "Execution Readiness", "question": "Q5"}'
        ']'
    )
    mock_model_inst.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_inst
    
    qs = generate_questions("pitch text", provider="google", api_key="key", model_name="gemini-1.5-flash")
    assert len(qs) == 5
    assert qs[0]["question"] == "Q1"
