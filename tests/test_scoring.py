"""Tests for the scoring engine scoring.py."""

from unittest.mock import MagicMock, patch
import pytest
from app.backend.scoring import evaluate_pitch, evaluate_pitch_offline, clean_json_response

def test_clean_json_response():
    """Verify that JSON objects wrapped in markdown formatting are cleaned and parsed."""
    raw_markdown = "```json\n{\n  \"test_score\": 90\n}\n```"
    raw_text = "{\n  \"test_score\": 90\n}"
    
    assert clean_json_response(raw_markdown) == {"test_score": 90}
    assert clean_json_response(raw_text) == {"test_score": 90}

def test_evaluate_pitch_offline():
    """Verify the local fallback scorer produces valid structured scores and lists."""
    pitch = (
        "We are building StripePay, a new B2B SaaS platform for merchant invoicing. "
        "The market size is huge with a TAM of $15B. The founder has senior experience. "
        "We need $1M in seed funding to execute on our milestones."
    )
    thesis = "Looking for early B2B SaaS startups with a technical founder and large TAM."
    
    res = evaluate_pitch_offline(pitch, thesis)
    
    # Assert output keys
    assert "overall_score" in res
    assert "market_fit_score" in res
    assert "financial_strategy_score" in res
    assert "execution_readiness_score" in res
    assert "strengths" in res
    assert "gaps" in res
    assert "recommendations" in res
    
    # Verify bounds
    assert 0 <= res["overall_score"] <= 100
    assert 0 <= res["market_fit_score"] <= 100
    assert 0 <= res["financial_strategy_score"] <= 100
    assert 0 <= res["execution_readiness_score"] <= 100
    
    # Verify content types
    assert isinstance(res["strengths"], list)
    assert isinstance(res["gaps"], list)
    assert isinstance(res["recommendations"], list)
    
    # Verify we successfully extracted strengths based on matching words (e.g. SaaS / TAM / B2B)
    assert len(res["strengths"]) > 0

@patch("app.backend.scoring.genai.GenerativeModel")
@patch("app.backend.scoring.genai.configure")
def test_evaluate_pitch_gemini(mock_configure, mock_gen_model):
    """Verify the Gemini scoring wrapper calls the API and parses results."""
    mock_model_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"overall_score": 85, "market_fit_score": 90, "financial_strategy_score": 80, "execution_readiness_score": 85, "strengths": ["test strength"], "gaps": ["test gap"], "recommendations": ["test rec"]}'
    mock_model_inst.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_inst
    
    res = evaluate_pitch("pitch text", "thesis text", provider="google", api_key="test-key", model_name="gemini-1.5-flash")
    
    assert res["overall_score"] == 85
    mock_configure.assert_called_once_with(api_key="test-key")
    mock_model_inst.generate_content.assert_called_once()

@patch("app.backend.scoring.OpenAI")
def test_evaluate_pitch_openai(mock_openai_class):
    """Verify the OpenAI scoring wrapper calls the API and parses results."""
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completion = MagicMock()
    mock_message = MagicMock()
    
    mock_message.content = '{"overall_score": 95, "market_fit_score": 95, "financial_strategy_score": 95, "execution_readiness_score": 95, "strengths": [], "gaps": [], "recommendations": []}'
    mock_completion.message = mock_message
    mock_chat.choices = [mock_completion]
    mock_client.chat.completions.create.return_value = mock_chat
    mock_openai_class.return_value = mock_client
    
    res = evaluate_pitch("pitch text", "thesis text", provider="openai", api_key="test-key", model_name="gpt-4o-mini")
    
    assert res["overall_score"] == 95
    mock_openai_class.assert_called_once_with(api_key="test-key")
    mock_client.chat.completions.create.assert_called_once()
