"""Scoring engine to evaluate pitch deck alignment against investor requirements."""

# pylint: disable=duplicate-code

import json
import re
from typing import Dict, Any
import google.generativeai as genai
from openai import OpenAI

from app.backend.nlp_extractor import extract_entities, calculate_local_similarity
from app.backend.exceptions import LLMIntegrationError

def evaluate_pitch(
    pitch_text: str,
    investor_thesis: str,
    provider: str = "offline",
    api_key: str = "",
    model_name: str = ""
) -> Dict[str, Any]:
    """Route evaluation request to the correct engine based on provider."""
    if provider == "google" and api_key:
        return evaluate_pitch_gemini(pitch_text, investor_thesis, api_key, model_name)
    if provider == "openai" and api_key:
        return evaluate_pitch_openai(pitch_text, investor_thesis, api_key, model_name)
    return evaluate_pitch_offline(pitch_text, investor_thesis)

def _calc_market_score(pitch_lower: str, base_similarity: float) -> int:
    market_keywords = [
        "market", "customer", "user", "audience", "competitor",
        "tam", "sam", "som", "pmf", "problem", "solution"
    ]
    market_matches = sum(1 for kw in market_keywords if kw in pitch_lower)
    return min(40 + (market_matches * 5) + int(base_similarity * 40), 100)

def _calc_financial_score(pitch_lower: str, entities: dict, base_similarity: float) -> int:
    financial_keywords = [
        "revenue", "profit", "margin", "pricing", "saas",
        "arr", "funding", "seed", "valuation", "financial"
    ]
    financial_matches = sum(1 for kw in financial_keywords if kw in pitch_lower)
    money_bonus = 15 if entities.get("financials") else 0
    return min(35 + (financial_matches * 5) + money_bonus + int(base_similarity * 20), 100)

def _calc_execution_score(pitch_lower: str, entities: dict, base_similarity: float) -> int:
    execution_keywords = [
        "team", "founder", "experience", "timeline", "milestone",
        "roadmap", "launch", "hiring", "strategy"
    ]
    execution_matches = sum(1 for kw in execution_keywords if kw in pitch_lower)
    dates_bonus = 15 if entities.get("dates_milestones") else 0
    return min(35 + (execution_matches * 5) + dates_bonus + int(base_similarity * 20), 100)

# pylint: disable=too-many-branches
def _analyze_gaps_and_recs(pitch_lower: str, thesis_lower: str, entities: dict) -> tuple:
    critical_terms = [
        "b2b", "saas", "growth", "revenue", "traction", "profitable",
        "scale", "seed", "experienced", "customer", "market", "product",
        "technology", "ai", "ml", "platform", "user", "retention"
    ]

    investor_demands = set()
    for term in critical_terms:
        if term in thesis_lower:
            investor_demands.add(term)

    if "team" in thesis_lower or "founder" in thesis_lower:
        investor_demands.add("team")
    if "financial" in thesis_lower or "model" in thesis_lower or "forecast" in thesis_lower:
        investor_demands.add("financial projections")
    if "competitor" in thesis_lower or "competition" in thesis_lower:
        investor_demands.add("competition analysis")

    if not investor_demands:
        investor_demands = {"market", "team", "revenue", "scale"}

    found_requirements = [d for d in investor_demands if d in pitch_lower]
    missing_requirements = [d for d in investor_demands if d not in pitch_lower]

    strengths = [
        f"Addresses investor need for '{req.title()}' with relevant context in the pitch."
        for req in found_requirements
    ]
    if entities.get("financials"):
        financials_str = ", ".join(entities['financials'][:2])
        strengths.append(
            f"Provides clear financial figures or funding targets, e.g., {financials_str}."
        )
    if entities.get("dates_milestones"):
        strengths.append("Includes chronological milestone details or execution dates.")

    if not strengths:
        strengths = ["Pitch contains standard startup definitions and structure."]

    gaps = [
        f"Missing explicit detail regarding the investor's interest in '{req.title()}'."
        for req in missing_requirements
    ]
    if not entities.get("financials"):
        gaps.append(
            "Lack of specific financial requirements, dollar values, or funding round details."
        )
    if not entities.get("dates_milestones"):
        gaps.append("Missing concrete timeline dates or execution roadmap items.")

    if not gaps:
        gaps = ["No major structural gaps identified based on the provided thesis."]

    recommendations = []
    for gap in gaps:
        if "financial" in gap.lower():
            recommendations.append(
                "Incorporate a clear financials section detailing the size of the round, "
                "expected runway, and capital allocation."
            )
        elif "timeline" in gap.lower() or "roadmap" in gap.lower() or "dates" in gap.lower():
            recommendations.append(
                "Add a 12-to-18 month product development and milestone roadmap with specific "
                "launch target dates."
            )
        elif "market" in gap.lower():
            recommendations.append(
                "Include market sizing analysis using TAM, SAM, SOM metrics to show addressable scale."
            )
        elif "team" in gap.lower():
            recommendations.append(
                "Add a 'Team' slide showcasing founders' background, key hires, and advisors to "
                "signal domain expertise."
            )
        else:
            match = re.search(r"'(.*?)'", gap)
            term_name = match.group(1) if match else "critical areas"
            recommendations.append(
                f"Elaborate on how the startup addresses '{term_name.title()}' to satisfy "
                "specific investor expectations."
            )

    if not recommendations:
        recommendations = [
            "Continue refining the pitch deck to focus on customer retention and long-term defensibility."
        ]

    return strengths, gaps, recommendations

def evaluate_pitch_offline(pitch_text: str, investor_thesis: str) -> Dict[str, Any]:
    """Calculate scores and find strengths/gaps using local NLP heuristics."""
    pitch_lower = pitch_text.lower()
    thesis_lower = investor_thesis.lower()

    entities = extract_entities(pitch_text)
    base_similarity = calculate_local_similarity(pitch_text, investor_thesis)

    market_fit_score = _calc_market_score(pitch_lower, base_similarity)
    financial_strategy_score = _calc_financial_score(pitch_lower, entities, base_similarity)
    execution_readiness_score = _calc_execution_score(pitch_lower, entities, base_similarity)

    overall_score = int(
        (market_fit_score * 0.35) +
        (financial_strategy_score * 0.35) +
        (execution_readiness_score * 0.30)
    )

    strengths, gaps, recommendations = _analyze_gaps_and_recs(pitch_lower, thesis_lower, entities)

    return {
        "overall_score": overall_score,
        "market_fit_score": market_fit_score,
        "financial_strategy_score": financial_strategy_score,
        "execution_readiness_score": execution_readiness_score,
        "strengths": strengths,
        "gaps": gaps,
        "recommendations": recommendations[:5]
    }

def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Helper to extract a JSON object from text (handling markdown code blocks if present)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    json_str = match.group(1) if match else raw_text
    json_str = json_str.strip()
    return json.loads(json_str)

def get_system_prompt() -> str:
    """Returns the central instruction prompt for pitch evaluation."""
    return (
        "You are an expert venture capitalist and startup pitch coach. Analyze the provided startup pitch text "
        "and score it against the investor's requirements / thesis.\n"
        "You must return a raw JSON object with the following schema:\n"
        "{\n"
        "  \"overall_score\": 75, \n"
        "  \"market_fit_score\": 80, \n"
        "  \"financial_strategy_score\": 70, \n"
        "  \"execution_readiness_score\": 75, \n"
        "  \"strengths\": [\"Strength 1\", \"Strength 2\"],\n"
        "  \"gaps\": [\"Gap 1\", \"Gap 2\"],\n"
        "  \"recommendations\": [\"Recommendation 1\", \"Recommendation 2\"]\n"
        "}\n"
        "Scores must be integers between 0 and 100.\n"
        "Do not include any chat prefix, markdown formatting, or text outside the JSON object. Return ONLY valid JSON."
    )

def evaluate_pitch_gemini(
    pitch_text: str,
    investor_thesis: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Call Google Gemini to score and analyze the pitch."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name or "gemini-1.5-flash",
            system_instruction=get_system_prompt()
        )

        prompt = (
            f"STARTUP PITCH TEXT:\n{pitch_text[:40000]}\n\n"
            f"INVESTOR THESIS / VC REQUIREMENTS:\n{investor_thesis[:10000]}\n\n"
            f"Analyze and output the evaluation in JSON format."
        )

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        return clean_json_response(response.text)
    except Exception as e:
        raise LLMIntegrationError(f"Gemini API analysis failed: {str(e)}") from e

def evaluate_pitch_openai(
    pitch_text: str,
    investor_thesis: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Call OpenAI GPT model to score and analyze the pitch."""
    try:
        client = OpenAI(api_key=api_key)

        prompt = (
            f"STARTUP PITCH TEXT:\n{pitch_text[:40000]}\n\n"
            f"INVESTOR THESIS / VC REQUIREMENTS:\n{investor_thesis[:10000]}\n\n"
            f"Analyze and output the evaluation in JSON format."
        )

        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return clean_json_response(response.choices[0].message.content)
    except Exception as e:
        raise LLMIntegrationError(f"OpenAI API analysis failed: {str(e)}") from e
