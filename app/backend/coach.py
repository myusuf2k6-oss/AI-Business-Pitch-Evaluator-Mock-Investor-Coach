# pylint: disable=duplicate-code, too-many-arguments, too-many-positional-arguments
"""Mock Investor Coach engine for generating VC questions and evaluating founder responses."""

from typing import Dict, List, Any
import google.generativeai as genai
from openai import OpenAI

from app.backend.nlp_extractor import get_nlp
from app.backend.scoring import clean_json_response
from app.backend.exceptions import LLMIntegrationError

# Offline template questions
OFFLINE_QUESTIONS = [
    {
        "id": 1,
        "category": "Market Fit",
        "question": "How did you validate the demand for your solution, and what specific customer feedback drove your "
                    "initial product-market fit?"
    },
    {
        "id": 2,
        "category": "Market Fit",
        "question": "Who is your primary competitor in this space, and what is your sustainable unfair advantage "
                    "(defensibility) against them?"
    },
    {
        "id": 3,
        "category": "Financial Strategy",
        "question": "Walk me through your unit economics. What is your estimated Customer Acquisition Cost (CAC) "
                    "relative to the Customer Lifetime Value (LTV)?"
    },
    {
        "id": 4,
        "category": "Financial Strategy",
        "question": "If you raise this round, how exactly will you allocate the capital over the next 18 months, and "
                    "what major milestones will this unlock?"
    },
    {
        "id": 5,
        "category": "Execution Readiness",
        "question": "What are the core technical or execution risks that could keep you from scaling, and how does "
                    "your founding team's experience uniquely position you to mitigate them?"
    }
]

def generate_questions(
    pitch_text: str,
    provider: str = "offline",
    api_key: str = "",
    model_name: str = ""
) -> List[Dict[str, Any]]:
    """Generate 5 tailored investor questions based on the pitch text."""
    if provider == "google" and api_key:
        return generate_questions_gemini(pitch_text, api_key, model_name)
    if provider == "openai" and api_key:
        return generate_questions_openai(pitch_text, api_key, model_name)
    return generate_questions_offline(pitch_text)

def generate_questions_offline(pitch_text: str) -> List[Dict[str, Any]]:
    """Return template questions slightly customized with extracted terms from pitch text."""
    try:
        nlp = get_nlp()
        doc = nlp(pitch_text.lower()[:20000])

        # Extract nouns to insert dynamically
        nouns = [
            token.text for token in doc
            if token.pos_ == "NOUN" and not token.is_stop and len(token.text) > 3
        ]
        tech_nouns = [
            n for n in nouns
            if n in ["platform", "software", "app", "service", "technology", "product", "algorithm", "data"]
        ]
        domain = tech_nouns[0] if tech_nouns else (nouns[0] if nouns else "solution")

        custom_questions = []
        for q in OFFLINE_QUESTIONS:
            new_q = q.copy()
            if "solution" in new_q["question"] and domain != "solution":
                new_q["question"] = new_q["question"].replace("solution", f"{domain} solution")
            custom_questions.append(new_q)
        return custom_questions
    except Exception:  # pylint: disable=broad-exception-caught
        return OFFLINE_QUESTIONS

def generate_questions_gemini(pitch_text: str, api_key: str, model_name: str) -> List[Dict[str, Any]]:
    """Generate questions using Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name or "gemini-1.5-flash")

        prompt = (
            "You are an active venture capitalist. Review this startup pitch text and generate exactly 5 tough, "
            "analytical investor questions. Tailor the questions specifically to the weaknesses or gaps you find "
            "in the text. Return a JSON array of objects, with each object containing 'id' (integer 1-5), "
            "'category' ('Market Fit', 'Financial Strategy', or 'Execution Readiness'), and 'question' (string).\n"
            f"PITCH TEXT:\n{pitch_text[:30000]}\n"
            "Return ONLY a valid JSON array, do not wrap in markdown or include conversational prefixes."
        )

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        data = clean_json_response(response.text)
        if isinstance(data, list) and len(data) >= 5:
            return data[:5]
        return OFFLINE_QUESTIONS
    except Exception as e:
        raise LLMIntegrationError(f"Gemini failed to generate questions: {str(e)}") from e

def generate_questions_openai(pitch_text: str, api_key: str, model_name: str) -> List[Dict[str, Any]]:
    """Generate questions using OpenAI GPT."""
    try:
        client = OpenAI(api_key=api_key)

        prompt = (
            "You are an active venture capitalist. Review this startup pitch text and generate exactly 5 tough, "
            "analytical investor questions. Tailor the questions specifically to the weaknesses or gaps you find "
            "in the text. Return a JSON array of objects, with each object containing 'id' (integer 1-5), "
            "'category' ('Market Fit', 'Financial Strategy', or 'Execution Readiness'), and 'question' (string).\n"
            f"PITCH TEXT:\n{pitch_text[:30000]}\n"
            "Return ONLY a valid JSON array."
        )

        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        raw_res = response.choices[0].message.content
        data = clean_json_response(raw_res)
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    data = val
                    break
        if isinstance(data, list) and len(data) >= 5:
            return data[:5]
        return OFFLINE_QUESTIONS
    except Exception as e:
        raise LLMIntegrationError(f"OpenAI failed to generate questions: {str(e)}") from e

def evaluate_answer(
    pitch_text: str,
    question: str,
    category: str,
    user_answer: str,
    provider: str = "offline",
    api_key: str = "",
    model_name: str = ""
) -> Dict[str, Any]:
    """Evaluate the founder's response to an investor question."""
    if not user_answer.strip():
        return {
            "score": 0,
            "strengths": "No response provided.",
            "weaknesses": "You submitted an empty answer.",
            "suggested_response": "Please attempt to answer the question to receive investor feedback."
        }

    if provider == "google" and api_key:
        return evaluate_answer_gemini(pitch_text, question, category, user_answer, api_key, model_name)
    if provider == "openai" and api_key:
        return evaluate_answer_openai(pitch_text, question, category, user_answer, api_key, model_name)
    return evaluate_answer_offline(question, category, user_answer)

def _evaluate_market_fit(ans_lower: str) -> tuple:
    strengths_found = []
    weaknesses_found = []

    market_keywords = [
        "customer", "validate", "competitor", "market", "tam",
        "acquisition", "users", "growth", "advantage"
    ]
    matches = [kw for kw in market_keywords if kw in ans_lower]
    score_mod = len(matches) * 3
    if matches:
        strengths_found.append(f"Used market terms: {', '.join(matches[:3])}.")
    else:
        weaknesses_found.append("Lacks specific market validation or acquisition terminology.")

    suggested = (
        "Provide quantitative proof of customer demand, detail your acquisition channels "
        "(e.g., SEO, outbound sales, partnerships), and name specific competitors while explaining "
        "your defensibility."
    )
    return score_mod, strengths_found, weaknesses_found, suggested

def _evaluate_financials(ans_lower: str, user_answer: str) -> tuple:
    strengths_found = []
    weaknesses_found = []

    fin_keywords = [
        "revenue", "pricing", "margin", "funding", "runway",
        "capital", "milestones", "cost", "projection"
    ]
    matches = [kw for kw in fin_keywords if kw in ans_lower]
    score_mod = len(matches) * 3

    # Check for numbers/figures
    has_numbers = any(char.isdigit() for char in user_answer)
    if has_numbers:
        score_mod += 5
        strengths_found.append("Included specific numbers/milestone values.")
    else:
        weaknesses_found.append("Failed to list specific financial targets, percentages, or dollar figures.")

    if matches:
        strengths_found.append(f"Addressed financial topics: {', '.join(matches[:3])}.")
    else:
        weaknesses_found.append("Lacks discussion of unit economics, margins, or capital deployment.")

    suggested = (
        "Mention exact funding requirements (e.g. '$1.5M'), show runway expectations (e.g. '18 months'), "
        "outline a clear breakdown of capital allocation (e.g. '50% R&D, 30% GTM'), and highlight key economics."
    )
    return score_mod, strengths_found, weaknesses_found, suggested

def _evaluate_execution(ans_lower: str) -> tuple:
    strengths_found = []
    weaknesses_found = []

    exec_keywords = [
        "team", "founder", "experience", "roadmap", "timeline",
        "launch", "milestones", "risk", "mitigate"
    ]
    matches = [kw for kw in exec_keywords if kw in ans_lower]
    score_mod = len(matches) * 3
    if matches:
        strengths_found.append(f"Referenced operational components: {', '.join(matches[:3])}.")
    else:
        weaknesses_found.append("Lacks timeline, roadmapping, or team execution experience details.")

    suggested = (
        "Outline key milestones for the next 4 quarters, specify roles and key track records "
        "of founders, and clearly state how technical or product risk will be addressed."
    )
    return score_mod, strengths_found, weaknesses_found, suggested

def evaluate_answer_offline(question: str, category: str, user_answer: str) -> Dict[str, Any]:
    """Offline answer evaluation using basic heuristics (length, keywords, sentiment)."""
    # pylint: disable=unused-argument
    ans_lower = user_answer.lower()
    words = ans_lower.split()
    word_count = len(words)

    if word_count < 10:
        base_score = 30
    elif word_count < 25:
        base_score = 55
    elif word_count < 75:
        base_score = 75
    else:
        base_score = 85

    if category == "Market Fit":
        score_mod, strengths_found, weaknesses_found, suggested = _evaluate_market_fit(ans_lower)
    elif category == "Financial Strategy":
        score_mod, strengths_found, weaknesses_found, suggested = _evaluate_financials(ans_lower, user_answer)
    else:  # Execution Readiness
        score_mod, strengths_found, weaknesses_found, suggested = _evaluate_execution(ans_lower)

    final_score = min(base_score + score_mod, 100)

    if not strengths_found:
        strengths_found.append("Provided a direct, grammatically coherent response.")
    if not weaknesses_found:
        weaknesses_found.append("Could include more empirical details, numbers, or customer case studies.")

    return {
        "score": final_score,
        "strengths": " ".join(strengths_found),
        "weaknesses": " ".join(weaknesses_found),
        "suggested_response": suggested
    }

def get_eval_system_prompt() -> str:
    """System prompt for answering investor questions."""
    return (
        "You are an active VC investor. Evaluate the founder's response "
        "to the given question based on their pitch text.\n"
        "Return a JSON object with this format:\n"
        "{\n"
        "  \"score\": 85,\n"
        "  \"strengths\": \"Brief explanation of what the founder answered well.\",\n"
        "  \"weaknesses\": \"Brief explanation of what was missing or weak.\",\n"
        "  \"suggested_response\": \"A highly polished, sample model answer "
        "illustrating how they should have answered the question.\"\n"
        "}\n"
        "Score must be an integer between 0 and 100. Return ONLY the JSON object."
    )

def evaluate_answer_gemini(
    pitch_text: str,
    question: str,
    category: str,
    user_answer: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Use Gemini to evaluate the answer."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name or "gemini-1.5-flash",
            system_instruction=get_eval_system_prompt()
        )

        prompt = (
            f"PITCH TEXT CONTEXT:\n{pitch_text[:20000]}\n\n"
            f"QUESTION CATEGORY: {category}\n"
            f"QUESTION: {question}\n"
            f"FOUNDER'S ANSWER:\n{user_answer}\n\n"
            "Analyze and output the evaluation in JSON format."
        )

        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        return clean_json_response(response.text)
    except Exception as e:
        raise LLMIntegrationError(f"Gemini failed to evaluate answer: {str(e)}") from e

def evaluate_answer_openai(
    pitch_text: str,
    question: str,
    category: str,
    user_answer: str,
    api_key: str,
    model_name: str
) -> Dict[str, Any]:
    """Use OpenAI to evaluate the answer."""
    try:
        client = OpenAI(api_key=api_key)

        prompt = (
            f"PITCH TEXT CONTEXT:\n{pitch_text[:20000]}\n\n"
            f"QUESTION CATEGORY: {category}\n"
            f"QUESTION: {question}\n"
            f"FOUNDER'S ANSWER:\n{user_answer}\n\n"
            "Analyze and output the evaluation in JSON format."
        )

        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_eval_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return clean_json_response(response.choices[0].message.content)
    except Exception as e:
        raise LLMIntegrationError(f"OpenAI failed to evaluate answer: {str(e)}") from e

def generate_performance_summary(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce overall coach report card based on question grades."""
    if not evaluations:
        return {
            "average_score": 0,
            "tier": "Not Evaluated",
            "summary": "No interview rounds completed.",
            "recommendation": "Complete the mock interview coach process."
        }

    scores = [e.get("score", 0) for e in evaluations]
    avg_score = int(sum(scores) / len(scores))

    if avg_score >= 85:
        tier = "Investment Grade (Highly Recommended)"
        summary = "Exceptional performance. You demonstrated highly robust understanding of your market, clear " \
                  "grasp of unit economics, and precise execution timelines."
        recommendation = "You are ready for actual VC investor pitches. Keep answers concise and continue backing " \
                         "claims with empirical data."
    elif avg_score >= 70:
        tier = "Term Sheet Potential (Conditional Follow-up)"
        summary = "Strong effort. Your fundamentals are sound, but you missed key opportunities to ground your " \
                  "statements in metrics, specific TAM valuations, or concrete milestone dates."
        recommendation = "Add quantitative proof and clear dollar breakdowns to your financial and market answers " \
                         "before scheduling formal VC meetings."
    else:
        tier = "Incubation / Pre-seed Stage (Needs Work)"
        summary = "Developing strategy. Responses were too generic or lacked substantial depth in critical " \
                  "venture areas like unit economics (LTV/CAC) or sustainable defensibility."
        recommendation = "Flesh out detailed operational roadmaps and work closely with financial advisors to " \
                         "structure realistic revenue models."

    return {
        "average_score": avg_score,
        "tier": tier,
        "summary": summary,
        "recommendation": recommendation
    }
