# pylint: disable=wrong-import-position, broad-exception-caught
"""Streamlit application frontend for the AI Business Pitch Evaluator & Mock Investor Coach."""

import os
import streamlit as st

# Set page config at the very beginning
st.set_page_config(
    page_title="AI Business Pitch Evaluator & Mock Investor Coach",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.config import AppConfig
from app.backend.document_parser import extract_text
from app.backend.scoring import evaluate_pitch
from app.backend.coach import generate_questions, evaluate_answer, generate_performance_summary
from app.backend.exceptions import PitchEvaluatorError

# Load custom CSS
def local_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/custom.css")

# Session State initialization
if "pitch_text" not in st.session_state:
    st.session_state.pitch_text = ""
if "pitch_name" not in st.session_state:
    st.session_state.pitch_name = ""
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "coach_questions" not in st.session_state:
    st.session_state.coach_questions = []
if "coach_current_index" not in st.session_state:
    st.session_state.coach_current_index = 0
if "coach_evaluations" not in st.session_state:
    st.session_state.coach_evaluations = []
if "coach_answers" not in st.session_state:
    st.session_state.coach_answers = {}
if "coach_active" not in st.session_state:
    st.session_state.coach_active = False
if "coach_finished" not in st.session_state:
    st.session_state.coach_finished = False
if "temp_answer" not in st.session_state:
    st.session_state.temp_answer = ""

# Sidebar AI Configuration
st.sidebar.markdown(
    '<h1>🚀 <span class="gradient-text">AI Configuration</span></h1>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

provider = st.sidebar.selectbox(
    "AI Inference Provider",
    ["Offline Fallback", "Google Gemini", "OpenAI GPT"],
    index=0
)

api_key = ""
model_name = ""

if provider == "Google Gemini":
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google AI Studio API key."
    )
    model_name = st.sidebar.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    if not api_key:
        st.sidebar.warning("🔑 Please provide a Gemini API Key to enable AI analysis.")
elif provider == "OpenAI GPT":
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key."
    )
    model_name = st.sidebar.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"])
    if not api_key:
        st.sidebar.warning("🔑 Please provide an OpenAI API Key to enable AI analysis.")
else:
    st.sidebar.info("💡 Running in local NLP mode. No API keys or internet connection required!")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### About the App
    Analyze how well your business plan or pitch deck matches venture capital expectations.
    * **Tab 1**: Alignment Analysis & Scores
    * **Tab 2**: Stateful Mock VC Interview Coach
    """
)

# Convert provider name to backend key
provider_key = "offline"
if provider == "Google Gemini":
    provider_key = "google"
elif provider == "OpenAI GPT":
    provider_key = "openai"

# Main Layout Header
st.markdown(
    '<h1 style="text-align: center; margin-bottom: 5px;">🚀 AI Business Pitch Evaluator</h1>'
    '<p style="text-align: center; font-size: 1.2rem; color: #a5b4fc; margin-bottom: 25px;">'
    'Evaluate your startup against investor expectations & practice with a Mock VC Investor Coach'
    '</p>',
    unsafe_allow_html=True
)

# Create App Config
config = AppConfig(
    openai_api_key=api_key if provider_key == "openai" else None,
    gemini_api_key=api_key if provider_key == "google" else None,
    default_provider=provider_key,
    gemini_model=model_name if provider_key == "google" else None,
    openai_model=model_name if provider_key == "openai" else None
)

def render_gauge_html(score: int, label: str):
    """Generate SVG circular gauge markup."""
    # Stroke calculations: radius=40, circumference = 251.2
    dashoffset = 251.2 - (251.2 * score) / 100

    if score >= 80:
        stroke_color = "#10b981"  # Emerald
    elif score >= 60:
        stroke_color = "#f59e0b"  # Amber
    else:
        stroke_color = "#ef4444"  # Red

    return f"""
    <div class="gauge-wrapper">
        <div class="gauge-container">
            <svg class="gauge-svg" viewBox="0 0 100 100">
                <circle class="gauge-circle-bg" cx="50" cy="50" r="40" />
                <circle class="gauge-circle-progress" cx="50" cy="50" r="40"
                        stroke="{stroke_color}"
                        stroke-dasharray="251.2"
                        stroke-dashoffset="{dashoffset}" />
            </svg>
            <div class="gauge-text">{score}%</div>
        </div>
        <div style="margin-top: 10px; font-family: 'Outfit', sans-serif; font-weight: 600; """ \
        f"""font-size: 0.95rem; color: #e2e8f0; text-align: center;">{label}</div>
    </div>
    """

# Tab setup
tab_eval, tab_coach = st.tabs(["📊 Pitch Deck Evaluator", "🎙️ Mock Investor Coach"])

# --- TAB 1: EVALUATOR ---
with tab_eval:
    st.markdown(
        '<div class="glass-card">'
        '<h3>📤 Upload Document & Thesis</h3>'
        'Upload your startup pitch deck, business plan, or executive summary, and paste the investor requirements.'
        '</div>',
        unsafe_allow_html=True
    )

    col_upload, col_thesis = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Pitch Deck or Business Plan (PDF / DOCX)",
            type=["pdf", "docx"],
            help="Upload the PDF or DOCX file of your pitch deck or business plan."
        )

        # Parse uploaded file
        if uploaded_file:
            if st.session_state.pitch_name != uploaded_file.name:
                try:
                    with st.spinner("Extracting text from document..."):
                        file_bytes = uploaded_file.read()
                        text = extract_text(file_bytes, uploaded_file.name)
                        st.session_state.pitch_text = text
                        st.session_state.pitch_name = uploaded_file.name
                        # Reset evaluations and coach if new file uploaded
                        st.session_state.evaluation_results = None
                        st.session_state.coach_questions = []
                        st.session_state.coach_active = False
                        st.session_state.coach_finished = False
                        st.session_state.coach_evaluations = []
                        st.session_state.coach_current_index = 0
                        st.success(
                            f"✓ Successfully parsed {uploaded_file.name}! "
                            f"({len(text.split())} words extracted)"
                        )
                except PitchEvaluatorError as e:
                    st.error(f"Error parsing document: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during parsing: {str(e)}")
        else:
            # Clear state if file is removed
            st.session_state.pitch_text = ""
            st.session_state.pitch_name = ""
            st.session_state.evaluation_results = None

    with col_thesis:
        st.markdown("**Select a Venture Capital Thesis Preset:**")
        preset = st.selectbox(
            "Preload a Sample VC Thesis",
            [
                "Custom (Write your own)",
                "B2B SaaS Seed Fund (TAM, LTV/CAC, ARR Traction)",
                "Consumer Marketplace Angel (Retention, Viral Growth, Monetization)",
                "DeepTech / AI Pre-Seed VC (PhD founders, IP, High Tech Risk)"
            ]
        )

        default_thesis = ""
        if "B2B SaaS" in preset:
            default_thesis = (
                "We invest in early-stage B2B SaaS companies. Requirements: Clear value proposition, "
                "target market TAM > $1B, customer acquisition strategy (LTV/CAC expectation of 3x+), "
                "some initial traction or customer pipeline (ARR > $100k), and a technical founding team "
                "with domain expertise."
            )
        elif "Consumer Marketplace" in preset:
            default_thesis = (
                "Looking for high-growth consumer apps or marketplace platforms. Requirements: Strong organic loop, "
                "user retention metrics (MoM growth > 15%), large addressable consumer audience, and a clear "
                "monetization strategy. Pre-seed or seed stages. Founder-led sales/growth background is a major plus."
            )
        elif "DeepTech" in preset:
            default_thesis = (
                "Focusing on developer tools, AI infrastructure, or proprietary ML models. Requirements: "
                "Highly technical founders (PhDs or senior AI engineers), defensible IP or proprietary datasets, "
                "and a roadmap demonstrating deep engineering execution capability. Financial metrics are secondary "
                "to technical superiority at this stage."
            )

        investor_thesis = st.text_area(
            "Investor Thesis / VC Requirements",
            value=default_thesis,
            height=150,
            placeholder="E.g., We look for Series A SaaS startups with ARR over $1M..."
        )

    # Evaluate action
    can_evaluate = len(st.session_state.pitch_text) > 0 and len(investor_thesis.strip()) > 0

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn_eval, _ = st.columns([1, 3])
    with col_btn_eval:
        btn_eval = st.button(
            "📊 Run Pitch Alignment Evaluation",
            disabled=not can_evaluate,
            type="primary",
            use_container_width=True,
            help="Evaluate how well the pitch fits the investor thesis requirements."
        )

    if not can_evaluate:
        st.info("ℹ Please upload a pitch document and enter/select an investor thesis to enable evaluation.")

    if btn_eval and can_evaluate:
        # Check API key configuration if LLM selected
        if provider_key in ["google", "openai"] and not api_key:
            st.error(f"❌ Cannot proceed. You selected {provider} but did not input an API Key in the sidebar.")
        else:
            try:
                with st.spinner("Analyzing Pitch Deck Alignment..."):
                    res = evaluate_pitch(
                        st.session_state.pitch_text,
                        investor_thesis,
                        provider=provider_key,
                        api_key=api_key,
                        model_name=model_name
                    )
                    st.session_state.evaluation_results = res
                    st.success("✓ Evaluation completed successfully!")
            except PitchEvaluatorError as e:
                st.error(f"Evaluation Error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected Evaluation Failure: {str(e)}")

    # Display Evaluation Results
    if st.session_state.evaluation_results:
        res = st.session_state.evaluation_results

        st.markdown("---")
        st.markdown(
            '<h2>📊 <span class="gradient-text">Pitch Alignment Scores</span></h2>',
            unsafe_allow_html=True
        )

        # Render Score Gauges
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(render_gauge_html(res["overall_score"], "Overall Alignment"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_gauge_html(res["market_fit_score"], "Market Fit"), unsafe_allow_html=True)
        with col3:
            st.markdown(
                render_gauge_html(res["financial_strategy_score"], "Financial Strategy"),
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                render_gauge_html(res["execution_readiness_score"], "Execution Readiness"),
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Strengths and Gaps
        col_str, col_gaps = st.columns(2)

        with col_str:
            st.markdown(
                '<div class="glass-card">'
                '<h3 style="color: #a7f3d0;">✓ Matched Strengths</h3>'
                'Key areas where your pitch aligns with the investor expectations:'
                '</div>',
                unsafe_allow_html=True
            )
            for strength in res.get("strengths", []):
                st.markdown(
                    f'<div class="strength-badge"><span class="badge-icon">🛡️</span>{strength}</div>',
                    unsafe_allow_html=True
                )

        with col_gaps:
            st.markdown(
                '<div class="glass-card">'
                '<h3 style="color: #fca5a5;">⚠️ Missing Requirements / Gaps</h3>'
                'Key areas that need attention to satisfy the investor thesis:'
                '</div>',
                unsafe_allow_html=True
            )
            for gap in res.get("gaps", []):
                st.markdown(
                    f'<div class="gap-badge"><span class="badge-icon">⚠️</span>{gap}</div>',
                    unsafe_allow_html=True
                )

        # Actionable Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-card">'
            '<h3>💡 Actionable Recommendations</h3>'
            'Apply these detailed changes to improve your pitch deck and increase your fundraising success chances:'
            '</div>',
            unsafe_allow_html=True
        )

        for i, rec in enumerate(res.get("recommendations", []), 1):
            st.markdown(
                f'<div class="glass-card" style="margin-top: 10px; border-left: 4px solid #6366f1 !important; '
                f'padding: 15px !important;">'
                f'<strong>Recommendation #{i}:</strong> {rec}'
                f'</div>',
                unsafe_allow_html=True
            )


# --- TAB 2: MOCK INVESTOR COACH ---
with tab_coach:
    if not st.session_state.pitch_text:
        st.info("📥 Please upload a Pitch Deck / Business Plan document in the first tab to initialize the Coach.")
    else:
        # Setup Start Button if Coach not active
        if not st.session_state.coach_active and not st.session_state.coach_finished:
            st.markdown(
                '<div class="glass-card">'
                '<h3>🎙️ Live Mock VC Interview Session</h3>'
                'Simulate a real partner meeting. We will analyze your pitch text and generate '
                '<strong>5 customized investor questions</strong> testing your market, financials, and '
                'team readiness.<br><br>'
                'You will answer them one-by-one, receive instant score grades and detailed critiques, '
                'and get a final VC scorecard investment decision.'
                '</div>',
                unsafe_allow_html=True
            )

            col_start, _ = st.columns([1, 3])
            with col_start:
                btn_start_coach = st.button(
                    "🎙️ Start VC Interview Simulation",
                    type="primary",
                    use_container_width=True
                )

            if btn_start_coach:
                # Check LLM key
                if provider_key in ["google", "openai"] and not api_key:
                    st.error(f"❌ API Key required. Please provide a key for {provider} in the sidebar.")
                else:
                    with st.spinner("Generating 5 tailored partner questions..."):
                        try:
                            qs = generate_questions(
                                st.session_state.pitch_text,
                                provider=provider_key,
                                api_key=api_key,
                                model_name=model_name
                            )
                            st.session_state.coach_questions = qs
                            st.session_state.coach_active = True
                            st.session_state.coach_current_index = 0
                            st.session_state.coach_evaluations = []
                            st.session_state.coach_answers = {}
                            st.session_state.temp_answer = ""
                            st.rerun()
                        except PitchEvaluatorError as e:
                            st.error(f"Coach Generation Error: {str(e)}")
                        except Exception as e:
                            st.error(f"Unexpected Coach Error: {str(e)}")

        # Stateful Question Form
        elif st.session_state.coach_active and st.session_state.coach_questions:
            idx = st.session_state.coach_current_index
            questions = st.session_state.coach_questions
            q_data = questions[idx]

            # Progress bar
            progress = idx / len(questions)
            st.progress(progress)
            st.write(f"**Question {idx + 1} of {len(questions)}** — *Category: {q_data.get('category')}*")

            # Display Question Card
            st.markdown(
                f'<div class="glass-card question-card">'
                f'<h3 style="color: #818cf8; margin-top: 0;">Question:</h3>'
                f'<p style="font-size: 1.25rem; font-weight: 500; line-height: 1.5; color: #ffffff;">'
                f'{q_data.get("question")}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Create unique key per question index to avoid layout bleed
            ans_input_key = f"user_ans_{idx}"

            user_ans = st.text_area(
                "Your Response:",
                key=ans_input_key,
                height=150,
                placeholder="Type your response to the investor here..."
            )

            col_submit, col_cancel = st.columns([1, 4])

            with col_submit:
                btn_submit_ans = st.button("Submit Answer", type="primary", use_container_width=True)

            with col_cancel:
                if st.button("Cancel Interview", use_container_width=False):
                    st.session_state.coach_active = False
                    st.session_state.coach_finished = False
                    st.session_state.coach_evaluations = []
                    st.session_state.coach_questions = []
                    st.rerun()

            # Handle Submit Answer
            if btn_submit_ans:
                if not user_ans.strip():
                    st.warning("⚠️ Please write an answer before submitting.")
                else:
                    with st.spinner("VC Partner is evaluating your answer..."):
                        try:
                            # Evaluate answer
                            eval_res = evaluate_answer(
                                st.session_state.pitch_text,
                                q_data.get("question"),
                                q_data.get("category"),
                                user_ans,
                                provider=provider_key,
                                api_key=api_key,
                                model_name=model_name
                            )

                            # Cache answer & evaluation result
                            st.session_state.coach_answers[idx] = user_ans
                            st.session_state.coach_evaluations.append({
                                "question": q_data.get("question"),
                                "category": q_data.get("category"),
                                "answer": user_ans,
                                "evaluation": eval_res
                            })
                            st.session_state.temp_answer = user_ans
                            st.rerun()
                        except PitchEvaluatorError as e:
                            st.error(f"Evaluation Error: {str(e)}")
                        except Exception as e:
                            st.error(f"Unexpected Evaluation Error: {str(e)}")

            # Render evaluation for current question if answered
            if idx in st.session_state.coach_answers:
                latest_eval = st.session_state.coach_evaluations[-1]["evaluation"]

                st.markdown("---")
                st.markdown(
                    '<h4>🎙️ <span class="gradient-text">VC Feedback & Grade</span></h4>',
                    unsafe_allow_html=True
                )

                # Show grade card
                col_score, col_feedback = st.columns([1, 3])
                with col_score:
                    st.markdown(render_gauge_html(latest_eval["score"], "Response Score"), unsafe_allow_html=True)
                with col_feedback:
                    st.markdown(
                        f'<div class="glass-card" style="margin-top: 0;">'
                        f'<strong>👍 Key Strengths:</strong><br>{latest_eval.get("strengths")}<br><br>'
                        f'<strong>⚠️ Areas to Improve:</strong><br>{latest_eval.get("weaknesses")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f'<div class="suggested-response-box">'
                    f'<strong style="color: #a5b4fc;">💡 Recommended Model Answer:</strong><br>'
                    f'<p style="font-style: italic; color: #cbd5e1; margin-top: 5px;">'
                    f'"{latest_eval.get("suggested_response")}"</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)

                is_last_q = idx == len(questions) - 1
                btn_lbl = "View Final VC Scorecard" if is_last_q else "Next Question"

                col_next, _ = st.columns([1, 3])
                with col_next:
                    if st.button(btn_lbl, type="primary", use_container_width=True):
                        if is_last_q:
                            st.session_state.coach_active = False
                            st.session_state.coach_finished = True
                        else:
                            st.session_state.coach_current_index += 1
                        st.rerun()

        # Final Scorecard / Report Card
        elif st.session_state.coach_finished:
            st.balloons()

            # Generate final performance report
            evals = [item["evaluation"] for item in st.session_state.coach_evaluations]
            summary = generate_performance_summary(evals)

            is_good_investment = summary["average_score"] >= 70
            status_css = "investment-grade" if is_good_investment else "incubation-grade"
            status_icon = "🟢" if is_good_investment else "🔴"

            st.markdown(
                '<h2 style="text-align: center;"><span class="gradient-text">VC Mock Interview Scorecard</span></h2>',
                unsafe_allow_html=True
            )

            col_final_score, col_status = st.columns([1, 2])
            with col_final_score:
                st.markdown(
                    render_gauge_html(summary["average_score"], "Partner Consensus Score"),
                    unsafe_allow_html=True
                )
            with col_status:
                st.markdown(
                    f'<div class="glass-card {status_css}">'
                    f'<h3>{status_icon} VC Decision: {summary.get("tier")}</h3>'
                    f'<p style="line-height: 1.6;"><strong>Overall Partner Feedback:</strong><br>'
                    f'{summary.get("summary")}</p>'
                    f'<p style="line-height: 1.6;"><strong>Actionable Preparation Next Steps:</strong><br>'
                    f'{summary.get("recommendation")}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3>📋 Question-by-Question Breakdown</h3>", unsafe_allow_html=True)

            for i, item in enumerate(st.session_state.coach_evaluations, 1):
                with st.expander(f"Question #{i}: {item['question'][:60]}... — Score: {item['evaluation']['score']}%"):
                    st.markdown(f"**Full Question:** {item['question']}")
                    st.markdown(f"**Your Answer:** *\"{item['answer']}\"*")
                    st.markdown(f"**VC Strengths:** {item['evaluation']['strengths']}")
                    st.markdown(f"**VC Weaknesses:** {item['evaluation']['weaknesses']}")
                    st.markdown(f"**Ideal Response:** *\"{item['evaluation']['suggested_response']}\"*")

            st.markdown("<br>", unsafe_allow_html=True)

            col_reset, _ = st.columns([1, 3])
            with col_reset:
                if st.button("🎙️ Restart VC Interview Session", type="primary", use_container_width=True):
                    st.session_state.coach_active = False
                    st.session_state.coach_finished = False
                    st.session_state.coach_evaluations = []
                    st.session_state.coach_questions = []
                    st.session_state.coach_current_index = 0
                    st.rerun()
