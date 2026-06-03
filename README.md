# AI Business Pitch Evaluator & Mock Investor Coach

A premium, containerized full-stack Streamlit web application that evaluates startup pitch decks (PDF/DOCX) against investor thesis requirements using NLP. It also features a stateful **Mock Investor Coach** which generates customized VC partner questions, evaluates founder responses, and generates investor partner scorecards.

---

## 💡 Key Features

1. **AI Business Pitch Evaluator**:
   * Analyzes pitch deck uploads (PDF/DOCX) against a venture capitalist requirements text.
   * Generates four key progress metrics: **Overall Alignment**, **Market Fit**, **Financial Strategy**, and **Execution Readiness**.
   * Identifies matched strengths, structural gaps, and lists detailed actionable recommendations.
   * Preloaded with VC presets (B2B SaaS, Consumer Tech, DeepTech) for instant testing.
2. **Stateful Mock VC Coach**:
   * Dynamically generates 5 tough startup partner questions based on the pitch text.
   * Guides the founder through a sequential, state-managed partner meeting.
   * Grades each answer with scores, strengths, weaknesses, and a suggested model response.
   * Concludes with an **Investment Decision Scorecard** (Consensus Score, Tier, and actionable next steps).
3. **Hybrid AI Engine**:
   * Supports **Google Gemini** (via `gemini-1.5-flash` or `gemini-1.5-pro`).
   * Supports **OpenAI GPT** (via `gpt-4o-mini` or `gpt-4o`).
   * Supports a robust local **Offline Fallback Mode** utilizing spaCy NLP keyword chunk matching and Jaccard similarity metrics (works entirely offline without API keys).
4. **Premium Dark UI**:
   * Sleek dark space theme accented with glowing indigo/cyan gradients.
   * Glassmorphism cards with blurred backdrops and subtle border hover animations.
   * Responsive layout and custom animated SVG progress gauges.

---

## 🛠️ Project Architecture & Folder Structure

```
AI-Business-Pitch-Evaluator-Mock-Investor-Coach/
│
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD test automation
│
├── app/
│   ├── main.py                 # Streamlit frontend & UI view manager
│   ├── config.py               # Settings manager (OpenAI/Gemini/Offline keys)
│   ├── assets/
│   │   └── custom.css          # Glassmorphism cards & score chart animations
│   └── backend/
│       ├── document_parser.py  # Extracts PDF & DOCX text in-memory
│       ├── nlp_extractor.py    # spaCy entities, noun chunks, similarity
│       ├── scoring.py          # Local & LLM-based scoring calculators
│       ├── coach.py            # Question generator & answer evaluator
│       └── exceptions.py       # Centralized exception classes
│
├── tests/
│   ├── test_document_parser.py # Document extraction tests
│   ├── test_nlp_extractor.py    # spaCy model tests
│   ├── test_scoring.py          # Local/LLM scoring checks
│   ├── test_coach.py            # VC partner Q&A state checks
│   └── test_config.py           # Configuration manager checks
│
├── Dockerfile                  # Multi-stage python image
├── docker-compose.yml          # Container configuration (runs on Port 8502)
├── requirements.txt            # Python dependencies
├── .dockerignore               # Ignores local venvs to speed up builds
├── pylintrc                    # Pylint configuration
└── .env.example                # Sample environment file
```

---

## 🛡️ Robust Centralized Exception Handling

This project uses a dedicated, hierarchy-based exception structure defined in `app/backend/exceptions.py`:
* **`PitchEvaluatorError`**: Base exception class.
  * **`DocumentParsingError`**: Raised if text extraction fails, file is corrupted, or has an unsupported format.
  * **`NLPProcessingError`**: Raised if spaCy model loading or similarity calculation fails.
  * **`LLMIntegrationError`**: Raised if Gemini/OpenAI API requests fail (e.g. invalid keys or timeouts).
  * **`ConfigurationError`**: Raised if environment settings are invalid.

### UI Safety Implementation
Errors are caught at the controller level in `app/main.py` using try-except blocks:
```python
try:
    with st.spinner("VC Partner is evaluating your answer..."):
        eval_res = evaluate_answer(...)
except PitchEvaluatorError as e:
    st.error(f"Evaluation Error: {str(e)}")
except Exception as e:
    st.error(f"Unexpected Evaluation Error: {str(e)}")
```
This design ensures that **the web application never crashes** when experiencing API network errors or invalid files; instead, it catches exceptions gracefully and outputs beautiful, diagnostic alerts to the user.

---

## 🧪 Testing & Code Quality

### 1. Unit Tests
We have a comprehensive test suite in `tests/` covering configuration overrides, document stream mock parsers, spaCy pipelines, and Q&A grading.

**Run tests locally**:
```bash
.venv\Scripts\python -m pytest --cov=app tests/
```
* **Status**: **100% PASSING (23/23 tests)**
* **Coverage**: Core modules achieve up to **92%** coverage.

### 2. Code Quality (Linter)
The project adheres strictly to PEP8 standards.
* **Status**: **🏆 10.00 / 10.00** rating from `pylint` (`pylint app/` with PYTHONPATH=.).

---

## ⚡ Setup & Deployment

### Run via Docker Compose (Port 8502)
1. Build and boot the container:
   ```bash
   docker-compose up --build
   ```
2. Navigate to **`http://localhost:8502`**.

### Run Locally (Python 3.11/3.12/3.13)
1. Install dependencies & spaCy model:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Start Streamlit:
   ```bash
   streamlit run app/main.py
   ```
