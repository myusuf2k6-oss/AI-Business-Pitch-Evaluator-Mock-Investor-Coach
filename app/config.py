"""Configuration manager for the AI Business Pitch Evaluator application."""

# pylint: disable=too-many-positional-arguments

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class AppConfig:
    """Application configuration container."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        default_provider: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None
    ):
        # API Keys priority: 1. Passed explicitly (from UI), 2. Environment variable
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.default_provider = default_provider or os.getenv("DEFAULT_PROVIDER") or "offline"
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL") or "gemini-1.5-flash"
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is set."""
        return bool(self.openai_api_key)

    @property
    def has_gemini(self) -> bool:
        """Check if Gemini API key is set."""
        return bool(self.gemini_api_key)
