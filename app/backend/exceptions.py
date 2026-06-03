"""Centralized exception handling for the AI Business Pitch Evaluator & Mock Investor Coach."""

class PitchEvaluatorError(Exception):
    """Base exception for all errors in the application."""

class DocumentParsingError(PitchEvaluatorError):
    """Raised when document parsing fails (e.g. invalid format or corrupted file)."""

class NLPProcessingError(PitchEvaluatorError):
    """Raised when NLP processing or entity/concept extraction fails."""

class LLMIntegrationError(PitchEvaluatorError):
    """Raised when an API call to Google Gemini or OpenAI fails."""

class ConfigurationError(PitchEvaluatorError):
    """Raised when application configuration or environment variables are missing/invalid."""
