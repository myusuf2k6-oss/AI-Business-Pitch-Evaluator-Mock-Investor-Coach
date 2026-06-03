"""Tests for the AppConfig class in config.py."""

import os
from unittest import mock
from app.config import AppConfig

def test_default_config():
    """Verify that default settings apply when no env/args are provided."""
    with mock.patch.dict(os.environ, {}, clear=True):
        cfg = AppConfig()
        assert cfg.openai_api_key is None
        assert cfg.gemini_api_key is None
        assert cfg.default_provider == "offline"
        assert cfg.has_openai is False
        assert cfg.has_gemini is False

def test_env_var_override():
    """Verify that environment variables are read correctly."""
    env_mock = {
        "OPENAI_API_KEY": "sk-test-openai",
        "GEMINI_API_KEY": "gemini-test-key",
        "DEFAULT_PROVIDER": "google",
        "GEMINI_MODEL": "gemini-1.5-pro",
        "OPENAI_MODEL": "gpt-4o"
    }
    with mock.patch.dict(os.environ, env_mock, clear=True):
        cfg = AppConfig()
        assert cfg.openai_api_key == "sk-test-openai"
        assert cfg.gemini_api_key == "gemini-test-key"
        assert cfg.default_provider == "google"
        assert cfg.gemini_model == "gemini-1.5-pro"
        assert cfg.openai_model == "gpt-4o"
        assert cfg.has_openai is True
        assert cfg.has_gemini is True

def test_argument_override():
    """Verify that arguments override environment variables."""
    env_mock = {
        "OPENAI_API_KEY": "sk-test-openai",
        "GEMINI_API_KEY": "gemini-test-key"
    }
    with mock.patch.dict(os.environ, env_mock, clear=True):
        cfg = AppConfig(
            openai_api_key="sk-override",
            gemini_api_key="gemini-override",
            default_provider="openai"
        )
        assert cfg.openai_api_key == "sk-override"
        assert cfg.gemini_api_key == "gemini-override"
        assert cfg.default_provider == "openai"
