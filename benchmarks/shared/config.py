"""Configuration: LM setup, model presets, and path helpers."""

import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_REPO = PROJECT_ROOT.parent / "humanities_data_benchmark"

# Model presets: short name -> DSPy model string
MODEL_PRESETS = {
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    # Google Gemini
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
    # Anthropic Claude
    "claude-sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "claude-haiku": "anthropic/claude-haiku-3-5-20241022",
    # OpenRouter (prefix any model with openrouter/)
    "or-gemini-2.5-pro": "openrouter/google/gemini-2.5-pro",
    "or-claude-sonnet": "openrouter/anthropic/claude-sonnet-4-5-20250929",
    "or-gpt-4o": "openrouter/openai/gpt-4o",
}

DEFAULT_MODEL = "gpt-4o"


def resolve_model(model: str) -> str:
    """Resolve a short preset name or pass through a full model string."""
    return MODEL_PRESETS.get(model, model)


def results_dir(benchmark: str) -> Path:
    """Return the results directory for a given benchmark."""
    return RESULTS_DIR / benchmark


def configure_dspy(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    """Configure DSPy with the specified LM.

    Args:
        model: Either a preset name (e.g. "gemini-2.5-pro") or a full
               DSPy model string (e.g. "openai/gpt-4o").
        temperature: Sampling temperature.
    """
    model_id = resolve_model(model)

    # Set provider-specific API key env vars that litellm expects
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your-gemini-key-here":
        os.environ["GEMINI_API_KEY"] = gemini_key

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key and openrouter_key != "your-openrouter-key-here":
        os.environ["OPENROUTER_API_KEY"] = openrouter_key

    lm = dspy.LM(model_id, temperature=temperature)
    dspy.configure(lm=lm)
    return lm
