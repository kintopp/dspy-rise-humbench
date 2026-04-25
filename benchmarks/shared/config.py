"""Configuration: LM setup, model presets, and path helpers."""

import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

# Load env vars from both project-local .env (if any) and the user's ~/.env.
# The RISE_-prefixed keys live in ~/.env; the project .env (if present) may
# override individual values. `override=False` preserves anything already set
# in the shell.
load_dotenv()  # cwd .env (project-local, highest priority)
load_dotenv(Path.home() / ".env", override=False)  # user-level fallback


# RISE-prefixed keys in ~/.env are this project's billing-tracked keys; map
# them onto the standard env names litellm reads. RISE_ values take precedence
# over any standard names also present in the env so that personal keys never
# accidentally bill against personal accounts when running this project.
# RISE_GOOGLE_STUDIO_API maps to BOTH GOOGLE_API_KEY and GEMINI_API_KEY because
# litellm checks GEMINI_API_KEY first for `gemini/*` model strings, falling back
# to GOOGLE_API_KEY in some versions.
_RISE_KEY_MAP = {
    "RISE_OPENAI": "OPENAI_API_KEY",
    "RISE_ANTHROPIC": "ANTHROPIC_API_KEY",
    "RISE_MISTRAL_API_KEY": "MISTRAL_API_KEY",
    "RISE_GOOGLE_STUDIO_API": "GOOGLE_API_KEY",
}
# Also wire the Google studio key into GEMINI_API_KEY (separate name; litellm uses both).
if os.environ.get("RISE_GOOGLE_STUDIO_API"):
    os.environ["GEMINI_API_KEY"] = os.environ["RISE_GOOGLE_STUDIO_API"]
for _src, _dst in _RISE_KEY_MAP.items():
    if os.environ.get(_src):
        os.environ[_dst] = os.environ[_src]

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
    "gpt-5.2": "openai/gpt-5.2-2025-12-11",
    # Google Gemini
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
    "gemini-3.1-pro-preview": "gemini/gemini-3.1-pro-preview",
    "gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview": "gemini/gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",  # deprecated, shuts down 2026-06-01
    # Anthropic Claude
    "claude-sonnet": "anthropic/claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-haiku": "anthropic/claude-haiku-3-5-20241022",
    # OpenRouter (prefix any model with openrouter/)
    "or-gemini-2.5-pro": "openrouter/google/gemini-2.5-pro",
    "or-claude-sonnet": "openrouter/anthropic/claude-sonnet-4-5-20250929",
    "or-gpt-4o": "openrouter/openai/gpt-4o",
}

DEFAULT_MODEL = "gemini-2.5-flash"


def resolve_model(model: str) -> str:
    """Resolve a short preset name or pass through a full model string."""
    return MODEL_PRESETS.get(model, model)


def results_dir(benchmark: str) -> Path:
    """Return the results directory for a given benchmark."""
    return RESULTS_DIR / benchmark


def configure_dspy(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> dspy.LM:
    """Configure DSPy with the specified LM.

    Args:
        model: Either a preset name (e.g. "gemini-2.5-pro") or a full
               DSPy model string (e.g. "openai/gpt-4o").
        temperature: Sampling temperature.
    """
    model_id = resolve_model(model)

    lm = dspy.LM(model_id, temperature=temperature)
    dspy.configure(lm=lm, track_usage=True)
    return lm
