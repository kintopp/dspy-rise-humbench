#!/usr/bin/env python3
"""Check rate limits for configured API providers.

Makes lightweight API calls to each provider and reports rate limit headers.
Helps decide which provider to use for optimization runs.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SEPARATOR = "-" * 60


def check_openai():
    """Check OpenAI rate limits via a lightweight models.list() call."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("your-"):
        print("  OPENAI_API_KEY: not configured")
        return None

    print("  OPENAI_API_KEY: configured")
    try:
        import httpx

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 1,
            },
            timeout=30,
        )
        if resp.status_code == 200 or resp.status_code == 429:
            limits = {}
            for header in resp.headers:
                h = header.lower()
                if "ratelimit" in h or "rate-limit" in h:
                    limits[header] = resp.headers[header]
            if limits:
                print("  Rate limit headers:")
                for k, v in sorted(limits.items()):
                    print(f"    {k}: {v}")
                return limits
            else:
                print("  No rate limit headers found in response")
                return {}
        else:
            print(f"  API returned status {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def check_anthropic():
    """Check Anthropic rate limits via a minimal messages.create() call."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your-"):
        print("  ANTHROPIC_API_KEY: not configured")
        return None

    print("  ANTHROPIC_API_KEY: configured")
    try:
        import httpx

        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30,
        )
        limits = {}
        for header in resp.headers:
            h = header.lower()
            if "ratelimit" in h or "rate-limit" in h:
                limits[header] = resp.headers[header]
        if limits:
            print("  Rate limit headers:")
            for k, v in sorted(limits.items()):
                print(f"    {k}: {v}")
            return limits
        else:
            print(f"  Status {resp.status_code}, no rate limit headers found")
            return {}
    except Exception as e:
        print(f"  Error: {e}")
        return None


def check_gemini():
    """Check Google Gemini API access and infer rate limits."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key.startswith("your-"):
        print("  GEMINI_API_KEY: not configured")
        return None

    print("  GEMINI_API_KEY: configured")
    try:
        import httpx

        # Use the models.list endpoint to verify the key works
        resp = httpx.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
            timeout=30,
        )
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            gemini_models = [m["name"] for m in models if "gemini" in m.get("name", "").lower()]
            print(f"  API key valid. {len(gemini_models)} Gemini models available.")
            # Now make a minimal generation call to check for rate limit headers
            gen_resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                json={"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}},
                timeout=30,
            )
            limits = {}
            for header in gen_resp.headers:
                h = header.lower()
                if "ratelimit" in h or "rate-limit" in h or "quota" in h:
                    limits[header] = gen_resp.headers[header]
            if limits:
                print("  Rate limit headers:")
                for k, v in sorted(limits.items()):
                    print(f"    {k}: {v}")
            else:
                print("  No rate limit headers in response (Gemini uses plan-based limits)")
                print("  Free tier: 15 RPM, 1M TPM, 1500 RPD")
                print("  Pay-as-you-go: 2000 RPM, 4M TPM")
            return limits
        else:
            print(f"  API returned status {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def check_openrouter():
    """Check OpenRouter rate limits and credit balance."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key or key.startswith("your-"):
        print("  OPENROUTER_API_KEY: not configured")
        return None

    print("  OPENROUTER_API_KEY: configured")
    try:
        import httpx

        # Check auth/key endpoint for credit info
        resp = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            print(f"  Label: {data.get('label', 'N/A')}")
            limit = data.get("limit")
            usage = data.get("usage")
            if limit is not None:
                print(f"  Credit limit: ${limit:.2f}")
            if usage is not None:
                print(f"  Usage: ${usage:.4f}")
            if limit is not None and usage is not None:
                print(f"  Remaining: ${limit - usage:.2f}")
            rate_limit = data.get("rate_limit", {})
            if rate_limit:
                print(f"  Rate limit: {rate_limit}")
            print("  OpenRouter: rate limits depend on model + credits (typically 200 RPM free, more with credits)")
            return data
        else:
            print(f"  API returned status {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("=" * 60)
    print("PROVIDER RATE LIMIT CHECK")
    print("=" * 60)

    results = {}

    print(f"\n{SEPARATOR}")
    print("1. OpenAI")
    print(SEPARATOR)
    results["openai"] = check_openai()

    print(f"\n{SEPARATOR}")
    print("2. Anthropic")
    print(SEPARATOR)
    results["anthropic"] = check_anthropic()

    print(f"\n{SEPARATOR}")
    print("3. Google Gemini")
    print(SEPARATOR)
    results["gemini"] = check_gemini()

    print(f"\n{SEPARATOR}")
    print("4. OpenRouter")
    print(SEPARATOR)
    results["openrouter"] = check_openrouter()

    # Summary / recommendation
    print(f"\n{'=' * 60}")
    print("RECOMMENDATION")
    print("=" * 60)
    configured = [k for k, v in results.items() if v is not None]
    if not configured:
        print("  No providers configured! Add API keys to .env")
    else:
        print(f"  Configured providers: {', '.join(configured)}")
        print()
        print("  For MIPROv2 optimization (many parallel calls with images):")
        print("  - Gemini 2.5 Pro/Flash: Best for high throughput (generous rate limits)")
        print("  - OpenRouter: Good fallback with automatic retries across providers")
        print("  - Anthropic Claude: Good quality but check token limits above")
        print("  - OpenAI gpt-4o: Check TPM limit above — 30k TPM is too low for optimization")


if __name__ == "__main__":
    main()
