"""Token counting and cost estimation utilities."""

from __future__ import annotations


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken for OpenAI models, word-based fallback for others."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except (KeyError, ImportError):
        return int(len(text.split()) * 1.3)


# Pricing per 1M tokens (input/output) as of 2026
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def estimate_cost(
    input_tokens: int, output_tokens: int, model: str = "gpt-4o"
) -> float:
    """Estimate cost in USD for a single request."""
    pricing = MODEL_PRICING.get(model, {"input": 2.50, "output": 10.00})
    return (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000
