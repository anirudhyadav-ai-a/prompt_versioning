"""Demo: drift detection on a prompt test suite (single-model and cross-model).

Phase 7 | Drift Detection
"""

from __future__ import annotations

import argparse
import asyncio
import os

from prompt_versioning.examples.drift_detector.detector import (
    CrossModelDriftDetector,
    DriftDetector,
    DriftTestCase,
)
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry
from prompt_versioning.utils.llm_client import LLMClient, MultiModelClient
from prompt_versioning.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def main() -> None:
    """Register a prompt, define test cases, and run drift detection."""
    parser = argparse.ArgumentParser(description="Drift Detection demo")
    parser.add_argument("--live", action="store_true", help="Call the API for real")
    args = parser.parse_args()

    live = args.live and bool(os.environ.get("OPENAI_API_KEY"))

    registry = PromptRegistry()
    registry.register(
        "classify-intent",
        (
            "You are an intent classifier. Given a user message, classify it into "
            "exactly one category: greeting, question, complaint, feedback, "
            "escalation, other. Respond with only the category name."
        ),
        "Production prompt",
    )

    test_cases = [
        DriftTestCase("Hello there!", "greeting"),
        DriftTestCase("What are your hours?", "question"),
        DriftTestCase("This product is terrible", "complaint"),
        DriftTestCase("I love your service", "feedback"),
    ]

    # --- Single-model drift detection ---
    print("=" * 60)
    print("Single-Model Drift Detection Demo")
    print("=" * 60)
    print(f"Prompt: classify-intent (v{registry.get_current('classify-intent').version})")  # type: ignore[union-attr]
    print(f"Test cases: {len(test_cases)}")

    if live:
        client = LLMClient()
        detector = DriftDetector(registry, client=client)
        reports = await detector.run_test_suite("classify-intent", test_cases)
        summary = DriftDetector.summarize(reports)
        print("\nResults:")
        for r in reports:
            status = "DRIFT" if r.drifted else "PASS"
            print(f"  [{status}] {r.test_input:<35} sim={r.similarity_score:.3f}")
        print(
            f"\nSummary: total={summary['total_tests']}, passed={summary['passed']}, "
            f"drifted={summary['drifted']}, drift_rate={summary['drift_rate']:.1%}, "
            f"avg_similarity={summary['avg_similarity']:.3f}"
        )
    else:
        print("(Requires OPENAI_API_KEY \u2014 skipping live run in demo mode)\n")
        print("Expected output with a live API key:")
        for tc in test_cases:
            print(f"  Input: {tc.input_text:<35} Expected: {tc.expected_output}")
        print("\nSummary: total=4, passed=?, drifted=?, drift_rate=?, avg_similarity=?")

    # --- Cross-model drift detection ---
    print("\n" + "=" * 60)
    print("Cross-Model Drift Detection Demo")
    print("=" * 60)
    print(
        "Models: Claude (claude-sonnet-4-20250514), GPT-4o, Gemini (gemini-2.0-flash)"
    )
    print(f"Test cases: {len(test_cases)}")

    if live:
        multi_client = MultiModelClient()
        cross_detector = CrossModelDriftDetector(registry, multi_client)
        cross_results = await cross_detector.run_cross_model_suite(
            "classify-intent", test_cases
        )
        summary = cross_detector.cross_model_summary(cross_results)
        print()
        print("-" * 60)
        print(f"{'Model':<15} {'Drift Rate':>12} {'Avg Similarity':>16} {'Tests':>8}")
        print("-" * 60)
        for label, stats in summary["per_model"].items():
            print(
                f"{label:<15} {stats['drift_rate']:>11.1%} {stats['avg_similarity']:>16.2f} "
                f"{stats['total_tests']:>8}"
            )
        print("-" * 60)
        print("\nCross-model summary:")
        print(f"  Most stable model:     {summary['most_stable_model']}")
        print(f"  Most drifted model:    {summary['most_drifted_model']}")
        print(f"  Cross-model agreement: {summary['cross_model_agreement']:.1%}")
    else:
        print(
            "(Requires API keys for all three providers \u2014 showing expected format)\n"
        )
        print("Expected cross-model drift report:")
        print("-" * 60)
        print(f"{'Model':<15} {'Drift Rate':>12} {'Avg Similarity':>16} {'Tests':>8}")
        print("-" * 60)
        print(f"{'claude':<15} {'2.0%':>12} {'0.95':>16} {'4':>8}")
        print(f"{'gpt-4o':<15} {'0.0%':>12} {'0.98':>16} {'4':>8}")
        print(f"{'gemini':<15} {'5.0%':>12} {'0.90':>16} {'4':>8}")
        print("-" * 60)
        print("\nCross-model summary:")
        print("  Most stable model:     gpt-4o")
        print("  Most drifted model:    gemini")
        print("  Cross-model agreement: 75.0%")
        print("(Run with --live and API keys set to see real results)")


if __name__ == "__main__":
    asyncio.run(main())
