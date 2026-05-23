"""Demo: A/B testing two prompt versions and cross-model comparison.

Phase 7 | A/B Testing
"""

from __future__ import annotations

import argparse
import asyncio
import os

from prompt_versioning.examples.ab_tester.tester import ABTester, CrossModelABTester
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry
from prompt_versioning.utils.llm_client import LLMClient, MultiModelClient
from prompt_versioning.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def main() -> None:
    """Register two prompt versions and run A/B + cross-model experiments."""
    parser = argparse.ArgumentParser(description="A/B Testing demo")
    parser.add_argument("--live", action="store_true", help="Call the API for real")
    args = parser.parse_args()

    live = args.live and bool(os.environ.get("OPENAI_API_KEY"))

    registry = PromptRegistry()
    registry.register(
        "summarize",
        "Summarize the following text in 2-3 sentences.",
        "Simple summarizer",
    )
    registry.register(
        "summarize",
        (
            "You are a professional summarizer. Given a text, produce a concise "
            "summary that captures the key points in exactly 2 sentences. Use "
            "clear, simple language."
        ),
        "Professional summarizer with strict 2-sentence constraint",
    )

    queries = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It uses algorithms to identify patterns and make decisions with minimal human intervention. Applications range from image recognition to natural language processing.",
        "Cloud computing delivers computing services over the internet. These services include servers, storage, databases, networking, and software. Organizations use cloud computing to reduce costs and increase scalability.",
        "Cybersecurity involves protecting systems and networks from digital attacks. These attacks typically aim to access, change, or destroy sensitive information. Effective cybersecurity requires multiple layers of protection.",
        "Agile methodology is an iterative approach to software development. Teams deliver work in small increments called sprints. This allows for rapid feedback and adaptation to changing requirements.",
        "Blockchain is a decentralized digital ledger that records transactions. Each block contains a cryptographic hash of the previous block. This makes the chain tamper-resistant and transparent.",
        "DevOps combines software development and IT operations. It aims to shorten the development lifecycle while delivering features and fixes frequently. Automation and continuous integration are key practices.",
    ]

    # --- Prompt-vs-Prompt A/B Test ---
    print("=" * 60)
    print("Prompt-vs-Prompt A/B Testing Demo")
    print("=" * 60)
    print("Prompt: summarize (v1 vs v2)")
    print(f"Queries: {len(queries)}")

    if live:
        client = LLMClient()
        tester = ABTester(registry, "summarize", 1, 2, client=client)
        experiment = await tester.run_experiment(queries)
        tester.print_report(experiment)
    else:
        print("(Requires OPENAI_API_KEY \u2014 showing expected format)\n")
        print("Expected report format:")
        print("=" * 60)
        print(f"{'Metric':<20} {'Variant A':>15} {'Variant B':>15}")
        print("-" * 60)
        print(f"{'Version':<20} {'v1':>15} {'v2':>15}")
        print(f"{'Queries':<20} {'~3':>15} {'~3':>15}")
        print(f"{'Avg Quality':<20} {'?.??':>15} {'?.??':>15}")
        print(f"{'Avg Latency (ms)':<20} {'?.?':>15} {'?.?':>15}")
        print("-" * 60)
        print("Winner: (determined by avg quality score)")
        print("=" * 60)

    # --- Cross-Model Comparison ---
    print("\n" + "=" * 70)
    print("Cross-Model Comparison Demo")
    print("=" * 70)
    print("Same prompt (summarize v2) tested across Claude, GPT-4o, Gemini")
    print(f"Queries: {len(queries)}")

    if live:
        multi_client = MultiModelClient()
        cross_tester = CrossModelABTester(registry, "summarize", 2, multi_client)
        comparison = await cross_tester.run_model_comparison(queries)
        cross_tester.print_comparison(comparison)
    else:
        print(
            "(Requires API keys for all three providers \u2014 showing expected format)\n"
        )
        print("Expected cross-model comparison report:")
        print("=" * 70)
        print(
            f"{'Model':<15} {'Queries':>10} {'Avg Quality':>13} {'Avg Latency':>13} {'Avg Tokens':>12}"
        )
        print("-" * 70)
        print(f"{'claude':<15} {'6':>10} {'4.20':>13} {'850.0ms':>13} {'35.0':>12}")
        print(f"{'gpt-4o':<15} {'6':>10} {'4.35':>13} {'620.0ms':>13} {'32.0':>12}")
        print(f"{'gemini':<15} {'6':>10} {'4.10':>13} {'480.0ms':>13} {'28.0':>12}")
        print("-" * 70)
        print("Winner: gpt-4o")
        print("=" * 70)
        print("\n(Run with --live and API keys set to see real results)")


if __name__ == "__main__":
    asyncio.run(main())
