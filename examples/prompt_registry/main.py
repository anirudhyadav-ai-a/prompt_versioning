"""Demo: prompt registry with versioning, diff, and rollback.

Phase 7 | Prompt Registry
"""

from __future__ import annotations

import argparse
import asyncio
import os

from prompt_versioning.examples.prompt_registry.registry import PromptRegistry
from prompt_versioning.utils.llm_client import LLMClient
from prompt_versioning.utils.logging_utils import get_logger

logger = get_logger(__name__)


async def main() -> None:
    """Walk through the full prompt registry lifecycle."""
    parser = argparse.ArgumentParser(description="Prompt Registry demo")
    parser.add_argument("--live", action="store_true", help="Call the API for real")
    args = parser.parse_args()

    live = args.live and bool(os.environ.get("OPENAI_API_KEY"))
    client = LLMClient() if live else None
    registry = PromptRegistry(client=client)

    # v1
    v1 = registry.register(
        "classify-intent",
        "Classify the user's intent into one of: greeting, question, complaint, other.",
        "Initial version",
    )
    print(f"Registered: {v1.name} v{v1.version}")

    # Duplicate — should skip
    dup = registry.register(
        "classify-intent",
        "Classify the user's intent into one of: greeting, question, complaint, other.",
    )
    print(f"Duplicate check: returned v{dup.version} (expected v1)")

    # v2
    v2 = registry.register(
        "classify-intent",
        (
            "You are an intent classifier. Given a user message, classify it into "
            "exactly one category: greeting, question, complaint, feedback, other. "
            "Respond with only the category name."
        ),
        "Added feedback category, clearer instructions",
    )
    print(f"Registered: {v2.name} v{v2.version}")

    # v3
    v3 = registry.register(
        "classify-intent",
        (
            "You are an intent classifier. Given a user message, classify it into "
            "exactly one category: greeting, question, complaint, feedback, "
            'escalation, other. Respond with JSON: {"intent": "...", "confidence": 0.X}'
        ),
        "Added escalation category, JSON output",
    )
    print(f"Registered: {v3.name} v{v3.version}")

    # Print history
    print("\n--- Version History ---")
    for pv in registry.get_history("classify-intent"):
        print(
            f"  v{pv.version}: {pv.description or '(no description)'} [{pv.content_hash[:12]}...]"
        )

    # Text diff v1 vs v3
    diff = registry.text_diff("classify-intent", 1, 3)
    print("\n--- Text Diff (v1 \u2192 v3) ---")
    for line in diff["removed_lines"]:
        print(f"  - {line}")
    for line in diff["added_lines"]:
        print(f"  + {line}")

    # Semantic diff
    print("\n--- Semantic Diff (v1 vs v3) ---")
    if live:
        sim = await registry.semantic_diff("classify-intent", 1, 3)
        print(f"  Cosine similarity: {sim:.4f}")
    else:
        print("  (Skipped \u2014 requires OPENAI_API_KEY for embedding)")

    # Rollback to v2
    v4 = registry.rollback("classify-intent", 2)
    print("\n--- Rollback to v2 ---")
    print(f"  Created: v{v4.version} with content from v2")
    print(f"  Description: {v4.description}")

    # Final history
    print("\n--- Final History ---")
    for pv in registry.get_history("classify-intent"):
        print(f"  v{pv.version}: {pv.description or '(no description)'}")


if __name__ == "__main__":
    asyncio.run(main())
