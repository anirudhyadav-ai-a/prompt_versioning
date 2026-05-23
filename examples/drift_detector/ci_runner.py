"""CI runner for drift detection — loads test cases and outputs JSON report.

Phase 7 | Drift Detection — CI/CD Integration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from prompt_versioning.examples.drift_detector.detector import (
    DriftDetector,
    DriftTestCase,
)
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry
from prompt_versioning.utils.llm_client import LLMClient
from prompt_versioning.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_TEST_CASES = Path(__file__).parent / "test_cases.json"
DEFAULT_THRESHOLD = 0.10


PROMPTS_CONFIG_PATH = Path(__file__).parent / "prompts_config.json"


async def run(
    test_cases_path: Path,
    drift_threshold: float,
    output_path: Path,
    prompts_config_path: Path = PROMPTS_CONFIG_PATH,
) -> bool:
    """Run drift detection and write results to *output_path*.

    Prompts are loaded from *prompts_config_path* (a JSON file mapping
    prompt name → content) rather than being hardcoded here. This ensures
    the CI runner tests the same prompt that is actually deployed, not a
    stale inline copy. See prompts_config.json for the expected format.

    Returns True if drift rate is within threshold, False otherwise.
    """
    with open(test_cases_path) as f:
        raw = json.load(f)
    cases = [
        DriftTestCase(input_text=tc["input"], expected_output=tc["expected"])
        for tc in raw
    ]

    registry = PromptRegistry()
    if prompts_config_path.exists():
        with open(prompts_config_path) as f:
            prompts_config = json.load(f)
        for name, entry in prompts_config.items():
            registry.register(name, entry["content"], entry.get("description", ""))
    else:
        # Fallback: register the default classify-intent prompt so the runner
        # works out-of-the-box. Replace prompts_config.json with your own
        # production prompts in CI.
        logger.warning(
            "prompts_config.json not found at %s — using built-in fallback prompt. "
            "Create this file to test your actual production prompts.",
            prompts_config_path,
        )
        registry.register(
            "classify-intent",
            (
                "You are an intent classifier. Given a user message, classify it into "
                "exactly one category: greeting, question, complaint, feedback, "
                "escalation, other. Respond with only the category name."
            ),
            "Default fallback prompt — replace via prompts_config.json",
        )

    client = LLMClient()
    detector = DriftDetector(registry, client=client)
    reports = await detector.run_test_suite("classify-intent", cases)
    summary = DriftDetector.summarize(reports)

    result = {
        "summary": {
            "total_tests": summary["total_tests"],
            "passed": summary["passed"],
            "drifted": summary["drifted"],
            "drift_rate": summary["drift_rate"],
            "avg_similarity": summary["avg_similarity"],
        },
        "reports": [r.model_dump() for r in reports],
    }

    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Drift report written to %s", output_path)

    if summary["drift_rate"] > drift_threshold:
        logger.error(
            "Drift rate %.1f%% exceeds threshold %.1f%%",
            summary["drift_rate"] * 100,
            drift_threshold * 100,
        )
        return False
    return True


def main() -> None:
    """CLI entry point for CI drift detection."""
    parser = argparse.ArgumentParser(description="CI drift detection runner")
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=DEFAULT_TEST_CASES,
        help="Path to test cases JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Maximum allowed drift rate (0.0-1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("drift-report.json"),
        help="Output file for the drift report",
    )
    args = parser.parse_args()

    ok = asyncio.run(run(args.test_cases, args.threshold, args.output))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
