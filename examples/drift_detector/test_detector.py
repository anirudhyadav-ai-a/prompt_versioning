"""Tests for drift detector.

Phase 7 | Drift Detection
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_versioning.examples.drift_detector.detector import (
    DriftDetector,
    DriftTestCase,
)
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry


def _make_registry(prompt_content: str = "Classify intent") -> PromptRegistry:
    """Create a registry with a single registered prompt."""
    registry = PromptRegistry()
    registry.register("test-prompt", prompt_content)
    return registry


def _mock_client(chat_output: str, embed_pairs: list[list[float]]) -> AsyncMock:
    """Build a mock LLMClient with controlled chat and embed outputs."""
    client = AsyncMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = chat_output
    client.chat = AsyncMock(return_value=response)
    client.embed = AsyncMock(side_effect=embed_pairs)
    return client


class TestDriftDetector:
    """Unit tests for the DriftDetector class."""

    @pytest.mark.asyncio
    async def test_no_drift_when_outputs_match(self):
        vec = [1.0, 0.0, 0.0]
        client = _mock_client("greeting", [vec, vec])
        detector = DriftDetector(
            _make_registry(), similarity_threshold=0.85, client=client
        )
        reports = await detector.run_test_suite(
            "test-prompt", [DriftTestCase("Hi", "greeting")]
        )
        assert len(reports) == 1
        assert reports[0].similarity_score == pytest.approx(1.0)
        assert reports[0].drifted is False

    @pytest.mark.asyncio
    async def test_drift_detected_when_outputs_differ(self):
        client = _mock_client("unknown", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        detector = DriftDetector(
            _make_registry(), similarity_threshold=0.85, client=client
        )
        reports = await detector.run_test_suite(
            "test-prompt", [DriftTestCase("Hi", "greeting")]
        )
        assert len(reports) == 1
        assert reports[0].similarity_score == pytest.approx(0.0, abs=0.01)
        assert reports[0].drifted is True

    @pytest.mark.asyncio
    async def test_partial_drift(self):
        same_vec = [1.0, 0.0, 0.0]
        diff_vec = [0.0, 1.0, 0.0]
        client = _mock_client(
            "ok",
            [
                same_vec,
                same_vec,  # test 1: match
                same_vec,
                same_vec,  # test 2: match
                same_vec,
                diff_vec,  # test 3: drift
            ],
        )
        detector = DriftDetector(
            _make_registry(), similarity_threshold=0.85, client=client
        )
        reports = await detector.run_test_suite(
            "test-prompt",
            [
                DriftTestCase("a", "ok"),
                DriftTestCase("b", "ok"),
                DriftTestCase("c", "ok"),
            ],
        )
        summary = detector.summarize(reports)
        assert summary["total_tests"] == 3
        assert summary["passed"] == 2
        assert summary["drifted"] == 1
        assert summary["drift_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_summarize_empty_reports(self):
        detector = DriftDetector(PromptRegistry(), similarity_threshold=0.85)
        summary = detector.summarize([])
        assert summary["total_tests"] == 0
        assert summary["passed"] == 0
        assert summary["drifted"] == 0
        assert summary["avg_similarity"] == 0.0
        assert summary["worst_case"] is None

    def test_summarize_worst_case(self):
        from prompt_versioning.utils.types import DriftReport

        reports = [
            DriftReport(
                prompt_name="p",
                prompt_version=1,
                test_input="a",
                baseline_output="x",
                current_output="x",
                similarity_score=0.95,
                drifted=False,
            ),
            DriftReport(
                prompt_name="p",
                prompt_version=1,
                test_input="b",
                baseline_output="y",
                current_output="z",
                similarity_score=0.40,
                drifted=True,
            ),
        ]
        detector = DriftDetector(PromptRegistry(), similarity_threshold=0.85)
        summary = detector.summarize(reports)
        assert summary["worst_case"] is not None
        assert summary["worst_case"].similarity_score == 0.40
