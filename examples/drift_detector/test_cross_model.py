"""Tests for cross-model drift detection.

Phase 7 | Cross-Model Drift Detection
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_versioning.examples.drift_detector.detector import (
    CrossModelDriftDetector,
    DriftTestCase,
)
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry
from prompt_versioning.utils.llm_client import MultiModelClient


def _make_registry(prompt_content: str = "Classify intent") -> PromptRegistry:
    """Create a registry with a single registered prompt."""
    registry = PromptRegistry()
    registry.register("test-prompt", prompt_content)
    return registry


def _mock_multi_client(
    model_outputs: dict[str, str],
    embed_vectors: dict[str, list[list[float]]],
) -> MultiModelClient:
    """Build a mock MultiModelClient with per-model chat and embed outputs."""
    multi = MagicMock(spec=MultiModelClient)
    multi.labels.return_value = list(model_outputs.keys())

    clients: dict[str, AsyncMock] = {}
    for label in model_outputs:
        client = AsyncMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = model_outputs[label]
        client.chat = AsyncMock(return_value=response)
        client.embed = AsyncMock(side_effect=embed_vectors[label])
        clients[label] = client

    multi.get_client.side_effect = lambda lbl: clients[lbl]
    return multi


class TestCrossModelDriftDetector:
    """Unit tests for the CrossModelDriftDetector class."""

    @pytest.mark.asyncio
    async def test_runs_against_all_models(self):
        """Suite runs against each model and returns results keyed by label."""
        vec_match = [1.0, 0.0, 0.0]
        multi = _mock_multi_client(
            model_outputs={
                "claude": "greeting",
                "gpt-4o": "greeting",
                "gemini": "greeting",
            },
            embed_vectors={
                "claude": [vec_match, vec_match],
                "gpt-4o": [vec_match, vec_match],
                "gemini": [vec_match, vec_match],
            },
        )

        detector = CrossModelDriftDetector(_make_registry(), multi, threshold=0.85)
        results = await detector.run_cross_model_suite(
            "test-prompt", [DriftTestCase("Hi", "greeting")]
        )

        assert set(results.keys()) == {"claude", "gpt-4o", "gemini"}
        for label, reports in results.items():
            assert len(reports) == 1
            assert reports[0].drifted is False

    @pytest.mark.asyncio
    async def test_detects_per_model_drift(self):
        """Detects drift on one model while others pass."""
        vec_match = [1.0, 0.0, 0.0]
        vec_drift = [0.0, 1.0, 0.0]
        multi = _mock_multi_client(
            model_outputs={
                "claude": "greeting",
                "gpt-4o": "greeting",
                "gemini": "unknown",
            },
            embed_vectors={
                "claude": [vec_match, vec_match],
                "gpt-4o": [vec_match, vec_match],
                "gemini": [vec_match, vec_drift],
            },
        )

        detector = CrossModelDriftDetector(_make_registry(), multi, threshold=0.85)
        results = await detector.run_cross_model_suite(
            "test-prompt", [DriftTestCase("Hi", "greeting")]
        )

        assert results["claude"][0].drifted is False
        assert results["gpt-4o"][0].drifted is False
        assert results["gemini"][0].drifted is True

    @pytest.mark.asyncio
    async def test_cross_model_summary_identifies_stable_and_drifted(self):
        """Summary correctly identifies most stable and most drifted model."""
        vec_match = [1.0, 0.0, 0.0]
        vec_drift = [0.0, 1.0, 0.0]
        multi = _mock_multi_client(
            model_outputs={"claude": "ok", "gpt-4o": "ok", "gemini": "bad"},
            embed_vectors={
                "claude": [vec_match, vec_match, vec_match, vec_match],
                "gpt-4o": [vec_match, vec_match, vec_match, vec_match],
                "gemini": [vec_match, vec_drift, vec_match, vec_drift],
            },
        )

        detector = CrossModelDriftDetector(_make_registry(), multi, threshold=0.85)
        results = await detector.run_cross_model_suite(
            "test-prompt",
            [DriftTestCase("a", "ok"), DriftTestCase("b", "ok")],
        )

        summary = detector.cross_model_summary(results)
        assert summary["most_stable_model"] in ("claude", "gpt-4o")
        assert summary["most_drifted_model"] == "gemini"
        assert summary["per_model"]["gemini"]["drift_rate"] > 0
        assert summary["per_model"]["claude"]["drift_rate"] == 0.0
        assert 0.0 <= summary["cross_model_agreement"] <= 1.0

    def test_cross_model_summary_empty(self):
        """Summary handles empty results gracefully."""
        multi = MagicMock(spec=MultiModelClient)
        multi.labels.return_value = []
        detector = CrossModelDriftDetector(PromptRegistry(), multi)
        summary = detector.cross_model_summary({})
        assert summary["most_stable_model"] == ""
        assert summary["most_drifted_model"] == ""
        assert summary["cross_model_agreement"] == 0.0
