"""Tests for A/B tester.

Phase 7 | A/B Testing
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_versioning.examples.ab_tester.stats import (
    minimum_sample_size,
    welch_t_test,
)
from prompt_versioning.examples.ab_tester.tester import ABTester
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry


def _make_registry() -> PromptRegistry:
    """Create a registry with two prompt versions."""
    registry = PromptRegistry()
    registry.register("summarize", "Summarize in 2-3 sentences.", "v1")
    registry.register("summarize", "Professional summary in 2 sentences.", "v2")
    return registry


def _mock_client(chat_output: str, score: str = "4.0") -> AsyncMock:
    """Build a mock LLMClient for A/B testing."""
    client = AsyncMock()

    def make_response(content: str) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        return resp

    client.chat = AsyncMock(
        side_effect=[
            make_response(chat_output),
            make_response(score),
        ]
        * 20
    )  # enough for multiple calls
    return client


# --- ABTester tests ---


class TestABTester:
    """Unit tests for the ABTester class."""

    @pytest.mark.asyncio
    async def test_routing_respects_split_ratio(self):
        client = _mock_client("output", "4.0")
        tester = ABTester(
            _make_registry(),
            "summarize",
            1,
            2,
            split_ratio=0.5,
            client=client,
        )
        with patch("prompt_versioning.examples.ab_tester.tester.random") as mock_random:
            mock_random.random.return_value = 0.3  # < 0.5 -> variant A
            result = await tester.run_query("test query")
            assert result.variant == "A"

        client2 = _mock_client("output", "4.0")
        tester2 = ABTester(
            _make_registry(),
            "summarize",
            1,
            2,
            split_ratio=0.5,
            client=client2,
        )
        with patch("prompt_versioning.examples.ab_tester.tester.random") as mock_random:
            mock_random.random.return_value = 0.7  # > 0.5 -> variant B
            result = await tester2.run_query("test query")
            assert result.variant == "B"

    @pytest.mark.asyncio
    async def test_run_query_returns_result(self):
        client = _mock_client("summary output", "4.5")
        tester = ABTester(
            _make_registry(),
            "summarize",
            1,
            2,
            split_ratio=0.5,
            client=client,
        )
        with patch("prompt_versioning.examples.ab_tester.tester.random") as mock_random:
            mock_random.random.return_value = 0.3
            result = await tester.run_query("Summarize this text")
        assert result.query == "Summarize this text"
        assert result.variant == "A"
        assert result.output == "summary output"
        assert result.quality_score == 4.5
        assert result.latency_ms >= 0
        assert result.token_count >= 0

    @pytest.mark.asyncio
    async def test_experiment_determines_winner(self):
        """A always scores 4.0 and B always scores 2.0.

        We use a large sample size (10 per group) so the mean difference
        is statistically significant and the winner is deterministically "A".
        """
        registry = _make_registry()
        client = AsyncMock()

        # Alternate: odd calls = LLM output, even calls = judge score.
        # A group scores cluster around 4.0, B group around 2.0.
        # Small variance is needed so Welch's t-test can compute a
        # meaningful t-statistic (zero variance → not significant).
        call_count = 0
        a_scores = [3.9, 4.0, 4.1, 3.8, 4.2, 4.0, 3.9, 4.1, 4.0, 4.0]
        b_scores = [2.1, 2.0, 1.9, 2.2, 1.8, 2.0, 2.1, 1.9, 2.0, 2.0]

        def make_response(content: str) -> MagicMock:
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = content
            return resp

        async def mock_chat(messages: list, **kwargs) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return make_response("output")
            pair_idx = (call_count // 2) - 1
            if pair_idx < 10:
                return make_response(str(a_scores[pair_idx]))
            return make_response(str(b_scores[pair_idx - 10]))

        client.chat = mock_chat
        tester = ABTester(registry, "summarize", 1, 2, client=client)

        queries = [f"q{i}" for i in range(20)]
        with patch("prompt_versioning.examples.ab_tester.tester.random") as mock_random:
            # First 10 → A (< 0.5), last 10 → B (>= 0.5)
            mock_random.random.side_effect = [0.1] * 10 + [0.9] * 10
            experiment = await tester.run_experiment(queries)

        assert "statistics" in experiment
        assert experiment["statistics"].mean_a == pytest.approx(4.0, abs=0.1)
        assert experiment["statistics"].mean_b == pytest.approx(2.0, abs=0.1)
        assert experiment["winner"] == "A"

    @pytest.mark.asyncio
    async def test_experiment_tie(self):
        registry = _make_registry()
        client = AsyncMock()

        def make_response(content: str) -> MagicMock:
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = content
            return resp

        client.chat = AsyncMock(
            side_effect=[
                make_response("out"),
                make_response("3.5"),
                make_response("out"),
                make_response("3.5"),
            ]
        )
        tester = ABTester(registry, "summarize", 1, 2, client=client)

        with patch("prompt_versioning.examples.ab_tester.tester.random") as mock_random:
            mock_random.random.side_effect = [0.1, 0.9]
            experiment = await tester.run_experiment(["q1", "q2"])

        assert experiment["winner"] == "tie"

    def test_print_report_format(self):
        registry = _make_registry()
        tester = ABTester(registry, "summarize", 1, 2)
        experiment = {
            "variant_a": {
                "version": 1,
                "count": 3,
                "avg_quality": 4.0,
                "avg_latency_ms": 150.0,
                "avg_tokens": 20.0,
            },
            "variant_b": {
                "version": 2,
                "count": 3,
                "avg_quality": 3.5,
                "avg_latency_ms": 160.0,
                "avg_tokens": 18.0,
            },
            "winner": "A",
            "statistics": None,
            "results": [],
        }
        report = tester.print_report(experiment)
        assert "Variant A" in report
        assert "Variant B" in report
        assert "Winner: A" in report


# --- Statistical significance tests ---


class TestWelchTTest:
    """Unit tests for Welch's t-test."""

    def test_significant_difference(self):
        scores_a = [4.0, 4.5, 4.2, 4.8, 4.1, 4.6, 4.3, 4.7, 4.4, 4.9]
        scores_b = [2.0, 2.5, 2.2, 2.8, 2.1, 2.6, 2.3, 2.7, 2.4, 2.9]
        result = welch_t_test(scores_a, scores_b)
        assert result.significant is True
        assert result.mean_a > result.mean_b
        assert result.p_value < 0.05
        assert result.effect_size > 0

    def test_not_significant_difference(self):
        scores_a = [3.0, 3.1, 2.9, 3.0, 3.05]
        scores_b = [3.0, 2.9, 3.1, 3.0, 2.95]
        result = welch_t_test(scores_a, scores_b)
        assert result.significant is False

    def test_insufficient_data(self):
        result = welch_t_test([4.0], [2.0])
        assert result.significant is False
        assert result.p_value == 1.0

    def test_confidence_interval(self):
        scores_a = [4.0, 4.5, 4.2, 4.8, 4.1, 4.6, 4.3, 4.7, 4.4, 4.9]
        scores_b = [2.0, 2.5, 2.2, 2.8, 2.1, 2.6, 2.3, 2.7, 2.4, 2.9]
        result = welch_t_test(scores_a, scores_b)
        assert result.confidence_interval[0] > 0  # lower bound positive
        assert result.confidence_interval[1] > result.confidence_interval[0]


class TestMinimumSampleSize:
    """Unit tests for minimum_sample_size."""

    def test_medium_effect(self):
        n = minimum_sample_size(0.5)
        assert 30 < n < 200  # reasonable range for medium effect

    def test_large_effect(self):
        n = minimum_sample_size(0.8)
        assert n < 100

    def test_zero_effect_returns_zero(self):
        assert minimum_sample_size(0.0) == 0

    def test_small_effect_needs_large_n(self):
        n = minimum_sample_size(0.2)
        assert n > 100
