"""Tests for cross-model A/B testing.

Phase 7 | Cross-Model A/B Testing
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_versioning.examples.ab_tester.tester import CrossModelABTester
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry
from prompt_versioning.utils.llm_client import LLMClient, MultiModelClient


def _make_registry() -> PromptRegistry:
    """Create a registry with a single prompt version."""
    registry = PromptRegistry()
    registry.register(
        "summarize",
        "Summarize the following text in 2-3 sentences.",
        "v1",
    )
    return registry


def _make_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_judge(score: str) -> AsyncMock:
    """Build a mock judge LLMClient that always returns the given score."""
    judge = AsyncMock(spec=LLMClient)
    judge.chat = AsyncMock(return_value=_make_response(score))
    return judge


def _mock_multi_client(
    model_labels: list[str],
) -> MultiModelClient:
    """Build a mock MultiModelClient where each model returns a fixed output."""
    multi = MagicMock(spec=MultiModelClient)
    multi.labels.return_value = model_labels

    clients: dict[str, AsyncMock] = {}
    for label in model_labels:
        client = AsyncMock()
        client.chat = AsyncMock(return_value=_make_response(f"Summary by {label}"))
        clients[label] = client

    multi.get_client.side_effect = lambda lbl: clients[lbl]
    return multi


class TestCrossModelABTester:
    """Unit tests for the CrossModelABTester class."""

    @pytest.mark.asyncio
    async def test_identifies_winner_by_quality(self):
        """Model with highest avg quality score wins when using a separate judge."""
        multi = _mock_multi_client(["claude", "gpt-4o", "gemini"])

        # Judge returns different scores based on output content
        call_count = 0

        async def judge_side_effect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            content = str(messages)
            if "Summary by claude" in content:
                return _make_response("4.5")
            elif "Summary by gpt-4o" in content:
                return _make_response("3.0")
            else:
                return _make_response("3.5")

        judge = AsyncMock(spec=LLMClient)
        judge.chat = AsyncMock(side_effect=judge_side_effect)

        tester = CrossModelABTester(
            _make_registry(), "summarize", 1, multi, judge_client=judge
        )
        comparison = await tester.run_model_comparison(["test query 1", "test query 2"])

        assert comparison["winner"] == "claude"
        assert set(comparison["per_model"].keys()) == {"claude", "gpt-4o", "gemini"}
        assert comparison["per_model"]["claude"]["avg_quality"] == pytest.approx(4.5)
        assert comparison["per_model"]["gpt-4o"]["avg_quality"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_runs_all_queries_against_all_models(self):
        """Each model receives all queries."""
        multi = _mock_multi_client(["claude", "gpt-4o"])
        judge = _mock_judge("4.0")

        tester = CrossModelABTester(
            _make_registry(), "summarize", 1, multi, judge_client=judge
        )
        comparison = await tester.run_model_comparison(["q1", "q2", "q3"])

        for label in ("claude", "gpt-4o"):
            assert comparison["per_model"][label]["count"] == 3

    @pytest.mark.asyncio
    async def test_handles_tie(self):
        """When all models score equally, one is still selected as winner."""
        multi = _mock_multi_client(["claude", "gpt-4o", "gemini"])
        judge = _mock_judge("4.0")

        tester = CrossModelABTester(
            _make_registry(), "summarize", 1, multi, judge_client=judge
        )
        comparison = await tester.run_model_comparison(["q1"])

        assert comparison["winner"] in ("claude", "gpt-4o", "gemini")

    def test_print_comparison_format(self):
        """print_comparison produces formatted output."""
        multi = MagicMock(spec=MultiModelClient)
        multi.labels.return_value = ["claude", "gpt-4o"]
        multi.get_client.return_value = AsyncMock(spec=LLMClient)
        tester = CrossModelABTester(_make_registry(), "summarize", 1, multi)
        comparison = {
            "per_model": {
                "claude": {
                    "model": "claude",
                    "count": 3,
                    "avg_quality": 4.5,
                    "avg_latency_ms": 800.0,
                    "avg_tokens": 35.0,
                },
                "gpt-4o": {
                    "model": "gpt-4o",
                    "count": 3,
                    "avg_quality": 4.2,
                    "avg_latency_ms": 600.0,
                    "avg_tokens": 30.0,
                },
            },
            "winner": "claude",
            "results": {},
        }
        report = tester.print_comparison(comparison)
        assert "claude" in report
        assert "gpt-4o" in report
        assert "Winner: claude" in report
