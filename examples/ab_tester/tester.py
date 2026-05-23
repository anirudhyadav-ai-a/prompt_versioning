"""A/B tester — compare prompt versions by routing traffic and scoring.

Phase 7 | A/B Testing
# Phase 7 dependency: uses PromptRegistry and DriftDetector patterns
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from prompt_versioning.utils.llm_client import LLMClient, MultiModelClient
from prompt_versioning.utils.logging_utils import get_logger
from prompt_versioning.utils.token_utils import count_tokens

from prompt_versioning.examples.ab_tester.stats import welch_t_test
from prompt_versioning.examples.prompt_registry.registry import PromptRegistry

logger = get_logger(__name__)


@dataclass
class ABResult:
    """Result of a single A/B test query."""

    query: str
    variant: str
    prompt_version: int
    output: str
    quality_score: float
    latency_ms: float
    token_count: int


class ABTester:
    """Route queries between two prompt versions and score results."""

    def __init__(
        self,
        registry: PromptRegistry,
        prompt_name: str,
        version_a: int,
        version_b: int,
        split_ratio: float = 0.5,
        client: LLMClient | None = None,
    ) -> None:
        self._registry = registry
        self._prompt_name = prompt_name
        self._version_a = version_a
        self._version_b = version_b
        self._split_ratio = split_ratio
        self._client = client

    async def run_query(self, query: str) -> ABResult:
        """Route a single query to variant A or B, score the output."""
        if self._client is None:
            raise ValueError("LLMClient required for A/B testing")

        is_a = random.random() < self._split_ratio
        variant = "A" if is_a else "B"
        version_num = self._version_a if is_a else self._version_b

        prompt = self._registry.get_version(self._prompt_name, version_num)
        if prompt is None:
            raise ValueError(f"Version {version_num} not found")

        start = time.perf_counter()
        response = await self._client.chat(
            [
                {"role": "system", "content": prompt.content},
                {"role": "user", "content": query},
            ]
        )
        latency_ms = (time.perf_counter() - start) * 1000
        output = response.choices[0].message.content or ""

        score_response = await self._client.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Rate the following response on a scale of 1.0 to 5.0 for "
                        "quality, relevance, and completeness. Respond with only a "
                        f"number.\n\nQuery: {query}\nResponse: {output}"
                    ),
                },
            ]
        )
        score_text = score_response.choices[0].message.content or "3.0"
        try:
            quality_score = float(score_text.strip())
        except ValueError:
            quality_score = 3.0

        token_count = count_tokens(output)

        return ABResult(
            query=query,
            variant=variant,
            prompt_version=version_num,
            output=output,
            quality_score=quality_score,
            latency_ms=latency_ms,
            token_count=token_count,
        )

    async def run_experiment(self, queries: list[str]) -> dict:
        """Run all queries and produce experiment results."""
        results: list[ABResult] = []
        for q in queries:
            results.append(await self.run_query(q))

        group_a = [r for r in results if r.variant == "A"]
        group_b = [r for r in results if r.variant == "B"]

        def _stats(group: list[ABResult], version: int) -> dict:
            if not group:
                return {
                    "version": version,
                    "count": 0,
                    "avg_quality": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_tokens": 0.0,
                }
            return {
                "version": version,
                "count": len(group),
                "avg_quality": sum(r.quality_score for r in group) / len(group),
                "avg_latency_ms": sum(r.latency_ms for r in group) / len(group),
                "avg_tokens": sum(r.token_count for r in group) / len(group),
            }

        stats_a = _stats(group_a, self._version_a)
        stats_b = _stats(group_b, self._version_b)

        if not group_a or not group_b:
            winner = "insufficient_data"
            stat_result = None
        else:
            scores_a = [r.quality_score for r in group_a]
            scores_b = [r.quality_score for r in group_b]
            stat_result = welch_t_test(scores_a, scores_b)
            if stat_result.significant:
                winner = "A" if stat_result.mean_a > stat_result.mean_b else "B"
            else:
                winner = "tie"

        return {
            "variant_a": stats_a,
            "variant_b": stats_b,
            "winner": winner,
            "statistics": stat_result,
            "results": results,
        }

    def print_report(self, experiment: dict) -> str:
        """Format a human-readable A/B test report."""
        a = experiment["variant_a"]
        b = experiment["variant_b"]
        lines = [
            "=" * 60,
            "A/B Test Report",
            "=" * 60,
            f"{'Metric':<20} {'Variant A':>15} {'Variant B':>15}",
            "-" * 60,
            f"{'Version':<20} {'v' + str(a['version']):>15} {'v' + str(b['version']):>15}",
            f"{'Queries':<20} {a['count']:>15} {b['count']:>15}",
            f"{'Avg Quality':<20} {a['avg_quality']:>15.2f} {b['avg_quality']:>15.2f}",
            f"{'Avg Latency (ms)':<20} {a['avg_latency_ms']:>15.1f} {b['avg_latency_ms']:>15.1f}",
            f"{'Avg Tokens':<20} {a['avg_tokens']:>15.1f} {b['avg_tokens']:>15.1f}",
            "-" * 60,
            f"Winner: {experiment['winner']}",
            "=" * 60,
        ]
        stat = experiment.get("statistics")
        if stat is not None:
            lines.extend(
                [
                    "",
                    "Statistical Significance",
                    "-" * 60,
                    f"  p-value:        {stat.p_value:.4f}",
                    f"  Significant:    {'Yes' if stat.significant else 'No'} (alpha=0.05)",
                    f"  95% CI (A-B):   ({stat.confidence_interval[0]:.3f}, {stat.confidence_interval[1]:.3f})",
                    f"  Effect size:    {stat.effect_size:.3f} (Cohen's d)",
                    f"  Recommended N:  {stat.recommended_sample_size} per group",
                ]
            )
        report = "\n".join(lines)
        print(report)
        return report


@dataclass
class ModelComparisonResult:
    """Result of a single model comparison query."""

    query: str
    model_label: str
    output: str
    quality_score: float
    latency_ms: float
    token_count: int


class CrossModelABTester:
    """A/B test the same prompt across different models."""

    def __init__(
        self,
        registry: PromptRegistry,
        prompt_name: str,
        version: int,
        models: MultiModelClient,
        judge_client: LLMClient | None = None,
    ) -> None:
        self._registry = registry
        self._prompt_name = prompt_name
        self._version = version
        self._models = models
        labels = models.labels()
        self._judge = judge_client or (models.get_client(labels[0]) if labels else None)
        if self._judge is None:
            raise ValueError(
                "CrossModelABTester requires a judge client. Either provide "
                "judge_client= explicitly, or ensure MultiModelClient has at "
                "least one model configured."
            )

    async def run_model_comparison(self, queries: list[str]) -> dict:
        """Run each query against each model, score with LLM-as-judge.

        Returns per-model stats (avg quality, avg latency, avg tokens, cost
        estimate) and a winner.
        """
        prompt = self._registry.get_version(self._prompt_name, self._version)
        if prompt is None:
            raise ValueError(
                f"Version {self._version} not found for '{self._prompt_name}'"
            )

        all_results: dict[str, list[ModelComparisonResult]] = {
            label: [] for label in self._models.labels()
        }

        for label in self._models.labels():
            client = self._models.get_client(label)
            for query in queries:
                start = time.perf_counter()
                response = await client.chat(
                    [
                        {"role": "system", "content": prompt.content},
                        {"role": "user", "content": query},
                    ]
                )
                latency_ms = (time.perf_counter() - start) * 1000
                output = response.choices[0].message.content or ""

                score_response = await self._judge.chat(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Rate the following response on a scale of 1.0 to 5.0 for "
                                "quality, relevance, and completeness. Respond with only a "
                                f"number.\n\nQuery: {query}\nResponse: {output}"
                            ),
                        },
                    ]
                )
                score_text = score_response.choices[0].message.content or "3.0"
                try:
                    quality_score = float(score_text.strip())
                except ValueError:
                    quality_score = 3.0

                token_count = count_tokens(output)
                all_results[label].append(
                    ModelComparisonResult(
                        query=query,
                        model_label=label,
                        output=output,
                        quality_score=quality_score,
                        latency_ms=latency_ms,
                        token_count=token_count,
                    )
                )

        per_model: dict[str, dict] = {}
        for label, results in all_results.items():
            if not results:
                per_model[label] = {
                    "model": label,
                    "count": 0,
                    "avg_quality": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_tokens": 0.0,
                }
                continue
            per_model[label] = {
                "model": label,
                "count": len(results),
                "avg_quality": sum(r.quality_score for r in results) / len(results),
                "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
                "avg_tokens": sum(r.token_count for r in results) / len(results),
            }

        has_data = any(stats["count"] > 0 for stats in per_model.values())
        if per_model and has_data:
            winner_label = max(per_model, key=lambda k: per_model[k]["avg_quality"])
        else:
            winner_label = "insufficient_data"

        return {
            "per_model": per_model,
            "winner": winner_label,
            "results": all_results,
        }

    def print_comparison(self, comparison: dict) -> str:
        """Format a human-readable cross-model comparison report."""
        lines = [
            "=" * 70,
            "Cross-Model Comparison Report",
            "=" * 70,
            f"{'Model':<15} {'Queries':>10} {'Avg Quality':>13} {'Avg Latency':>13} {'Avg Tokens':>12}",
            "-" * 70,
        ]
        for label, stats in comparison["per_model"].items():
            lines.append(
                f"{label:<15} {stats['count']:>10} {stats['avg_quality']:>13.2f} "
                f"{stats['avg_latency_ms']:>12.1f}ms {stats['avg_tokens']:>11.1f}"
            )
        lines.extend(
            [
                "-" * 70,
                f"Winner: {comparison['winner']}",
                "=" * 70,
            ]
        )
        report = "\n".join(lines)
        print(report)
        return report
