"""Drift detector — detect output changes after model updates.

Phase 7 | Drift Detection
# Phase 7 dependency: uses PromptRegistry from prompt_registry
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_versioning.utils.llm_client import LLMClient, MultiModelClient
from prompt_versioning.utils.logging_utils import get_logger
from prompt_versioning.utils.math_utils import cosine_similarity
from prompt_versioning.utils.types import DriftReport

from prompt_versioning.examples.prompt_registry.registry import PromptRegistry

logger = get_logger(__name__)


@dataclass
class DriftTestCase:
    """A single drift detection test case."""

    input_text: str
    expected_output: str


class DriftDetector:
    """Detect output drift by comparing LLM outputs against a baseline."""

    def __init__(
        self,
        registry: PromptRegistry,
        similarity_threshold: float = 0.85,
        client: LLMClient | None = None,
    ) -> None:
        self._registry = registry
        self._threshold = similarity_threshold
        self._client = client

    async def run_test_suite(
        self, prompt_name: str, test_cases: list[DriftTestCase]
    ) -> list[DriftReport]:
        """Run all test cases and return drift reports."""
        prompt = self._registry.get_current(prompt_name)
        if prompt is None:
            raise ValueError(f"Prompt '{prompt_name}' not found in registry")
        if self._client is None:
            raise ValueError("LLMClient required for drift detection")

        reports: list[DriftReport] = []
        for tc in test_cases:
            messages = [
                {"role": "system", "content": prompt.content},
                {"role": "user", "content": tc.input_text},
            ]
            # temperature=0.0 is mandatory for drift detection — non-zero
            # temperature causes the same prompt to produce different outputs
            # on consecutive runs, generating false positive drift signals.
            response = await self._client.chat(messages, temperature=0.0)
            current_output = response.choices[0].message.content or ""

            emb_expected = await self._client.embed(tc.expected_output)
            emb_actual = await self._client.embed(current_output)
            similarity = cosine_similarity(emb_expected, emb_actual)

            report = DriftReport(
                prompt_name=prompt_name,
                prompt_version=prompt.version,
                test_input=tc.input_text,
                baseline_output=tc.expected_output,
                current_output=current_output,
                similarity_score=similarity,
                drifted=similarity < self._threshold,
            )
            reports.append(report)
            logger.info(
                "Test '%s': similarity=%.3f drifted=%s",
                tc.input_text[:40],
                similarity,
                report.drifted,
            )
        return reports

    @staticmethod
    def summarize(reports: list[DriftReport]) -> dict:
        """Produce a summary of drift detection results."""
        if not reports:
            return {
                "total_tests": 0,
                "passed": 0,
                "drifted": 0,
                "drift_rate": 0.0,
                "avg_similarity": 0.0,
                "worst_case": None,
            }
        passed = sum(1 for r in reports if not r.drifted)
        drifted = sum(1 for r in reports if r.drifted)
        avg_sim = sum(r.similarity_score for r in reports) / len(reports)
        worst = min(reports, key=lambda r: r.similarity_score)
        return {
            "total_tests": len(reports),
            "passed": passed,
            "drifted": drifted,
            "drift_rate": drifted / len(reports),
            "avg_similarity": avg_sim,
            "worst_case": worst,
        }


class CrossModelDriftDetector:
    """Run drift detection across multiple models for the same prompt."""

    def __init__(
        self,
        registry: PromptRegistry,
        multi_client: MultiModelClient,
        threshold: float = 0.85,
    ) -> None:
        self._registry = registry
        self._multi_client = multi_client
        self._threshold = threshold

    async def run_cross_model_suite(
        self, prompt_name: str, test_cases: list[DriftTestCase]
    ) -> dict[str, list[DriftReport]]:
        """Run the same test suite against each model. Returns {model_label: [DriftReport]}."""
        results: dict[str, list[DriftReport]] = {}
        for label in self._multi_client.labels():
            client = self._multi_client.get_client(label)
            detector = DriftDetector(
                self._registry, similarity_threshold=self._threshold, client=client
            )
            reports = await detector.run_test_suite(prompt_name, test_cases)
            results[label] = reports
            logger.info(
                "Completed drift suite for model '%s': %d reports", label, len(reports)
            )
        return results

    def cross_model_summary(self, results: dict[str, list[DriftReport]]) -> dict:
        """Compare drift rates across models.

        Returns:
            {
                "per_model": {label: {drift_rate, avg_similarity, ...}},
                "most_stable_model": str,
                "most_drifted_model": str,
                "cross_model_agreement": float  # % of test cases where all models agree
            }
        """
        per_model: dict[str, dict] = {}
        for label, reports in results.items():
            per_model[label] = DriftDetector.summarize(reports)

        if not per_model:
            return {
                "per_model": {},
                "most_stable_model": "",
                "most_drifted_model": "",
                "cross_model_agreement": 0.0,
            }

        most_stable = min(per_model, key=lambda k: per_model[k]["drift_rate"])
        most_drifted = max(per_model, key=lambda k: per_model[k]["drift_rate"])

        # Cross-model agreement: % of test cases where all models agree on drift/no-drift
        labels = list(results.keys())
        if labels and results[labels[0]]:
            num_cases = len(results[labels[0]])
            agreements = 0
            for i in range(num_cases):
                verdicts = [
                    results[lbl][i].drifted for lbl in labels if i < len(results[lbl])
                ]
                if len(set(verdicts)) == 1:
                    agreements += 1
            agreement_rate = agreements / num_cases if num_cases else 0.0
        else:
            agreement_rate = 0.0

        return {
            "per_model": per_model,
            "most_stable_model": most_stable,
            "most_drifted_model": most_drifted,
            "cross_model_agreement": agreement_rate,
        }
