"""Tests for ci_runner — drift detection CI entrypoint.

Phase 7 | Drift Detection — CI/CD Integration
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_versioning.examples.drift_detector.ci_runner import run
from prompt_versioning.examples.drift_detector.detector import (
    DriftDetector,
    DriftReport,
)


def _make_report(passed: bool, similarity: float) -> DriftReport:
    return DriftReport(
        prompt_name="classify-intent",
        prompt_version=1,
        test_input="test input",
        baseline_output="expected",
        current_output="actual",
        similarity_score=similarity,
        drifted=not passed,
    )


def _patch_runner(reports: list[DriftReport]):
    """Patch LLMClient and DriftDetector for ci_runner tests.

    Only mock the DriftDetector *constructor* and `run_test_suite`.
    Let `DriftDetector.summarize` (a static method) use the real
    implementation so its output is JSON-serializable.
    """
    mock_detector_instance = MagicMock()
    mock_detector_instance.run_test_suite = AsyncMock(return_value=reports)

    mock_cls = MagicMock()
    mock_cls.return_value = mock_detector_instance
    # Preserve the real static method so json.dumps works
    mock_cls.summarize = DriftDetector.summarize

    return (
        patch("prompt_versioning.examples.drift_detector.ci_runner.LLMClient"),
        patch(
            "prompt_versioning.examples.drift_detector.ci_runner.DriftDetector",
            mock_cls,
        ),
    )


@pytest.mark.asyncio
async def test_run_returns_true_when_drift_within_threshold(tmp_path: Path):
    test_cases_file = tmp_path / "test_cases.json"
    test_cases_file.write_text(json.dumps([{"input": "hello", "expected": "greeting"}]))
    output_file = tmp_path / "report.json"

    passing_report = _make_report(passed=True, similarity=0.95)
    patch_client, patch_detector = _patch_runner([passing_report])

    with patch_client, patch_detector:
        ok = await run(test_cases_file, drift_threshold=0.10, output_path=output_file)

    assert ok is True
    assert output_file.exists()
    report = json.loads(output_file.read_text())
    assert report["summary"]["drift_rate"] == 0.0


@pytest.mark.asyncio
async def test_run_returns_false_when_drift_exceeds_threshold(tmp_path: Path):
    test_cases_file = tmp_path / "test_cases.json"
    test_cases_file.write_text(
        json.dumps(
            [
                {"input": "hello", "expected": "greeting"},
                {"input": "help", "expected": "question"},
            ]
        )
    )
    output_file = tmp_path / "report.json"

    drifted_reports = [
        _make_report(passed=False, similarity=0.60),
        _make_report(passed=False, similarity=0.55),
    ]
    patch_client, patch_detector = _patch_runner(drifted_reports)

    with patch_client, patch_detector:
        ok = await run(test_cases_file, drift_threshold=0.10, output_path=output_file)

    assert ok is False
    report = json.loads(output_file.read_text())
    assert report["summary"]["drift_rate"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_writes_full_report_structure(tmp_path: Path):
    test_cases_file = tmp_path / "test_cases.json"
    test_cases_file.write_text(json.dumps([{"input": "ping", "expected": "pong"}]))
    output_file = tmp_path / "report.json"

    report = _make_report(passed=True, similarity=0.92)
    patch_client, patch_detector = _patch_runner([report])

    with patch_client, patch_detector:
        await run(test_cases_file, drift_threshold=0.10, output_path=output_file)

    data = json.loads(output_file.read_text())
    assert "summary" in data
    assert "reports" in data
    summary = data["summary"]
    for key in ("total_tests", "passed", "drifted", "drift_rate", "avg_similarity"):
        assert key in summary, f"Missing key in summary: {key}"
