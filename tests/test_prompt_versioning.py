"""
Test suite — prompt_versioning

Run: pytest prompt_versioning/ -v

TODO — implement tests for all must-have features:
  Each test file maps to one example:
    test_prompt_versioning.py → integration + unit tests

Coverage targets:
  - Unit tests: each class/function in isolation
  - Integration tests: end-to-end pipeline with real LLM calls (marked @pytest.mark.integration)
  - Smoke tests: basic import + instantiation (always run, no API key needed)
"""

import pytest

# ── Smoke tests (no API key needed) ────────────────────────────


def test_imports():
    """Package imports without error."""
    import prompt_versioning  # noqa: F401


# ── Unit tests ─────────────────────────────────────────────────

# TODO: add unit tests for each component
# Example structure:
#
# class TestExampleComponent:
#     def test_basic_case(self):
#         ...
#     def test_edge_case(self):
#         ...


# ── Integration tests (require API keys) ───────────────────────


@pytest.mark.integration
def test_end_to_end():
    """Full pipeline smoke test — requires OPENAI_API_KEY or ANTHROPIC_API_KEY."""
    pytest.skip("TODO: implement end-to-end integration test")
