"""Tests for prompt registry.

Phase 7 | Prompt Registry
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from prompt_versioning.examples.prompt_registry.registry import PromptRegistry


class TestPromptRegistry:
    """Unit tests for the PromptRegistry class."""

    def test_register_creates_version(self):
        registry = PromptRegistry()
        pv = registry.register("test-prompt", "Hello world")
        assert pv.version == 1
        assert pv.content == "Hello world"
        assert pv.content_hash == hashlib.sha256(b"Hello world").hexdigest()

    def test_register_skips_duplicate(self):
        registry = PromptRegistry()
        registry.register("test-prompt", "Hello world")
        registry.register("test-prompt", "Hello world")
        assert len(registry.get_history("test-prompt")) == 1

    def test_register_increments_version(self):
        registry = PromptRegistry()
        registry.register("test-prompt", "Version one")
        pv2 = registry.register("test-prompt", "Version two")
        assert pv2.version == 2

    def test_get_current_returns_latest(self):
        registry = PromptRegistry()
        registry.register("p", "v1")
        registry.register("p", "v2")
        registry.register("p", "v3")
        current = registry.get_current("p")
        assert current is not None
        assert current.version == 3

    def test_get_version_returns_specific(self):
        registry = PromptRegistry()
        registry.register("p", "v1")
        registry.register("p", "v2")
        registry.register("p", "v3")
        pv = registry.get_version("p", 2)
        assert pv is not None
        assert pv.content == "v2"

    def test_get_history_returns_all(self):
        registry = PromptRegistry()
        registry.register("p", "v1")
        registry.register("p", "v2")
        registry.register("p", "v3")
        history = registry.get_history("p")
        assert len(history) == 3
        assert [h.version for h in history] == [1, 2, 3]

    def test_rollback_creates_new_version(self):
        registry = PromptRegistry()
        registry.register("p", "content-v1", "first")
        registry.register("p", "content-v2", "second")
        rolled = registry.rollback("p", 1)
        assert rolled.version == 3
        assert rolled.content == "content-v1"
        assert rolled.description == "Rollback to v1"

    def test_rollback_invalid_version_raises(self):
        registry = PromptRegistry()
        registry.register("p", "content-v1")
        with pytest.raises(ValueError):
            registry.rollback("p", 99)

    def test_text_diff(self):
        registry = PromptRegistry()
        registry.register("p", "line one\nline two")
        registry.register("p", "line one\nline three")
        diff = registry.text_diff("p", 1, 2)
        assert "line three" in diff["added_lines"]
        assert "line two" in diff["removed_lines"]

    @pytest.mark.asyncio
    async def test_semantic_diff(self):
        client = AsyncMock()
        client.embed = AsyncMock(
            side_effect=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        registry = PromptRegistry(client=client)
        registry.register("p", "version one")
        registry.register("p", "version two")
        similarity = await registry.semantic_diff("p", 1, 2)
        assert similarity == pytest.approx(0.0, abs=0.01)

    # ------------------------------------------------------------------
    # mark_tested
    # ------------------------------------------------------------------

    def test_mark_tested_records_model_label(self):
        registry = PromptRegistry()
        registry.register("p", "content")
        registry.mark_tested("p", 1, "gpt-4o")
        pv = registry.get_version("p", 1)
        assert "gpt-4o" in pv.models_tested

    def test_mark_tested_no_duplicates(self):
        registry = PromptRegistry()
        registry.register("p", "content")
        registry.mark_tested("p", 1, "gpt-4o")
        registry.mark_tested("p", 1, "gpt-4o")
        pv = registry.get_version("p", 1)
        assert pv.models_tested.count("gpt-4o") == 1

    def test_mark_tested_multiple_models(self):
        registry = PromptRegistry()
        registry.register("p", "content")
        registry.mark_tested("p", 1, "gpt-4o")
        registry.mark_tested("p", 1, "claude-3.5-sonnet")
        pv = registry.get_version("p", 1)
        assert set(pv.models_tested) == {"gpt-4o", "claude-3.5-sonnet"}

    def test_mark_tested_invalid_version_raises(self):
        registry = PromptRegistry()
        registry.register("p", "content")
        with pytest.raises(ValueError):
            registry.mark_tested("p", 99, "gpt-4o")
