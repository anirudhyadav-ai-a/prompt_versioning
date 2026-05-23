"""Tests for prompt templating.

Phase 7 | Prompt Registry -- Templating
"""

from __future__ import annotations

import pytest

from prompt_versioning.examples.prompt_registry.templating import TemplatedPrompt


class TestTemplatedPrompt:
    """Unit tests for TemplatedPrompt."""

    def test_render_simple(self):
        tp = TemplatedPrompt(template="Hello, {name}!")
        assert tp.render(name="World") == "Hello, World!"

    def test_render_missing_variable_raises(self):
        tp = TemplatedPrompt(template="Hello, {name}! You are {role}.")
        with pytest.raises(KeyError, match="Missing template variables"):
            tp.render(name="Alice")

    def test_get_required_variables(self):
        tp = TemplatedPrompt(template="Classify {text} as {category}.")
        assert tp.get_required_variables() == {"text", "category"}

    def test_render_with_defaults(self):
        tp = TemplatedPrompt(
            template="Hello, {name}! Your role is {role}.",
            variables={"role": "user"},
        )
        result = tp.render(name="Alice")
        assert result == "Hello, Alice! Your role is user."

    def test_kwargs_override_defaults(self):
        tp = TemplatedPrompt(
            template="Hello, {name}!",
            variables={"name": "Default"},
        )
        assert tp.render(name="Override") == "Hello, Override!"

    def test_no_variables_needed(self):
        tp = TemplatedPrompt(template="No variables here.")
        assert tp.render() == "No variables here."

    def test_get_required_variables_empty(self):
        tp = TemplatedPrompt(template="Static text.")
        assert tp.get_required_variables() == set()
