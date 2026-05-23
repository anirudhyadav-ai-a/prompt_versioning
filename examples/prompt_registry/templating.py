"""Prompt templating -- variable substitution and Jinja2 support.

Phase 7 | Prompt Registry -- Templating
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TemplatedPrompt:
    """A prompt template with {variable} placeholders."""

    template: str
    variables: dict[str, str] = field(default_factory=dict)
    _jinja2: bool = field(default=False, repr=False)

    def render(self, **kwargs: str) -> str:
        """Render the template with provided variables.

        Merges kwargs with self.variables (kwargs take precedence).
        For Jinja2 templates (created via from_jinja2()), delegates to
        _render_jinja2() which uses {{ variable }} syntax.
        Raises KeyError if a required {variable} is missing (str.format mode only).
        """
        if self._jinja2:
            return self._render_jinja2(**kwargs)
        merged = {**self.variables, **kwargs}
        required = self.get_required_variables()
        missing = required - set(merged.keys())
        if missing:
            raise KeyError(f"Missing template variables: {missing}")
        return self.template.format(**merged)

    def get_required_variables(self) -> set[str]:
        """Return the set of variable names required by this template."""
        return set(re.findall(r"\{(\w+)\}", self.template))

    @classmethod
    def from_jinja2(cls, template_str: str) -> TemplatedPrompt:
        """Create from a Jinja2 template string (requires jinja2 installed).

        The render() method will use jinja2.Template internally.
        """
        return cls(template=template_str, _jinja2=True)

    def _render_jinja2(self, **kwargs: str) -> str:
        """Render using Jinja2 engine."""
        from jinja2 import Template  # type: ignore[import-untyped]

        merged = {**self.variables, **kwargs}
        tmpl = Template(self.template)
        return tmpl.render(**merged)
