"""Common Pydantic models shared across all whitepaper phases."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message."""

    role: str = Field(description="One of: system, user, assistant, tool")
    content: str = Field(description="Text content of the message")


class ToolCall(BaseModel):
    """Represents a tool/function call requested by the LLM."""

    name: str = Field(description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )


class ToolResult(BaseModel):
    """Result returned after executing a tool."""

    name: str = Field(description="Name of the tool that was executed")
    result: str = Field(description="Stringified result of the tool execution")


class RuleMatch(BaseModel):
    """Outcome of evaluating an input against a static rule."""

    rule_name: str = Field(description="Name of the rule that matched")
    matched: bool = Field(description="Whether the rule fired")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )


class SanitizationResult(BaseModel):
    """Outcome of input sanitization."""

    is_clean: bool = Field(description="Whether the input passed all checks")
    blocked_by: str | None = Field(
        default=None, description="Name of the check that blocked"
    )
    original_input: str = Field(description="The original input text")
    sanitized_input: str = Field(description="The cleaned input text")


class ValidationResult(BaseModel):
    """Outcome of output validation."""

    is_valid: bool = Field(description="Whether the output passed all checks")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    sanitized_output: str = Field(description="The output after redaction/cleanup")


class PromptVersion(BaseModel):
    """A versioned prompt with metadata."""

    version: int = Field(description="Monotonically increasing version number")
    name: str = Field(description="Prompt identifier (e.g., 'classify-intent')")
    content: str = Field(description="The full prompt text")
    content_hash: str = Field(description="SHA-256 hash of the content for dedup")
    description: str = Field(default="", description="What changed in this version")
    created_at: float = Field(description="Unix timestamp of creation")
    is_template: bool = Field(
        default=False,
        description="Whether the content is a template with {variable} placeholders",
    )
    models_tested: list[str] = Field(
        default_factory=list, description="Models this version was validated against"
    )


class DriftReport(BaseModel):
    """Result of a drift detection run."""

    prompt_name: str = Field(description="Name of the prompt tested")
    prompt_version: int = Field(description="Version of the prompt tested")
    test_input: str = Field(description="The test input used")
    baseline_output: str = Field(description="Expected output from baseline run")
    current_output: str = Field(description="Output from current run")
    similarity_score: float = Field(
        ge=-1.0, le=1.0, description="Cosine similarity between baseline and current"
    )
    drifted: bool = Field(description="True if similarity < threshold")
