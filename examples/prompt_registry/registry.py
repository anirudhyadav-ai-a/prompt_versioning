"""Prompt registry — version, diff, and rollback prompts as code.

Phase 7 | Prompt Registry
"""

from __future__ import annotations

import difflib
import hashlib
import time

from prompt_versioning.utils.llm_client import LLMClient
from prompt_versioning.utils.logging_utils import get_logger
from prompt_versioning.utils.math_utils import cosine_similarity
from prompt_versioning.utils.types import PromptVersion

logger = get_logger(__name__)


def _content_hash(content: str) -> str:
    """Compute SHA-256 hash of prompt content."""
    return hashlib.sha256(content.encode()).hexdigest()


class PromptRegistry:
    """In-memory prompt registry with versioning, diff, and rollback."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._store: dict[str, list[PromptVersion]] = {}
        self._client = client

    def register(
        self,
        name: str,
        content: str,
        description: str = "",
        *,
        force: bool = False,
        template: bool = False,
    ) -> PromptVersion:
        """Create a new prompt version (skips if content unchanged unless *force*)."""
        content_hash = _content_hash(content)
        history = self._store.setdefault(name, [])

        if not force and history and history[-1].content_hash == content_hash:
            logger.info("No change detected for '%s' — skipping", name)
            return history[-1]

        version = PromptVersion(
            version=len(history) + 1,
            name=name,
            content=content,
            content_hash=content_hash,
            description=description,
            created_at=time.time(),
            is_template=template,
        )
        history.append(version)
        logger.info("Registered '%s' v%d", name, version.version)
        return version

    def get_current(self, name: str) -> PromptVersion | None:
        """Return the latest version for a prompt name."""
        history = self._store.get(name, [])
        return history[-1] if history else None

    def get_version(self, name: str, version: int) -> PromptVersion | None:
        """Return a specific version by number."""
        for pv in self._store.get(name, []):
            if pv.version == version:
                return pv
        return None

    def get_history(self, name: str) -> list[PromptVersion]:
        """Return all versions for a prompt name."""
        return list(self._store.get(name, []))

    def rollback(self, name: str, to_version: int) -> PromptVersion:
        """Create a new version with the content from *to_version*."""
        target = self.get_version(name, to_version)
        if target is None:
            raise ValueError(f"Version {to_version} not found for '{name}'")
        return self.register(
            name, target.content, f"Rollback to v{to_version}", force=True
        )

    def mark_tested(self, name: str, version: int, model_label: str) -> None:
        """Record that a prompt version was tested against a specific model."""
        pv = self.get_version(name, version)
        if pv is None:
            raise ValueError(f"Version {version} not found for '{name}'")
        if model_label not in pv.models_tested:
            pv.models_tested.append(model_label)
            logger.info(
                "Marked '%s' v%d as tested against '%s'", name, version, model_label
            )

    async def semantic_diff(self, name: str, v1: int, v2: int) -> float:
        """Embed both versions and return cosine similarity."""
        pv1 = self.get_version(name, v1)
        pv2 = self.get_version(name, v2)
        if pv1 is None or pv2 is None:
            raise ValueError(f"Version not found for '{name}'")
        if self._client is None:
            raise ValueError("LLMClient required for semantic diff")
        emb1 = await self._client.embed(pv1.content)
        emb2 = await self._client.embed(pv2.content)
        return cosine_similarity(emb1, emb2)

    def text_diff(self, name: str, v1: int, v2: int) -> dict:
        """Return added/removed lines between two versions in document order.

        Uses difflib.unified_diff so duplicate lines and ordering are preserved,
        unlike set subtraction which loses both.
        """
        pv1 = self.get_version(name, v1)
        pv2 = self.get_version(name, v2)
        if pv1 is None or pv2 is None:
            raise ValueError(f"Version not found for '{name}'")
        lines1 = pv1.content.splitlines(keepends=True)
        lines2 = pv2.content.splitlines(keepends=True)
        added: list[str] = []
        removed: list[str] = []
        for line in difflib.unified_diff(lines1, lines2, lineterm=""):
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:].rstrip("\n"))
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:].rstrip("\n"))
        return {
            "added_lines": added,
            "removed_lines": removed,
        }
