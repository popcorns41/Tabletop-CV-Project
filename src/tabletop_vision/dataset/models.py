from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class DataImageMetadata:
    """Metadata associated with one captured dataset image."""

    filename: str
    width: int
    height: int
    timestamp: str
    environments: str | None = None