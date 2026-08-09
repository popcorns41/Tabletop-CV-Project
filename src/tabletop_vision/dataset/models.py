from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class DatasetImageMetadata:
    """Metadata associated with one captured dataset image."""

    filename: str
    width: int
    height: int
    timestamp: str

    environment: str | None = None
    session: str | None = None
    target_present: bool = True

@dataclass(frozen=True, slots=True)
class PolygonPoint:
    """One image-space point stored in normalised coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0:
            raise ValueError(
                "Polygon x coordinate must lie in [0, 1]."
            )

        if not 0.0 <= self.y <= 1.0:
            raise ValueError(
                "Polygon y coordinate must lie in [0, 1]."
            )


@dataclass(frozen=True, slots=True)
class InstanceAnnotation:
    """Segmentation annotation for one object instance."""

    class_name: str
    polygon: tuple[PolygonPoint, ...]

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError(
                "class_name must not be empty."
            )

        if len(self.polygon) < 3:
            raise ValueError(
                "A polygon requires at least three points."
            )

@dataclass(frozen=True, slots=True)
class ImageAnnotation:
    """All labelled object instances belonging to one image."""

    filename: str
    instances: tuple[InstanceAnnotation, ...]

    def __post_int__(self) -> None:
        if not self.filename.strip():
            raise ValueError(
                "filename must not be empty."
            )

@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Filename assignments for model training and evaluation."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Errors and warnings found while validating a dataset. """

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors