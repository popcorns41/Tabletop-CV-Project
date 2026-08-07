from __future__ import annotations

import cv2
import numpy as np

def create_kernel(
        size: int,
) -> np.ndarray:
    """Create a square moprhological kernel."""

    if size <= 0:
        raise ValueError(
            "Kernel size must be positive."
        )

    if size % 2 == 0:
        raise ValueError(
            "Kernel size must be odd."
        )

    return np.ones(
        (size,size),
        dtype=np.uint8,
    )

def apply_opening(
        mask: np.ndarray,
        kernel: np.ndarray,
) -> np.ndarray:
    """Remove small isolated forground regions."""

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

def apply_closing(
        mask: np.ndarray,
        kernel:np.ndarray,
) -> np.ndarray:
    """Fill small holes and gaps in foregrond regions."""

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

def clean_mask(
        mask: np.ndarray,
        kernel_size: int = 3,
) -> np.ndarray:
    """Apply basic moprhological cleanup to a binary mask."""

    kernel = create_kernel(kernel_size)

    opened = apply_opening(
        mask,
        kernel,
    )

    return apply_closing(
        opened,
        kernel,
    )