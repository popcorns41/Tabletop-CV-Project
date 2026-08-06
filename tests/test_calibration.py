import numpy as np
import pytest

from tabletop_vision.calibration import(
    CharucoBoardSpec,
    create_charuco_board_image,
    millimetres_to_pixels,
)

def test_default_board_dimensios() -> None:
    spec = CharucoBoardSpec()

    assert spec.board_width_mm == pytest.approx(175.0)
    assert spec.board_height_mm == pytest.approx(125.0)
    assert spec.internal_corner_count == 24

def test_millimetres_to_pixels() -> None:
    pixels = millimetres_to_pixels(
        length_mm=25.4,
        dpi=300
    )

    assert pixels == 300

def test_marker_must_be_smaller_than_square() -> None:
    with pytest.raises(ValueError):
        CharucoBoardSpec(
            square_length_mm=25.0,
            marker_length_mm=25.0,
        )

    def test_board_image_is_generated() -> None:
        spec = CharucoBoardSpec()

        image = create_charuco_board_image(
            spec,
            dpi=100,
        )

        expected_width = millimetres_to_pixels(
            spec.board_width_mm,
            100,
        )

        expected_height = millimetres_to_pixels(
            spec.board_height_mm,
            100,
        )

        assert image.shape == (
            expected_height,
            expected_width
        )

        assert image.dtype == np.uint8
        assert image.min() == 0
        assert image.max() == 255