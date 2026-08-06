from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from tabletop_vision.calibration import (
    CharucoBoardSpec,
    create_charuco_board_image,
)

DEFAULT_OUTPUT_DIRECTORY = Path("data/calibration/board")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Generate a printable ChArUco camera calibration board."
        )
    )

    parser.add_argument(
        "--squares-x",
        type=int,
        default =7,
        help="Number of chessboard squares horizontally. Default: 7",
    )

    parser.add_argument(
        "--squares-y",
        type=int,
        default=5,
        help="Number of chessboard squares vertically. Default: 5",
    )

    parser.add_argument(
        "--square-mm",
        type=float,
        default=25.0,
        help="Physical chessboard square size in millimetres. default: 25",
    )

    parser.add_argument(
        "--marker-mm",
        type=float,
        default=18.0,
        help="Physical ArUco marker size in millimetres. Default: 18",
    )

    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_5X5_100",
        help="OpenCV ArUco dictionary name. Default: DICT_5X5_100"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the generated PNG. Default: 300",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory for generated PNG and PDF files . "
            "Default: data/calibration/board"
        ),
    )

    return parser.parse_args()

def save_png(
        image,
        output_path: Path,
) -> None:
    successful = cv2.imwrite(
        str(output_path),
        image,
    )

    if not successful:
        raise RuntimeError(
            f"Could not save board image to {output_path}."
        )

def create_pdf(
        image,
        spec: CharucoBoardSpec,
        output_path: Path,
) -> None:
    page_width, page_height = A4

    board_width = spec.board_width_mm * mm
    board_height = spec.board_height_mm * mm

    horizontal_page_margin = 15 * mm
    avaliable_width = page_width - 2 * horizontal_page_margin

    if board_width > avaliable_width:
        raise ValueError(
            "The requested board is too wide to fit safely on A4 paper."
        )

    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4,
    )

    pdf.setTitle("ChArUco Camera Calibration Board")
    pdf.setAuthor("Vision-Guided Tabletop Robotics")

    pdf.setFont("Helvetica-Bold", 14)

    pdf.drawCentredString(
        page_width /2,
        page_height - 15 * mm,
        "ChArUco Camera Calibration Board",
    )

    board_x = (page_width - board_width) / 2
    board_top = page_height - 30 * mm
    board_y = board_top - board_height

    encoded_successfully, encoded_image = cv2.imencode(
        ".png",
        image,
    )

    if not encoded_successfully:
        raise RuntimeError(
            "Could not encode the board image for the PDF."
        )

    image_reader = ImageReader(
        BytesIO(encoded_image.tobytes())
    )

    pdf.drawImage(
        image_reader,
        board_x,
        board_y,
        width=board_width,
        height=board_height,
        preserveAspectRatio=True,
        mask="auto",
    )

    reference_length = 100 * mm
    reference_x_start = (
        page_width - reference_length
    ) /2

    reference_x_end = (
        reference_x_start + reference_length
    )

    reference_y = board_y - 20 * mm

    pdf.setLineWidth(0.5)

    pdf.line(
        reference_x_start,
        reference_y,
        reference_x_end,
        reference_y
    )

    tick_height = 2 * mm

    pdf.line(
        reference_x_start,
        reference_y - tick_height,
        reference_x_start,
        reference_y + tick_height
    )

    pdf.line(
        reference_x_end,
        reference_y - tick_height,
        reference_x_end,
        reference_y + tick_height
    )

    pdf.setFont("Helvetica", 9)

    pdf.drawCentredString(
        page_width / 2,
        reference_y - 6 * mm,
        "This line must measure exactly 100 mm after printing",
    )

    board_description = (
        f"Board: {spec.squares_x} * {spec.squares_y} squares"
        f" | square: {spec.square_length_mm: .1f} mm"
        f" | market: {spec.marker_length_mm: .1f} mm"
    )

    pdf.drawCentredString(
        page_width /2,
        reference_y - 12 * mm,
        board_description
    )

    pdf.drawCentredString(
        page_width /2,
        reference_y - 18 * mm,
        f"Dictionary: {spec.dictionary_name}",
    )

    pdf.showPage()
    pdf.save()

def main() -> int:
    arguments = parse_arguments()

    spec = CharucoBoardSpec(
        squares_x=arguments.squares_x,
        squares_y=arguments.squares_y,
        square_length_mm=arguments.square_mm,
        marker_length_mm=arguments.marker_mm,
        dictionary_name=arguments.dictionary,
    )

    arguments.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename_stem= (
        f"charuco_"
        f"{spec.squares_x}x{spec.squares_y}_"
        f"{spec.square_length_mm:g}mm"
    )

    png_path = (
        arguments.output_directory
        /f"{filename_stem}_{arguments.dpi}dpi.png"
    )

    pdf_path= (
        arguments.output_directory
        / f"{filename_stem}_a4.pdf"
    )

    board_image = create_charuco_board_image(
        spec,
        arguments.dpi,
    )

    save_png(
        board_image,
        png_path,
    )

    create_pdf(
        board_image,
        spec,
        pdf_path
    )

    print("ChArUco board generated successfully")
    print(f"  Squares:          {spec.squares_x} × {spec.squares_y}")
    print(f"  Internal corners: {spec.internal_corner_count}")
    print(
        f"  Physical size:    "
        f"{spec.board_width_mm:.1f} × "
        f"{spec.board_height_mm:.1f} mm"
    )
    print(f"  Square size:      {spec.square_length_mm:.1f} mm")
    print(f"  Marker size:      {spec.marker_length_mm:.1f} mm")
    print(f"  Dictionary:       {spec.dictionary_name}")
    print(f"  PNG:              {png_path}")
    print(f"  PDF:              {pdf_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())