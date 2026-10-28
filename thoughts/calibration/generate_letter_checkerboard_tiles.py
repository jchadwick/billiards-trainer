#!/usr/bin/env python3
"""Create a true-size, four-sheet US Letter checkerboard poster.

The active target is 20 x 14 inches with 2-inch squares. Four landscape
Letter pages overlap at the center lines so the pattern itself can register
the sheets before trimming and taping. The generated PDF uses vector squares;
it does not depend on a raster image or a third-party PDF package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile


POINTS_PER_INCH = 72.0
LETTER_LANDSCAPE = (11.0 * POINTS_PER_INCH, 8.5 * POINTS_PER_INCH)
SQUARE_INCHES = 2.0
SQUARE_COLUMNS = 10
SQUARE_ROWS = 7
TARGET_WIDTH = SQUARE_COLUMNS * SQUARE_INCHES
TARGET_HEIGHT = SQUARE_ROWS * SQUARE_INCHES

# (left, top, right, bottom), in target inches with a top-left origin.
# The left/right sheets overlap by 1/2 inch across the x=10-inch grid line.
# The upper/lower sheets overlap by one full square across y=6..8 inches.
TILES = (
    (0.0, 0.0, 10.25, 8.0),
    (9.75, 0.0, 20.0, 8.0),
    (0.0, 6.0, 10.25, 14.0),
    (9.75, 6.0, 20.0, 14.0),
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "pdf"
    / "ball-localization-checkerboard-letter-tiles.pdf"
)


def number(value: float) -> str:
    """Format an exact-enough decimal for a PDF drawing operator."""
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def tile_content(tile: tuple[float, float, float, float], page_number: int) -> bytes:
    """Draw one clipped checkerboard crop on a centered Letter landscape page."""
    x0, y0, x1, y1 = tile
    page_width, page_height = LETTER_LANDSCAPE
    panel_width = (x1 - x0) * POINTS_PER_INCH
    panel_height = (y1 - y0) * POINTS_PER_INCH
    left = (page_width - panel_width) / 2.0
    bottom = (page_height - panel_height) / 2.0

    commands = [
        "q",
        f"{number(left)} {number(bottom)} {number(panel_width)} {number(panel_height)} re W n",
        "0 g",
    ]
    side = SQUARE_INCHES * POINTS_PER_INCH
    for row in range(SQUARE_ROWS):
        for column in range(SQUARE_COLUMNS):
            if (row + column) % 2:
                continue
            square_x = left + (column * SQUARE_INCHES - x0) * POINTS_PER_INCH
            # PDF y grows upward; target y grows down from the board's top.
            square_y = bottom + (y1 - (row + 1) * SQUARE_INCHES) * POINTS_PER_INCH
            commands.append(
                f"{number(square_x)} {number(square_y)} {number(side)} {number(side)} re f"
            )
    commands.append("Q")
    # A small page number sits in the printer-safe part of the left paper
    # margin, outside the active checkerboard.
    commands.extend(("BT /F1 7 Tf 19 306 Td", f"({page_number}) Tj", "ET"))

    return ("\n".join(commands) + "\n").encode("ascii")


def build_pdf(streams: tuple[bytes, ...]) -> bytes:
    """Serialize a minimal PDF 1.4 document with four vector pages."""
    if len(streams) != len(TILES):
        raise ValueError("the letter poster must contain exactly four tiles")

    objects: list[bytes] = [b""]
    page_ids = [3 + 2 * index for index in range(len(streams))]
    kids = " ".join(f"{object_id} 0 R" for object_id in page_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(streams)} >>".encode("ascii")
    )

    page_box = "[0 0 792 612]"
    for index, stream in enumerate(streams):
        page_id = page_ids[index]
        stream_id = page_id + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox {page_box} "
                f"/Resources << /Font << /F1 11 0 R >> >> /Contents {stream_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects[1:], start=1):
        offsets.append(len(result))
        result.extend(f"{object_id} 0 obj\n".encode("ascii"))
        result.extend(body)
        result.extend(b"\nendobj\n")

    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects)}\n".encode("ascii"))
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend(
        (
            f"trailer\n<< /Size {len(objects)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(result)


def validate_layout() -> None:
    """Guard the paper size, board coverage, and intended registration overlaps."""
    if TARGET_WIDTH != 20.0 or TARGET_HEIGHT != 14.0:
        raise AssertionError("the physical checkerboard must remain 20 x 14 inches")
    page_width_in, page_height_in = 11.0, 8.5
    for x0, y0, x1, y1 in TILES:
        if x1 - x0 > page_width_in or y1 - y0 > page_height_in:
            raise AssertionError("tile is larger than a US Letter landscape sheet")
    if TILES[0][2] - TILES[1][0] != 0.5:
        raise AssertionError("left/right tiles must overlap by exactly 1/2 inch")
    if TILES[0][3] - TILES[2][1] != SQUARE_INCHES:
        raise AssertionError("top/bottom tiles must overlap by one square")
    if min(tile[0] for tile in TILES) != 0.0 or max(tile[2] for tile in TILES) != TARGET_WIDTH:
        raise AssertionError("horizontal tiles must cover the full target width")
    if min(tile[1] for tile in TILES) != 0.0 or max(tile[3] for tile in TILES) != TARGET_HEIGHT:
        raise AssertionError("vertical tiles must cover the full target height")


def self_test() -> None:
    validate_layout()
    streams = tuple(tile_content(tile, index + 1) for index, tile in enumerate(TILES))
    pdf = build_pdf(streams)
    if not pdf.startswith(b"%PDF-1.4") or b"/Count 4" not in pdf:
        raise AssertionError("output is not a four-page PDF")
    if pdf.count(b"/MediaBox [0 0 792 612]") != 4:
        raise AssertionError("each page must be US Letter landscape")
    if pdf.count(b" re f") < 40:
        raise AssertionError("checkerboard vector squares are missing")
    with tempfile.TemporaryDirectory(prefix="letter-checkerboard-test-") as directory:
        path = Path(directory) / "poster.pdf"
        path.write_bytes(pdf)
        if path.stat().st_size < 1000:
            raise AssertionError("generated PDF is unexpectedly small")
    print("Four Letter landscape tiles, 2-inch squares, overlaps, and PDF structure passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    validate_layout()
    streams = tuple(tile_content(tile, index + 1) for index, tile in enumerate(TILES))
    output = build_pdf(streams)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(f"Wrote {args.output} ({len(TILES)} US Letter landscape pages, 100% scale)")


if __name__ == "__main__":
    main()
