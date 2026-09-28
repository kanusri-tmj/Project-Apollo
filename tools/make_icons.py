"""Regenerate the browser icons from ``app/static/favicon.svg``.

The SVG is the single source of truth. A favicon still has to ship as ready-made
raster sizes as well as vector: browsers keep requesting ``/favicon.ico``, and
iOS wants an opaque 180 px PNG for the home screen. Rather than hand-drawing
three copies and letting them drift apart, this script reads the SVG and
rasterises it::

    python tools/make_icons.py

It parses only the small subset of SVG the icon actually uses — one rounded
rect, one linear gradient, and paths built from absolute ``M``/``C``/``Z``
commands — so it needs nothing beyond Pillow and numpy, which matplotlib
already brings in. No new dependency, and it runs on any platform.

Workflow: edit the SVG, re-run this, commit both.
"""

from __future__ import annotations

import io
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "app" / "static" / "favicon.svg"
ICO_PATH = ROOT / "app" / "static" / "favicon.ico"
APPLE_PATH = ROOT / "app" / "static" / "apple-touch-icon.png"

DESIGN = 64          # the SVG viewBox is 64 x 64
SUPERSAMPLE = 16     # render at 1024 x 1024, then downsample with Lanczos
CUBIC_STEPS = 48     # segments per cubic bezier; plenty at 1024 px
ICO_SIZES = (16, 32, 48, 64)
APPLE_SIZE = 180     # iOS home-screen icon

# Gradient direction, mirroring x1/y1/x2/y2 in the SVG.
GRADIENT_X2, GRADIENT_Y2 = 0.7, 1.0


# --------------------------------------------------------------------------- #
# SVG parsing (deliberately minimal)
# --------------------------------------------------------------------------- #
def _attrs(tag_body: str) -> dict[str, str]:
    return dict(re.findall(r'([\w-]+)="([^"]*)"', tag_body))


def _elements(svg: str) -> list[tuple[str, dict[str, str]]]:
    return [
        (name, _attrs(body))
        for name, body in re.findall(r"<(rect|path)\b([^>]*)/>", svg, re.S)
    ]


def _parse_path(d: str) -> list[list[tuple[float, float]]]:
    """Flatten ``M``/``C``/``Z`` path data into polylines.

    Absolute coordinates only — that is all the icon uses, and supporting
    relative commands would be more parser than this task deserves.
    """
    tokens = re.findall(r"[MCZ]|-?\d*\.\d+|-?\d+", d)
    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    start: tuple[float, float] | None = None

    index = 0
    while index < len(tokens):
        command = tokens[index]
        index += 1

        if command == "M":
            if current:
                subpaths.append(current)
            start = (float(tokens[index]), float(tokens[index + 1]))
            current = [start]
            index += 2
        elif command == "C":
            numbers = [float(t) for t in tokens[index:index + 6]]
            index += 6
            p0 = current[-1]
            current.extend(_cubic(p0, numbers[0:2], numbers[2:4], numbers[4:6]))
        elif command == "Z":
            if start is not None and current[-1] != start:
                current.append(start)

    if current:
        subpaths.append(current)
    return subpaths


def _cubic(p0, p1, p2, p3, steps: int = CUBIC_STEPS) -> list[tuple[float, float]]:
    """Sample a cubic bezier, excluding ``p0`` (already in the polyline)."""
    t = np.linspace(0.0, 1.0, steps + 1)[1:, None]
    omt = 1.0 - t
    points = (
        (omt ** 3) * np.array(p0)
        + 3 * (omt ** 2) * t * np.array(p1)
        + 3 * omt * (t ** 2) * np.array(p2)
        + (t ** 3) * np.array(p3)
    )
    return [(float(x), float(y)) for x, y in points]


def load_icon(svg: str) -> dict:
    """Pull the tile, curve and heart geometry out of the SVG document."""
    elements = _elements(svg)
    rect = next(attrs for name, attrs in elements if name == "rect")
    paths = [attrs for name, attrs in elements if name == "path"]

    curve_attrs = next(attrs for attrs in paths if "stroke-opacity" in attrs)
    heart_attrs = next(attrs for attrs in paths if attrs.get("fill") == "#ffffff")

    stops = [
        tuple(int(hex_value[i:i + 2], 16) for i in (1, 3, 5))
        for hex_value in re.findall(r'stop-color="(#[0-9a-fA-F]{6})"', svg)
    ]
    if len(stops) != 2:
        raise ValueError(f"Expected a two-stop gradient, found {len(stops)}")

    return {
        "radius": float(rect["rx"]),
        "stops": stops,
        "curve": max(_parse_path(curve_attrs["d"]), key=len),
        "curve_width": float(curve_attrs["stroke-width"]),
        "curve_alpha": round(float(curve_attrs["stroke-opacity"]) * 255),
        "heart": max(_parse_path(heart_attrs["d"]), key=len),
    }


# --------------------------------------------------------------------------- #
# Rasterising
# --------------------------------------------------------------------------- #
def _gradient(size: int, stops: list[tuple[int, int, int]]) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    t = np.clip((xx * GRADIENT_X2 + yy * GRADIENT_Y2) / ((size - 1) * (GRADIENT_X2 + GRADIENT_Y2)), 0, 1)
    t = t[..., None]
    low = np.array(stops[0], dtype=float)
    high = np.array(stops[1], dtype=float)
    return Image.fromarray((low + (high - low) * t).astype("uint8"), "RGB").convert("RGBA")


def render_master(icon: dict, radius: float, with_curve: bool = True) -> Image.Image:
    """Render the icon at SUPERSAMPLE x DESIGN and return it as RGBA."""
    size = DESIGN * SUPERSAMPLE
    factor = size / DESIGN

    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if radius > 0:
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, size - 1, size - 1], radius=radius * factor, fill=255
        )
    else:
        mask = Image.new("L", (size, size), 255)
    tile.paste(_gradient(size, icon["stops"]), (0, 0), mask)

    if with_curve:
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        points = [(x * factor, y * factor) for x, y in icon["curve"]]
        colour = (255, 255, 255, icon["curve_alpha"])
        width = max(1, round(icon["curve_width"] * factor))
        draw.line(points, fill=colour, width=width, joint="curve")
        cap = icon["curve_width"] * factor / 2
        for x, y in (points[0], points[-1]):  # round caps
            draw.ellipse([x - cap, y - cap, x + cap, y + cap], fill=colour)
        tile = Image.alpha_composite(tile, layer)

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(layer).polygon(
        [(x * factor, y * factor) for x, y in icon["heart"]], fill=(255, 255, 255, 255)
    )
    return Image.alpha_composite(tile, layer)


# --------------------------------------------------------------------------- #
# ICO container
# --------------------------------------------------------------------------- #
def write_ico(path: Path, frames: list[Image.Image]) -> None:
    """Write an ICO whose entries are PNG-compressed.

    Pillow's own ICO writer resamples with bicubic from whatever image you hand
    it; downscaling 1024 -> 16 that way rings badly. Rendering each frame with
    Lanczos and assembling the container here keeps a 16 px favicon crisp.
    """
    payloads = []
    for frame in frames:
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG", optimize=True)
        payloads.append(buffer.getvalue())

    header = struct.pack("<HHH", 0, 1, len(frames))
    offset = 6 + 16 * len(frames)
    directory, blob = b"", b""
    for frame, payload in zip(frames, payloads):
        width = frame.width if frame.width < 256 else 0
        height = frame.height if frame.height < 256 else 0
        directory += struct.pack(
            "<BBBBHHII", width, height, 0, 0, 1, 32, len(payload), offset
        )
        blob += payload
        offset += len(payload)

    path.write_bytes(header + directory + blob)


# --------------------------------------------------------------------------- #
def main() -> None:
    icon = load_icon(SVG_PATH.read_text(encoding="utf-8"))
    print(f"[icons] source: {SVG_PATH.name}  ({len(icon['heart'])} heart points, "
          f"{len(icon['curve'])} curve points)")

    master = render_master(icon, radius=icon["radius"])
    frames = [master.resize((s, s), Image.LANCZOS) for s in ICO_SIZES]
    write_ico(ICO_PATH, frames)
    print(f"[icons] wrote {ICO_PATH.name}  sizes={list(ICO_SIZES)}  "
          f"{ICO_PATH.stat().st_size:,} bytes")

    # iOS applies its own mask, so this one is full-bleed and opaque.
    apple = render_master(icon, radius=0.0, with_curve=False).resize(
        (APPLE_SIZE, APPLE_SIZE), Image.LANCZOS
    )
    apple.convert("RGB").save(APPLE_PATH, optimize=True)
    print(f"[icons] wrote {APPLE_PATH.name}  {APPLE_SIZE}x{APPLE_SIZE}  "
          f"{APPLE_PATH.stat().st_size:,} bytes")

    for size, frame in zip(ICO_SIZES, frames):
        pixels = np.asarray(frame).astype(int)
        visible = pixels[..., 3] > 200
        rgb = pixels[..., :3]
        white = (visible & (rgb.min(axis=-1) > 235)).mean()
        red = (visible & (rgb[..., 0] > 120) & (rgb[..., 1] < 90)).mean()
        print(f"[icons]   {size:>3}px: heart {white * 100:5.1f}%  red {red * 100:5.1f}%")


if __name__ == "__main__":
    main()
