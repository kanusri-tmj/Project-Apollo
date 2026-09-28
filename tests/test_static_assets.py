"""Tests for the favicon and the static assets behind it.

A favicon fails quietly. A link tag that 404s, an ``.ico`` missing its 16 px
frame, or a mistyped content type all still leave every page test passing while
the browser tab shows a broken-image glyph. Worse, an asset excluded by
``.gitignore`` exists locally and is simply absent in production. These tests
pin all of that down.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC = PROJECT_ROOT / "app" / "static"
FAVICON_SVG = STATIC / "favicon.svg"
FAVICON_ICO = STATIC / "favicon.ico"
APPLE_ICON = STATIC / "apple-touch-icon.png"

ICO_SIZES = {(16, 16), (32, 32), (48, 48), (64, 64)}


@pytest.fixture(scope="module")
def client():
    flask_app = pytest.importorskip("app.flask_app")
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


# --------------------------------------------------------------------------- #
# The assets themselves
# --------------------------------------------------------------------------- #
def test_svg_is_well_formed_and_on_palette():
    root = ET.parse(FAVICON_SVG).getroot()
    assert root.tag.endswith("svg")
    assert root.get("viewBox") == "0 0 64 64"

    source = FAVICON_SVG.read_text(encoding="utf-8")
    # Same gradient as the header brand mark, so the tab and the page agree.
    for hex_value in ("#d81f36", "#a8162a"):
        assert hex_value in source, f"{hex_value} missing from the icon gradient"
    assert source.count("<path") == 2, "expected exactly the sigmoid curve and the heart"


def test_ico_holds_every_frame_browsers_ask_for():
    ico = Image.open(FAVICON_ICO)
    try:
        assert set(ico.ico.sizes()) == ICO_SIZES
        ico.size = (16, 16)
        frame = ico.convert("RGBA")
    finally:
        ico.close()

    pixels = np.asarray(frame).astype(int)
    assert pixels.shape == (16, 16, 4)

    visible = pixels[..., 3] > 200
    white = (visible & (pixels[..., :3].min(axis=-1) > 235)).mean()
    red = (visible & (pixels[..., 0] > 120) & (pixels[..., 1] < 90)).mean()

    # The mark has to survive at 16 px: a white heart on the red tile, not a
    # blank square and not a solid blob.
    assert 0.10 < white < 0.35, f"heart covers {white:.1%} of the 16 px frame"
    assert red > 0.55, f"brand tile covers only {red:.1%} of the 16 px frame"


def test_apple_icon_is_square_and_opaque():
    with Image.open(APPLE_ICON) as apple:
        assert apple.size == (180, 180)
        # iOS applies its own rounded mask, so this one must be full-bleed RGB.
        assert apple.mode == "RGB"


# --------------------------------------------------------------------------- #
# Serving them
# --------------------------------------------------------------------------- #
def test_favicon_route_serves_an_ico(client):
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.data[:4] == b"\x00\x00\x01\x00", "not an ICO container"
    assert "icon" in response.headers["Content-Type"]


def test_static_route_serves_the_svg_as_an_image(client):
    response = client.get("/static/favicon.svg")
    assert response.status_code == 200
    assert response.data.lstrip().startswith(b"<svg")
    # A wrong content type here makes some browsers ignore the icon entirely.
    assert response.headers["Content-Type"].startswith("image/svg")


def test_home_page_declares_every_icon(client):
    body = client.get("/").data.decode("utf-8")
    for asset in ("favicon.svg", "favicon.ico", "apple-touch-icon.png"):
        assert asset in body, f"the page does not reference {asset}"
    assert 'rel="icon"' in body
    assert 'rel="apple-touch-icon"' in body
    assert 'name="theme-color"' in body
