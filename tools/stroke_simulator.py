"""Stroke Simulator — CLI tool for autonomous pipeline testing.

Converts text expressions (e.g., "2x + 3 = 7") into realistic stroke paths
using font glyph tracing, sends them over WebSocket to the running server,
and verifies clustering/transcription results via the REST API.

Usage:
    uv run python tools/stroke_simulator.py "2x + 3 = 7" --verify
    uv run python tools/stroke_simulator.py "2x + 3 = 7" "x = 2" --verify --expected-lines 2
    uv run python tools/stroke_simulator.py "f(x) = x^2" --preview
    uv run python tools/stroke_simulator.py "a + b" --clear --session-id test-1 --verify
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
import uuid
from pathlib import Path

import websockets
import httpx

from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen

# Allow imports from project root when run via `uv run python tools/stroke_simulator.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_FONT = "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf"
DEFAULT_CHAR_HEIGHT = 35.0
DEFAULT_JITTER = 1.0
DEFAULT_LINE_SPACING = 100.0
DEFAULT_CHAR_SPACING = 5.0
BEZIER_STEPS = 20


# ---------------------------------------------------------------------------
# Bezier helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _cubic_bezier(p0, p1, p2, p3, steps: int = BEZIER_STEPS) -> list[tuple[float, float]]:
    """Evaluate cubic Bezier curve at `steps` evenly-spaced t values."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _quadratic_bezier(p0, p1, p2, steps: int = BEZIER_STEPS) -> list[tuple[float, float]]:
    """Evaluate quadratic Bezier curve at `steps` evenly-spaced t values."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**2 * p0[0] + 2 * u * t * p1[0] + t**2 * p2[0]
        y = u**2 * p0[1] + 2 * u * t * p1[1] + t**2 * p2[1]
        points.append((x, y))
    return points


# ---------------------------------------------------------------------------
# Math symbol fallback generators
# ---------------------------------------------------------------------------

def _generate_plus(cx: float, cy: float, size: float) -> list[dict]:
    """Generate + as two crossing strokes."""
    half = size / 2
    h_stroke = {"points": [{"x": cx - half, "y": cy}, {"x": cx + half, "y": cy}]}
    v_stroke = {"points": [{"x": cx, "y": cy - half}, {"x": cx, "y": cy + half}]}
    return [h_stroke, v_stroke]


def _generate_minus(cx: float, cy: float, size: float) -> list[dict]:
    """Generate - as a horizontal stroke."""
    half = size / 2
    return [{"points": [{"x": cx - half, "y": cy}, {"x": cx + half, "y": cy}]}]


def _generate_equals(cx: float, cy: float, size: float) -> list[dict]:
    """Generate = as two horizontal strokes."""
    half = size / 2
    gap = size * 0.2
    top = {"points": [{"x": cx - half, "y": cy - gap}, {"x": cx + half, "y": cy - gap}]}
    bot = {"points": [{"x": cx - half, "y": cy + gap}, {"x": cx + half, "y": cy + gap}]}
    return [top, bot]


def _generate_lparen(cx: float, cy: float, size: float) -> list[dict]:
    """Generate ( as an arc."""
    half = size / 2
    points = []
    for i in range(BEZIER_STEPS + 1):
        t = i / BEZIER_STEPS
        angle = math.pi * 0.7 * (t - 0.5)
        x = cx + half * 0.4 * math.sin(angle)
        y = cy - half + t * size
        points.append({"x": x, "y": y})
    return [{"points": points}]


def _generate_rparen(cx: float, cy: float, size: float) -> list[dict]:
    """Generate ) as an arc."""
    half = size / 2
    points = []
    for i in range(BEZIER_STEPS + 1):
        t = i / BEZIER_STEPS
        angle = math.pi * 0.7 * (t - 0.5)
        x = cx - half * 0.4 * math.sin(angle)
        y = cy - half + t * size
        points.append({"x": x, "y": y})
    return [{"points": points}]


SYMBOL_GENERATORS = {
    "+": _generate_plus,
    "-": _generate_minus,
    "=": _generate_equals,
    "(": _generate_lparen,
    ")": _generate_rparen,
}


# ---------------------------------------------------------------------------
# GlyphRenderer
# ---------------------------------------------------------------------------

class GlyphRenderer:
    """Loads a TTF font and converts glyph outlines to stroke point arrays."""

    def __init__(self, font_path: str = DEFAULT_FONT, char_height: float = DEFAULT_CHAR_HEIGHT):
        self.font = TTFont(font_path)
        self.glyph_set = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.char_height = char_height

        # Compute scale from font units to canvas pixels
        units_per_em = self.font["head"].unitsPerEm
        self.scale = char_height / units_per_em

    def _get_glyph_outline(self, char: str) -> tuple[list[tuple], float] | None:
        """Get recorded pen operations and advance width for a character.

        Returns None if the character is not in the font.
        """
        code = ord(char)
        if code not in self.cmap:
            return None
        glyph_name = self.cmap[code]
        pen = RecordingPen()
        self.glyph_set[glyph_name].draw(pen)
        advance = self.glyph_set[glyph_name].width
        return pen.value, advance

    def render_char(self, char: str, origin_x: float, origin_y: float,
                    jitter: float = DEFAULT_JITTER) -> tuple[list[dict], float]:
        """Render a single character at (origin_x, origin_y).

        Returns (strokes, advance_width) where strokes is a list of
        {"points": [{"x": ..., "y": ...}]} dicts and advance_width is the
        horizontal distance to the next character's origin.
        """
        # Space character — no strokes, just advance
        if char == " ":
            return [], self.char_height * 0.5

        # Math symbol fallback
        if char in SYMBOL_GENERATORS:
            cx = origin_x + self.char_height * 0.4
            cy = origin_y
            strokes = SYMBOL_GENERATORS[char](cx, cy, self.char_height * 0.6)
            if jitter > 0:
                for stroke in strokes:
                    for pt in stroke["points"]:
                        pt["x"] += random.gauss(0, jitter)
                        pt["y"] += random.gauss(0, jitter)
            return strokes, self.char_height * 0.8

        outline = self._get_glyph_outline(char)
        if outline is None:
            # Unknown char — render as a small dot
            return [{"points": [{"x": origin_x, "y": origin_y}]}], self.char_height * 0.5

        operations, advance = outline
        advance_px = advance * self.scale

        # Convert pen operations to strokes
        strokes: list[dict] = []
        current_points: list[tuple[float, float]] = []
        cursor = (0.0, 0.0)

        for op_type, args in operations:
            if op_type == "moveTo":
                # Start a new stroke contour
                if len(current_points) >= 2:
                    strokes.append(self._points_to_stroke(current_points, origin_x, origin_y, jitter))
                current_points = [(args[0][0], args[0][1])]
                cursor = (args[0][0], args[0][1])

            elif op_type == "lineTo":
                current_points.append((args[0][0], args[0][1]))
                cursor = (args[0][0], args[0][1])

            elif op_type == "curveTo":
                # Cubic Bezier: cursor → cp1, cp2, end
                cp1, cp2, end = args
                bezier_pts = _cubic_bezier(cursor, cp1, cp2, end)
                current_points.extend(bezier_pts[1:])  # skip duplicate start
                cursor = (end[0], end[1])

            elif op_type == "qCurveTo":
                # Quadratic Bezier (possibly with implicit on-curve points)
                if len(args) == 2:
                    cp, end = args
                    bezier_pts = _quadratic_bezier(cursor, cp, end)
                    current_points.extend(bezier_pts[1:])
                    cursor = (end[0], end[1])
                else:
                    # TrueType implicit on-curve: pairs of off-curve points with
                    # implicit on-curve midpoints between them
                    for i in range(len(args) - 1):
                        cp = args[i]
                        if i < len(args) - 2:
                            # Implicit on-curve point between two off-curve points
                            next_cp = args[i + 1]
                            end = ((cp[0] + next_cp[0]) / 2, (cp[1] + next_cp[1]) / 2)
                        else:
                            end = args[i + 1]
                        bezier_pts = _quadratic_bezier(cursor, cp, end)
                        current_points.extend(bezier_pts[1:])
                        cursor = (end[0], end[1])

            elif op_type == "closePath" or op_type == "endPath":
                if len(current_points) >= 2:
                    strokes.append(self._points_to_stroke(current_points, origin_x, origin_y, jitter))
                current_points = []

        # Flush any remaining points
        if len(current_points) >= 2:
            strokes.append(self._points_to_stroke(current_points, origin_x, origin_y, jitter))

        return strokes, advance_px

    def _points_to_stroke(self, font_points: list[tuple[float, float]],
                          origin_x: float, origin_y: float,
                          jitter: float) -> dict:
        """Convert font-unit points to a canvas stroke dict.

        Font coordinates: y increases upward.
        Canvas coordinates: y increases downward. We flip y by negating it.
        """
        points = []
        for fx, fy in font_points:
            x = origin_x + fx * self.scale
            y = origin_y - fy * self.scale  # flip y
            if jitter > 0:
                x += random.gauss(0, jitter)
                y += random.gauss(0, jitter)
            points.append({"x": round(x, 2), "y": round(y, 2)})
        return {"points": points}


# ---------------------------------------------------------------------------
# ExpressionLayout
# ---------------------------------------------------------------------------

class ExpressionLayout:
    """Converts text expressions to positioned strokes in canvas coordinates."""

    def __init__(self, renderer: GlyphRenderer,
                 origin_x: float = 100.0, origin_y: float = 200.0,
                 char_spacing: float = DEFAULT_CHAR_SPACING,
                 line_spacing: float = DEFAULT_LINE_SPACING):
        self.renderer = renderer
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.char_spacing = char_spacing
        self.line_spacing = line_spacing

    def layout_expression(self, text: str) -> list[dict]:
        """Convert multi-line text to canvas strokes.

        Lines are separated by \\n. Each line is rendered left-to-right,
        character by character, with line_spacing between lines.
        """
        lines = text.split("\n")
        all_strokes: list[dict] = []

        for line_idx, line in enumerate(lines):
            cursor_x = self.origin_x
            cursor_y = self.origin_y + line_idx * self.line_spacing

            for char in line:
                char_strokes, advance = self.renderer.render_char(char, cursor_x, cursor_y)
                all_strokes.extend(char_strokes)
                cursor_x += advance + self.char_spacing

        return all_strokes


# ---------------------------------------------------------------------------
# StrokeSimulator
# ---------------------------------------------------------------------------

class StrokeSimulator:
    """Connects to the Reef server and sends/verifies strokes."""

    def __init__(self, server_url: str = "ws://localhost:8000",
                 session_id: str | None = None):
        self.server_url = server_url
        self.session_id = session_id or f"sim-{uuid.uuid4().hex[:8]}"
        self.ws = None
        # Derive HTTP URL from WS URL
        self.http_url = server_url.replace("ws://", "http://").replace("wss://", "https://")

    async def connect(self):
        """Open WebSocket connection to /ws/strokes."""
        url = f"{self.server_url}/ws/strokes?session_id={self.session_id}&user_id=simulator"
        self.ws = await websockets.connect(url)
        print(f"[sim] connected to {url}")

    async def send_strokes(self, strokes: list[dict], page: int = 1) -> None:
        """Send a batch of strokes and wait for ack."""
        if not self.ws:
            await self.connect()

        msg = {
            "type": "strokes",
            "session_id": self.session_id,
            "page": page,
            "strokes": strokes,
            "event_type": "draw",
            "deleted_count": 0,
            "user_id": "simulator",
        }
        await self.ws.send(json.dumps(msg))
        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "ack":
            print(f"[sim] unexpected response: {data}")
        else:
            print(f"[sim] ack received ({len(strokes)} strokes sent)")

    async def verify(self, expected_lines: int | None = None,
                     timeout: float = 30.0, poll_interval: float = 1.0) -> dict:
        """Poll GET /api/stroke-logs until transcriptions appear.

        Returns the API response dict.
        """
        url = f"{self.http_url}/api/stroke-logs?session_id={self.session_id}"
        start = time.time()

        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                resp = await client.get(url)
                data = resp.json()

                transcriptions = data.get("transcriptions", {})
                cluster_order = data.get("cluster_order", [])

                if transcriptions:
                    n_clusters = len(cluster_order)
                    print(f"[sim] {n_clusters} cluster(s) transcribed:")
                    for label in cluster_order:
                        print(f"  cluster {label}: {transcriptions.get(label, '(empty)')}")

                    if expected_lines is not None and n_clusters != expected_lines:
                        print(f"[sim] WARNING: expected {expected_lines} lines, got {n_clusters}")
                    else:
                        print("[sim] verification passed")
                    return data

                await asyncio.sleep(poll_interval)

        print(f"[sim] timeout after {timeout}s — no transcriptions found")
        return data

    async def clear(self) -> None:
        """Delete all stroke logs for this session via REST API."""
        url = f"{self.http_url}/api/stroke-logs?session_id={self.session_id}"
        async with httpx.AsyncClient() as client:
            resp = await client.delete(url)
            data = resp.json()
            print(f"[sim] cleared session {self.session_id}: {data}")

    async def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.ws = None


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_strokes(strokes: list[dict], session_id: str) -> str:
    """Render strokes to a PNG file and return the file path."""
    from lib.stroke_renderer import render_strokes

    png_bytes = render_strokes(strokes)
    out_path = Path(__file__).parent / f"preview_{session_id}.png"
    out_path.write_bytes(png_bytes)
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Stroke Simulator — generate and send handwriting strokes to Reef server"
    )
    parser.add_argument("expressions", nargs="*",
                        help="Text expressions to render (each becomes a line)")
    parser.add_argument("--server", default="ws://localhost:8000",
                        help="WebSocket server URL (default: ws://localhost:8000)")
    parser.add_argument("--page", type=int, default=1,
                        help="Page number (default: 1)")
    parser.add_argument("--font", default=DEFAULT_FONT,
                        help="Path to TTF font file")
    parser.add_argument("--char-height", type=float, default=DEFAULT_CHAR_HEIGHT,
                        help=f"Character height in pixels (default: {DEFAULT_CHAR_HEIGHT})")
    parser.add_argument("--jitter", type=float, default=DEFAULT_JITTER,
                        help=f"Gaussian jitter sigma in pixels (default: {DEFAULT_JITTER})")
    parser.add_argument("--line-spacing", type=float, default=DEFAULT_LINE_SPACING,
                        help=f"Vertical spacing between lines (default: {DEFAULT_LINE_SPACING})")
    parser.add_argument("--char-spacing", type=float, default=DEFAULT_CHAR_SPACING,
                        help=f"Horizontal spacing between characters (default: {DEFAULT_CHAR_SPACING})")
    parser.add_argument("--verify", action="store_true",
                        help="Poll server for transcription results after sending")
    parser.add_argument("--expected-lines", type=int, default=None,
                        help="Expected number of cluster lines (used with --verify)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear session data before sending strokes")
    parser.add_argument("--session-id", default=None,
                        help="Session ID (default: sim-<random>)")
    parser.add_argument("--preview", action="store_true",
                        help="Render strokes to PNG file (no server connection)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Verification timeout in seconds (default: 30)")

    args = parser.parse_args()

    if not args.expressions:
        parser.error("At least one expression is required")

    # Join multiple expressions with newlines for multi-line layout
    text = "\n".join(args.expressions)

    # Render glyphs to strokes
    renderer = GlyphRenderer(font_path=args.font, char_height=args.char_height)
    layout = ExpressionLayout(
        renderer,
        char_spacing=args.char_spacing,
        line_spacing=args.line_spacing,
    )
    strokes = layout.layout_expression(text)
    print(f"[sim] generated {len(strokes)} strokes for: {text!r}")

    session_id = args.session_id or f"sim-{uuid.uuid4().hex[:8]}"

    # Preview mode — render to PNG and exit
    if args.preview:
        out_path = preview_strokes(strokes, session_id)
        print(f"[sim] preview saved to {out_path}")
        return

    # Server mode
    sim = StrokeSimulator(server_url=args.server, session_id=session_id)

    try:
        if args.clear:
            await sim.clear()

        await sim.connect()
        await sim.send_strokes(strokes, page=args.page)

        if args.verify:
            # Small delay to let server process
            await asyncio.sleep(1.0)
            await sim.verify(expected_lines=args.expected_lines, timeout=args.timeout)
    finally:
        await sim.close()


if __name__ == "__main__":
    asyncio.run(main())
