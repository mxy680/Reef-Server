"""Render stroke dicts to a PNG image using Pillow."""

from io import BytesIO

from PIL import Image, ImageDraw


def render_strokes(strokes: list[dict]) -> bytes:
    """Render strokes as black lines on white background, return PNG bytes.

    Each stroke dict must have a 'points' list with 'x' and 'y' keys.
    """
    # Collect all points to find bounding box
    all_xs: list[float] = []
    all_ys: list[float] = []
    for stroke in strokes:
        for pt in stroke.get("points", []):
            all_xs.append(pt["x"])
            all_ys.append(pt["y"])

    if not all_xs:
        # Return a blank image if no points
        img = Image.new("RGB", (512, 128), "white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    data_w = max_x - min_x or 1
    data_h = max_y - min_y or 1

    # Image sizing: 512px wide, aspect-ratio height, min 128px
    padding = 20
    img_w = 512
    img_h = max(128, int(data_h / data_w * (img_w - padding * 2)) + padding * 2)

    draw_w = img_w - padding * 2
    draw_h = img_h - padding * 2
    scale = min(draw_w / data_w, draw_h / data_h)
    offset_x = padding + (draw_w - data_w * scale) / 2
    offset_y = padding + (draw_h - data_h * scale) / 2

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        points = stroke.get("points", [])
        if len(points) < 2:
            continue
        coords = [
            (
                (pt["x"] - min_x) * scale + offset_x,
                (pt["y"] - min_y) * scale + offset_y,
            )
            for pt in points
        ]
        draw.line(coords, fill="black", width=2)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
