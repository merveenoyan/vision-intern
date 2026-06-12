"""Shared bounding-box drawing for detection / judged datasets.

A single :func:`draw_detections` used by the standalone ``visualize_*`` scripts
**and** by :mod:`tools.hub_viz` (which overlays boxes on a sample of images
every time a dataset is pushed to the Hub).
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# Per-class colours for the DocVQA media classes; anything else falls back.
COLORS = {
    "table": "#FF6B6B",
    "image": "#4ECDC4",
    "chart": "#45B7D1",
    "diagram": "#96CEB4",
    "figure": "#FFEAA7",
}
FALLBACK_COLOR = "#DDA0DD"


def _font(size: int = 16):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _score_lookup(verdicts: list[dict] | None) -> dict[int, dict]:
    """Map detection index → verdict dict, tolerating both the legacy
    single-judge schema (``detection_idx`` + ``score``) and the ensemble
    schema (``detection_idx`` + ``mean_score``)."""
    out: dict[int, dict] = {}
    for v in verdicts or []:
        i = v.get("detection_idx", v.get("id"))
        if i is not None:
            out[int(i)] = v
    return out


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    verdicts: list[dict] | None = None,
) -> Image.Image:
    """Return a copy of *image* with each detection's box + label drawn.

    When *verdicts* is given, the judge score (``mean_score`` for the
    ensemble schema, else ``score``) is appended to the label.
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _font(16)

    score_by_idx = _score_lookup(verdicts)

    for i, det in enumerate(detections):
        bbox = det.get("bbox", det.get("box", []))
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        label = det.get("label", "?")
        color = COLORS.get(label, FALLBACK_COLOR)

        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        v = score_by_idx.get(i)
        if v is None:
            text = str(label)
        else:
            score = v.get("mean_score", v.get("score", 0.0))
            text = f"{label} {score:.2f}"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 4), text, fill="white", font=font)

    return img
