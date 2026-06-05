"""Visualize bounding box detections on dataset images."""

import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "table": "#FF6B6B",
    "image": "#4ECDC4",
    "chart": "#45B7D1",
    "diagram": "#96CEB4",
    "figure": "#FFEAA7",
}
FALLBACK_COLOR = "#DDA0DD"

DATASET_ID = "merve/docvqa-media-labeled"
SPLIT = "test"
NUM_EXAMPLES = 8
OUTPUT_DIR = Path("viz_output")


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det["bbox"]
        label = det.get("label", "?")
        color = COLORS.get(label, FALLBACK_COLOR)

        x1, y1, x2, y2 = bbox
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        text = label
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 4), text, fill="white", font=font)

    return img


def main() -> None:
    ds = load_dataset(DATASET_ID, split=SPLIT)

    has_dets = [i for i in range(len(ds)) if ds[i]["detections"]]
    print(f"{len(has_dets)}/{len(ds)} rows have detections")

    indices = random.sample(has_dets, min(NUM_EXAMPLES, len(has_dets)))
    indices.sort()

    OUTPUT_DIR.mkdir(exist_ok=True)

    for idx in indices:
        row = ds[idx]
        img = row["image"]
        dets = row["detections"]

        viz = draw_detections(img, dets)

        out_path = OUTPUT_DIR / f"example_{idx:04d}.png"
        viz.save(out_path)

        labels = [d["label"] for d in dets]
        print(f"  [{idx}] {len(dets)} detections {labels} → {out_path}")

    print(f"\nSaved {len(indices)} images to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
