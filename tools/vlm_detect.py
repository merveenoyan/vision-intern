"""Instruction-prompted object detection using Vision Language Models.

Adapted from `uv-scripts/vlm-object-detection
<https://huggingface.co/datasets/uv-scripts/vlm-object-detection>`_.
Sends a free-form detection prompt to a VLM and parses ``bbox_2d`` JSON
from the response.  Coordinates are denormalised from the Qwen-VL 0-1000
scale to original-image pixel coordinates.

Supports two backends (see :mod:`tools.vlm_client`):

- ``"openai"``  — HF Inference Providers, vLLM server, llama-server
- ``"transformers"`` — local GPU with ``transformers``

CLI usage::

    python -m tools.vlm_detect image.jpg --classes "cat,dog"

    python -m tools.vlm_detect image.jpg --classes "cat,dog" \\
        --backend openai --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from PIL import Image

from .utils import load_image
from .vlm_client import run_vlm

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

DEFAULT_PROMPT = (
    "Detect every distinct object in the image. For each object, output a JSON "
    'object with keys: "bbox_2d" (an array [x1, y1, x2, y2] normalised to '
    '0-1000), "label" (category), and "sub_label" (short attribute or ""). '
    "Return a JSON array."
)


# ------------------------------------------------------------------
# Parsing  (battle-tested logic from qwen3vl-detect.py)
# ------------------------------------------------------------------

def parse_bboxes_from_response(text: str) -> list[dict[str, Any]]:
    """Extract bbox objects from VLM text.

    Tolerates fenced code blocks, trailing commas, comments, and
    malformed JSON.  Falls back to regex extraction when no parseable
    objects are found.
    """
    results: list[dict[str, Any]] = []

    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[\d\s.,\-]+\][^{}]*\}'
    for match in re.findall(pattern, text, re.DOTALL):
        try:
            obj = json.loads(match)
            if "bbox_2d" in obj:
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        try:
            cleaned = re.sub(r"#.*$", "", match, flags=re.MULTILINE).strip()
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*\]", "]", cleaned)
            obj = json.loads(cleaned)
            if "bbox_2d" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue

    if not results:
        bbox_pat = (
            r'"bbox_2d"\s*:\s*\[\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,'
            r"\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*\]"
        )
        for i, (x1, y1, x2, y2) in enumerate(re.findall(bbox_pat, text)):
            results.append({
                "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
                "label": f"object_{i + 1}",
                "sub_label": "",
            })

    if not results:
        generic = r'\{[^{}]*"(?:box|bbox)"\s*:\s*\[[\d\s.,\-]+\][^{}]*\}'
        for match in re.findall(generic, text, re.DOTALL):
            try:
                obj = json.loads(match)
                key = "box" if "box" in obj else "bbox"
                obj["bbox_2d"] = obj.pop(key)
                if "label" not in obj:
                    obj["label"] = "object"
                results.append(obj)
            except (json.JSONDecodeError, KeyError):
                continue

    return results


def denormalize_bbox(
    bbox: list[float], width: int, height: int,
) -> list[int]:
    """Convert a 0-1000 normalised bbox to original-image pixel coords."""
    if len(bbox) != 4:
        return []
    sx, sy = width / 1000.0, height / 1000.0
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width, round(x1 * sx))),
        max(0, min(height, round(y1 * sy))),
        max(0, min(width, round(x2 * sx))),
        max(0, min(height, round(y2 * sy))),
    ]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def vlm_detect(
    image: str | Image.Image,
    prompt: str | None = None,
    classes: list[str] | None = None,
    model_id: str = DEFAULT_MODEL,
    backend: str = "transformers",
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """Detect objects in *image* using a VLM with a free-form prompt.

    Parameters
    ----------
    image : str or PIL.Image
        Path, URL, or PIL Image.
    prompt : str, optional
        Free-form detection instruction.  When *classes* is given the
        prompt is built automatically.
    classes : list[str], optional
        Object categories to look for.
    model_id : str
        Model identifier (HF repo id or server model name).
    backend : ``"openai"`` | ``"transformers"``
        Inference backend.  ``"openai"`` works with HF Inference
        Providers, vLLM, and llama-server.
    base_url : str, optional
        API endpoint for the ``openai`` backend.
    api_key : str, optional
        API key / HF token for the ``openai`` backend.

    Returns
    -------
    list[dict]
        Each dict contains ``bbox`` ([x1,y1,x2,y2] in pixels),
        ``label`` (str), and ``sub_label`` (str).
    """
    image = load_image(image)
    w, h = image.size

    if prompt is None:
        if classes:
            class_list = ", ".join(classes)
            prompt = (
                f"Detect every instance of these categories: {class_list}. "
                "For each, return a JSON object with "
                '"bbox_2d" ([x1, y1, x2, y2] normalised 0-1000), '
                '"label" (one of the listed categories), '
                'and "sub_label" (short attribute or ""). '
                "Return a JSON array."
            )
        else:
            prompt = DEFAULT_PROMPT

    response = run_vlm(
        image, prompt, model_id,
        backend=backend, base_url=base_url, api_key=api_key,
        max_tokens=4096,
    )

    parsed = parse_bboxes_from_response(response)

    detections = []
    for obj in parsed:
        raw = obj.get("bbox_2d", [])
        raw = [float(x) for x in raw] if isinstance(raw, list) and len(raw) == 4 else []
        bbox = denormalize_bbox(raw, w, h)
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        detections.append({
            "bbox": bbox,
            "label": str(obj.get("label", "")),
            "sub_label": str(obj.get("sub_label") or ""),
        })
    return detections


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM instruction-prompted object detection",
    )
    parser.add_argument("image", help="Image path or URL")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--classes", default=None, help="Comma-separated classes")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="transformers",
                        choices=["openai", "transformers"])
    parser.add_argument("--base-url", default=None,
                        help="API base URL (openai backend)")
    parser.add_argument("--api-key", default=None,
                        help="API key / HF token (openai backend)")
    args = parser.parse_args()

    cls = [c.strip() for c in args.classes.split(",")] if args.classes else None
    dets = vlm_detect(
        args.image, prompt=args.prompt, classes=cls, model_id=args.model,
        backend=args.backend, base_url=args.base_url, api_key=args.api_key,
    )
    print(json.dumps(dets, indent=2))
