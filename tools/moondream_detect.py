"""Object detection with moondream3 via its native ``.detect()`` skill.

moondream does **not** use the prompt-based ``bbox_2d`` 0-1000 convention that
:mod:`tools.vlm_detect` (Qwen-family) relies on.  Instead it exposes a
``model.detect(image, "<class>")`` call that returns normalized 0-1 boxes::

    result = model.detect(image, "table")
    result["objects"]  # -> [{"x_min", "y_min", "x_max", "y_max"}, ...]

We call it once per requested class and denormalize to pixel coordinates so the
output matches what :func:`tools.vlm_detect.vlm_detect` produces
(``{"bbox": [x1, y1, x2, y2], "label", "sub_label"}``).

The model must be loaded with ``trust_remote_code=True`` (custom architecture).
Loading helper :func:`load_moondream` is provided for the labelling Job.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from .utils import load_image

DEFAULT_MODEL = "moondream/moondream3-preview"


def load_moondream(model_id: str = DEFAULT_MODEL, compile: bool = True) -> Any:
    """Load moondream3 onto CUDA (bf16). ``compile=True`` enables FlexAttention
    fast-path as recommended by the model card."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    model.eval()
    if compile:
        try:
            model.compile()
        except Exception as e:  # noqa: BLE001 — compile is best-effort
            print(f"  [moondream] compile skipped: {e}")
    return model


def moondream_detect(
    image: str | Image.Image,
    classes: list[str],
    model: Any,
    max_objects: int = 50,
) -> list[dict]:
    """Detect *classes* in *image* with a loaded moondream *model*.

    Returns one dict per detected object with pixel ``bbox`` [x1, y1, x2, y2],
    ``label`` (the queried class) and ``sub_label`` ("").
    """
    image = load_image(image)
    w, h = image.size
    settings = {"max_objects": max_objects}

    detections: list[dict] = []
    for cls in classes:
        try:
            result = model.detect(image, cls, settings=settings)
        except TypeError:
            # Older signature without settings kwarg.
            result = model.detect(image, cls)
        except Exception as e:  # noqa: BLE001 — one bad class shouldn't abort
            print(f"  [moondream] detect('{cls}') failed: {e}")
            continue

        for obj in result.get("objects", []):
            x1 = max(0, min(w, round(obj["x_min"] * w)))
            y1 = max(0, min(h, round(obj["y_min"] * h)))
            x2 = max(0, min(w, round(obj["x_max"] * w)))
            y2 = max(0, min(h, round(obj["y_max"] * h)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": cls,
                "sub_label": "",
            })
    return detections
