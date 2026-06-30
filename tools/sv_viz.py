"""Bounding-box visualization backed by Roboflow ``supervision``.

:func:`annotate` is a drop-in alternative to :func:`tools.bbox_viz.draw_detections`
that renders boxes and labels with supervision's annotators — giving consistent
per-class / per-track colour palettes, and the same label conventions
(``#index`` prefixes, judge verdict scores) the PIL path already uses.

The light core keeps the pure-PIL :mod:`tools.bbox_viz` so dataset pushes work
without extra deps; this module is the opt-in, supervision-powered path and is
also what :mod:`tools.track_video` uses to draw each frame. ``supervision`` is
imported lazily (see :func:`tools.sv_convert.to_supervision`).

CLI
---
::

    python -m tools.sv_viz image.jpg --detections dets.json --out annotated.png
    python -m tools.sv_viz image.jpg -d dets.json --verdicts verdicts.json --show-conf
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from .sv_convert import _require_sv, to_supervision


def _color_lookup(sv: Any, color_by: str, dets: Any):
    """Resolve a ``color_by`` string to an ``sv.ColorLookup``.

    ``"track"`` falls back to ``CLASS`` when the detections carry no tracker_id.
    """
    cb = (color_by or "class").lower()
    if cb == "track" and getattr(dets, "tracker_id", None) is not None:
        return sv.ColorLookup.TRACK
    if cb == "index":
        return sv.ColorLookup.INDEX
    return sv.ColorLookup.CLASS


def _labels(dets: Any, show_index: bool, show_conf: bool) -> list[str]:
    """Build per-box label text, mirroring the PIL path's conventions.

    Prefix is the tracker id (``#7``) when tracking, else the detection index
    when *show_index*. A judge ``verdict_score`` is appended when present; the
    raw confidence is appended instead only when *show_conf*.
    """
    import numpy as np

    names = dets.data.get("class_name") if getattr(dets, "data", None) else None
    verd = dets.data.get("verdict_score") if getattr(dets, "data", None) else None
    out: list[str] = []
    for i in range(len(dets.xyxy)):
        if names is not None:
            name = str(names[i])
        elif dets.class_id is not None:
            name = str(int(dets.class_id[i]))
        else:
            name = "?"
        if dets.tracker_id is not None:
            prefix = f"#{int(dets.tracker_id[i])} "
        elif show_index:
            prefix = f"#{i} "
        else:
            prefix = ""
        text = f"{prefix}{name}"
        if verd is not None and not np.isnan(verd[i]):
            text += f" {verd[i]:.2f}"
        elif show_conf and dets.confidence is not None and not np.isnan(dets.confidence[i]):
            text += f" {dets.confidence[i]:.2f}"
        out.append(text)
    return out


def annotate_array(
    scene: Any,
    dets: Any,
    color_by: str = "class",
    show_index: bool = False,
    show_conf: bool = False,
    thickness: int = 2,
) -> Any:
    """Draw *dets* onto a numpy image array (in place on a copy is the caller's job).

    Operates directly on the ndarray supervision passes around (RGB for still
    images, BGR for video frames — annotators are channel-agnostic), so it is
    reused by both :func:`annotate` and :mod:`tools.track_video`.
    """
    sv = _require_sv()
    lookup = _color_lookup(sv, color_by, dets)
    # Instance masks (RF-DETR-Seg / Falcon) go down first so boxes + labels
    # stay crisp on top.
    if getattr(dets, "mask", None) is not None and len(dets):
        scene = sv.MaskAnnotator(color_lookup=lookup).annotate(scene=scene, detections=dets)
    box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=lookup)
    scene = box_annotator.annotate(scene=scene, detections=dets)
    if len(dets):
        label_annotator = sv.LabelAnnotator(
            color_lookup=lookup, text_scale=0.5, text_thickness=1, text_padding=4
        )
        scene = label_annotator.annotate(
            scene=scene, detections=dets, labels=_labels(dets, show_index, show_conf)
        )
    return scene


def annotate(
    image: str | Image.Image,
    detections: list[dict],
    verdicts: list[dict] | None = None,
    show_index: bool = False,
    show_conf: bool = False,
    color_by: str = "class",
    class_map: dict[str, int] | None = None,
    thickness: int = 2,
) -> Image.Image:
    """Return a copy of *image* with each detection's box + label drawn.

    Supervision-backed counterpart to :func:`tools.bbox_viz.draw_detections`.

    Parameters
    ----------
    image :
        PIL image, path or URL.
    detections :
        Detection-dicts (``label`` + ``box``/``bbox`` + optional ``score`` /
        ``sub_label`` / ``track_id``).
    verdicts :
        Optional judge verdicts; the score is appended to each box's label.
    show_index :
        Prefix each label with its detection index (``#0``, ``#1`` …) so a VLM
        judge can reference boxes by number. Ignored when the detections carry
        track ids (the tracker id is shown instead).
    show_conf :
        Append the detection confidence to labels that have no verdict score.
    color_by :
        ``"class"`` (default), ``"track"`` or ``"index"`` colour lookup.
    class_map :
        Optional ``{label: class_id}`` reused across calls for stable colours.
    thickness :
        Box outline thickness in pixels.
    """
    import numpy as np

    from .utils import load_image

    img = load_image(image).convert("RGB")
    dets = to_supervision(detections, verdicts=verdicts, class_map=class_map)
    scene = np.asarray(img).copy()
    scene = annotate_array(
        scene, dets, color_by=color_by, show_index=show_index,
        show_conf=show_conf, thickness=thickness,
    )
    return Image.fromarray(scene)


def _main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Annotate an image with supervision bbox annotators.")
    parser.add_argument("image", help="Image path or URL")
    parser.add_argument("-d", "--detections", required=True,
                        help="JSON file: list of {label, box/bbox, score?, track_id?}")
    parser.add_argument("--verdicts", help="JSON file: list of {detection_idx, mean_score/score}")
    parser.add_argument("-o", "--out", default="annotated.png", help="Output image path")
    parser.add_argument("--color-by", default="class", choices=["class", "track", "index"])
    parser.add_argument("--show-index", action="store_true")
    parser.add_argument("--show-conf", action="store_true")
    args = parser.parse_args(argv)

    with open(args.detections) as f:
        detections = json.load(f)
    verdicts = None
    if args.verdicts:
        with open(args.verdicts) as f:
            verdicts = json.load(f)

    out = annotate(
        args.image, detections, verdicts=verdicts,
        show_index=args.show_index, show_conf=args.show_conf, color_by=args.color_by,
    )
    out.save(args.out)
    print(f"Wrote {args.out} ({len(detections)} detections)")


if __name__ == "__main__":
    _main()
