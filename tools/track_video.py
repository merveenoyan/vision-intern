"""Multi-object tracking visualization with Roboflow ``trackers`` + ``supervision``.

Runs a per-frame detector over a video, associates detections across frames
with a `trackers`_ algorithm (ByteTrack / SORT / OC-SORT), and writes an
annotated video where each object keeps a stable colour + id and an optional
motion trail.

The detection step reuses this repo's existing models frame by frame — the box
detectors (RF-DETR, MM-Grounding-DINO, VLM) and the instance-segmentation models
(RF-DETR-Seg, Falcon-Perception). The trackers associate on boxes (masks are
derived from each instance and preserved through tracking), so tracking inherits
whatever the labelling pipeline already produces. Masks, boxes and labels are
drawn by :func:`tools.sv_viz.annotate_array`.

Needs both the ``viz`` extra (supervision + trackers) and a detector from the
``train`` extra (torch). All heavy imports are deferred to call time.

.. _trackers: https://github.com/roboflow/trackers

CLI
---
::

    python -m tools.track_video clip.mp4 --detector rfdetr --tracker bytetrack
    python -m tools.track_video clip.mp4 --detector grounded --classes "car,person" --out tracked.mp4
    python -m tools.track_video clip.mp4 --detector rfdetr-seg          # tracked instance masks
    python -m tools.track_video clip.mp4 --detector falcon --classes "red car,person"
"""

from __future__ import annotations

import os
from typing import Any, Callable

# Friendly name → class name in the ``trackers`` package.
_TRACKERS = {
    "bytetrack": "ByteTrackTracker",
    "byte": "ByteTrackTracker",
    "sort": "SORTTracker",
    "ocsort": "OCSORTTracker",
    "oc-sort": "OCSORTTracker",
    "oc_sort": "OCSORTTracker",
    "botsort": "BoTSORTTracker",
    "bot-sort": "BoTSORTTracker",
    "bot_sort": "BoTSORTTracker",
}


def _resolve_tracker(tracker: str) -> Any:
    """Instantiate a tracker from the ``trackers`` package by friendly name."""
    try:
        import trackers
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The Roboflow `trackers` package is required for tracking. Install it "
            "with `pip install 'vision-agent[viz]'` (or `pip install trackers`)."
        ) from e

    cls_name = _TRACKERS.get(tracker.lower())
    cls = getattr(trackers, cls_name, None) if cls_name else None
    if cls is None:
        available = sorted(n for n in dir(trackers) if n.endswith("Tracker"))
        raise ValueError(
            f"Unknown tracker {tracker!r}. Known aliases: {sorted(_TRACKERS)}. "
            f"trackers package exports: {available}"
        )
    return cls()


def _mask_to_box(mask: Any) -> list[float] | None:
    """Tight ``[x1, y1, x2, y2]`` box around a boolean mask, or None if empty."""
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1]


def _seg_to_dets(result: dict) -> list[dict]:
    """RF-DETR-Seg ``{segmentation, segments_info}`` → detection-dicts with masks.

    Each instance becomes a dict carrying both a tight ``box`` (so the tracker,
    which associates on boxes, can follow it) and its boolean ``mask``.
    """
    import numpy as np

    seg = result["segmentation"]
    seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
    out: list[dict] = []
    for s in result.get("segments_info", []):
        mask = seg == s["id"]
        box = _mask_to_box(mask)
        if box is None:
            continue
        out.append({"label": s["label"], "score": s.get("score"), "box": box, "mask": mask})
    return out


def _falcon_to_dets(preds: list[dict], label: str, width: int, height: int) -> list[dict]:
    """Falcon-Perception preds (RLE masks + normalized center/size) → dicts.

    The box is derived from the decoded mask when possible, else from the
    normalized center/size; every instance is tagged with the query *label*.
    """
    from .utils import rle_to_mask

    out: list[dict] = []
    for p in preds:
        mask = rle_to_mask(p["mask_rle"]).astype(bool)
        box = _mask_to_box(mask)
        if box is None:
            cx, cy = p["center"]["x"] * width, p["center"]["y"] * height
            bw, bh = p["size"]["w"] * width, p["size"]["h"] * height
            box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
        out.append({"label": label, "score": None, "box": box, "mask": mask})
    return out


def _make_detector(
    detector: str, threshold: float, classes: list[str] | None
) -> Callable[[Any], list[dict]]:
    """Return ``pil_image -> list[detection-dict]`` for the chosen detector.

    Detectors emitting instance masks (RF-DETR-Seg, Falcon-Perception) include a
    boolean ``mask`` per detection; the box-based trackers still associate on the
    derived boxes and preserve the masks through ``update()``.
    """
    name = detector.lower()
    if name in ("rfdetr", "detect", "rf-detr"):
        from .detect import detect
        return lambda img: detect(img, threshold=threshold)
    if name in ("grounded", "grounded_detect", "dino"):
        if not classes:
            raise ValueError("detector='grounded' requires --classes / classes=[...].")
        from .grounded_detect import grounded_detect
        return lambda img: grounded_detect(img, classes, threshold=threshold)
    if name in ("vlm", "vlm_detect"):
        from .vlm_detect import vlm_detect
        return lambda img: vlm_detect(img, classes=classes)
    if name in ("rfdetr-seg", "rf-detr-seg", "instance_segment", "seg"):
        from .instance_segment import instance_segment
        return lambda img: _seg_to_dets(instance_segment(img, threshold=threshold))
    if name in ("falcon", "falcon-perception", "segment_from_text"):
        if not classes:
            raise ValueError("detector='falcon' requires --classes / classes=[...] (text queries).")
        from .segment_from_text import segment_from_text

        def detect_falcon(img):
            w, h = img.size
            out: list[dict] = []
            for cls in classes:
                out.extend(_falcon_to_dets(segment_from_text(img, cls), cls, w, h))
            return out

        return detect_falcon
    raise ValueError(
        f"Unknown detector {detector!r}. Use 'rfdetr', 'grounded', 'vlm', "
        "'rfdetr-seg' or 'falcon'."
    )


def track_video(
    video: str,
    output: str | None = None,
    detector: str = "rfdetr",
    tracker: str = "bytetrack",
    threshold: float = 0.3,
    classes: list[str] | None = None,
    trace: bool = True,
    color_by: str = "track",
    show_conf: bool = True,
    max_frames: int | None = None,
) -> str:
    """Detect, track and annotate every frame of *video*; write an annotated copy.

    Parameters
    ----------
    video :
        Path to the input video.
    output :
        Output path. Defaults to ``<input>_tracked.mp4`` beside the input.
    detector :
        Box detectors — ``"rfdetr"`` (COCO closed-set, default), ``"grounded"``
        (open-vocabulary, needs *classes*) or ``"vlm"`` (instruction-prompted).
        Instance-segmentation detectors (boxes + masks are tracked and drawn) —
        ``"rfdetr-seg"`` (RF-DETR-Seg, COCO closed-set) or ``"falcon"``
        (Falcon-Perception, open-vocabulary, needs *classes*).
    tracker :
        ``"bytetrack"`` (default), ``"sort"``, ``"ocsort"`` or ``"botsort"`` —
        a Roboflow ``trackers`` algorithm.
    threshold :
        Detection confidence threshold.
    classes :
        Class names / text queries for the ``grounded``, ``vlm`` and ``falcon``
        detectors.
    trace :
        Draw each object's motion trail (supervision ``TraceAnnotator``).
    color_by :
        ``"track"`` (one colour per object, default) or ``"class"``.
    show_conf :
        Append detection confidence to labels.
    max_frames :
        Process at most this many frames (handy for quick previews).

    Returns
    -------
    str
        The path of the written annotated video.
    """
    import supervision as sv
    from PIL import Image

    from .sv_convert import to_supervision
    from .sv_viz import annotate_array

    if output is None:
        stem, _ = os.path.splitext(video)
        output = f"{stem}_tracked.mp4"

    detect_fn = _make_detector(detector, threshold, classes)
    obj_tracker = _resolve_tracker(tracker)
    trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK) if trace else None
    class_map: dict[str, int] = {}  # stable label → colour index across frames

    def callback(frame, _index: int):
        pil = Image.fromarray(frame[..., ::-1])  # supervision frames are BGR
        detections = to_supervision(detect_fn(pil), class_map=class_map)
        detections = obj_tracker.update(detections)
        scene = frame.copy()
        if trace_annotator is not None and detections.tracker_id is not None:
            scene = trace_annotator.annotate(scene=scene, detections=detections)
        return annotate_array(scene, detections, color_by=color_by, show_conf=show_conf)

    sv.process_video(
        source_path=video,
        target_path=output,
        callback=callback,
        max_frames=max_frames,
        show_progress=True,
    )
    return output


def _main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Track objects through a video and write an annotated copy."
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("-o", "--out", dest="output", help="Output video path")
    parser.add_argument("--detector", default="rfdetr",
                        choices=["rfdetr", "grounded", "vlm", "rfdetr-seg", "falcon"])
    parser.add_argument("--tracker", default="bytetrack",
                        choices=["bytetrack", "sort", "ocsort", "botsort"])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--classes", help="Comma-separated class names (grounded / vlm detectors)")
    parser.add_argument("--no-trace", dest="trace", action="store_false")
    parser.add_argument("--color-by", default="track", choices=["track", "class"])
    parser.add_argument("--no-conf", dest="show_conf", action="store_false")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    out = track_video(
        args.video, output=args.output, detector=args.detector, tracker=args.tracker,
        threshold=args.threshold, classes=classes, trace=args.trace,
        color_by=args.color_by, show_conf=args.show_conf, max_frames=args.max_frames,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    _main()
