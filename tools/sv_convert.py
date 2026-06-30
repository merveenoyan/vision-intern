"""Bridge between this repo's detection-dicts and ``supervision.Detections``.

Detections in vision-agent are plain dicts::

    {"label": str, "score": float, "box"/"bbox": [x1, y1, x2, y2]}

with optional ``"sub_label"`` and, for tracked video frames, ``"track_id"``.
Judge verdicts are a parallel list of ``{"detection_idx", "mean_score"/"score"}``.

Roboflow's `supervision`_ annotators and `trackers`_ operate on
``supervision.Detections`` (an ``xyxy`` array plus ``class_id`` /
``confidence`` / ``tracker_id`` and a ``data`` dict).  These helpers convert
between the two formats.  ``supervision`` is **not** part of the light core,
so the import is deferred to call time and raises a friendly install hint if
it is missing — listing tools or importing other helpers stays torch- and
supervision-free.

.. _supervision: https://supervision.roboflow.com
.. _trackers: https://github.com/roboflow/trackers
"""

from __future__ import annotations

from typing import Any


def _require_sv():
    """Import ``supervision`` or raise with an install hint."""
    try:
        import supervision as sv
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "supervision is required for bbox visualization. Install it with "
            "`pip install 'vision-agent[viz]'` (or `pip install supervision`)."
        ) from e
    return sv


def to_supervision(
    detections: list[dict],
    verdicts: list[dict] | None = None,
    class_map: dict[str, int] | None = None,
) -> Any:
    """Convert a list of detection-dicts to a ``supervision.Detections``.

    Parameters
    ----------
    detections :
        List of dicts with ``label``, ``box``/``bbox`` ``[x1, y1, x2, y2]`` and
        optionally ``score``, ``sub_label``, ``track_id`` and ``mask`` (a boolean
        ``(H, W)`` instance mask, e.g. from RF-DETR-Seg / Falcon-Perception).
    verdicts :
        Optional judge verdicts (``detection_idx`` + ``mean_score``/``score``),
        indexed against the *original* ``detections`` order. Surfaced as a
        ``verdict_score`` entry in ``Detections.data``.
    class_map :
        Optional ``{label: class_id}`` mapping. New labels are appended in
        place, so passing the same dict across video frames keeps each class's
        colour stable. When ``None`` a fresh mapping is built per call.

    Returns
    -------
    supervision.Detections
        Boxes whose ``box``/``bbox`` is malformed are dropped. ``class_name``
        (and ``sub_label`` / ``verdict_score`` when present) live in ``.data``.
    """
    sv = _require_sv()
    import numpy as np

    if class_map is None:
        class_map = {}

    xyxy: list[list[float]] = []
    labels: list[str] = []
    sub_labels: list[str] = []
    scores: list[float | None] = []
    track_ids: list[int | None] = []
    masks: list[Any] = []
    orig_idx: list[int] = []

    for i, det in enumerate(detections or []):
        bbox = det.get("box", det.get("bbox"))
        if not bbox or len(bbox) != 4:
            continue
        orig_idx.append(i)
        xyxy.append([float(c) for c in bbox])
        labels.append(str(det.get("label", "object")))
        sub_labels.append(str(det.get("sub_label", "")))
        scores.append(det.get("score"))
        track_ids.append(det.get("track_id", det.get("tracker_id")))
        masks.append(det.get("mask"))

    if not xyxy:
        return sv.Detections.empty()

    for lab in labels:
        if lab not in class_map:
            class_map[lab] = len(class_map)
    class_id = np.array([class_map[lab] for lab in labels], dtype=int)

    confidence = None
    if any(s is not None for s in scores):
        confidence = np.array(
            [float(s) if s is not None else float("nan") for s in scores], dtype=float
        )

    tracker_id = None
    if any(t is not None for t in track_ids):
        tracker_id = np.array(
            [int(t) if t is not None else -1 for t in track_ids], dtype=int
        )

    # Instance masks are carried only when *every* kept box has one of the same
    # (H, W) shape, since supervision stacks them into a single (N, H, W) array.
    mask = None
    if all(m is not None for m in masks):
        arrs = [np.asarray(m).astype(bool) for m in masks]
        if len({a.shape for a in arrs}) == 1:
            mask = np.stack(arrs)

    data: dict[str, Any] = {"class_name": np.array(labels, dtype=object)}
    if any(sub_labels):
        data["sub_label"] = np.array(sub_labels, dtype=object)
    if verdicts:
        vmap: dict[int, float] = {}
        for v in verdicts:
            idx = v.get("detection_idx", v.get("id"))
            if idx is not None:
                vmap[int(idx)] = v.get("mean_score", v.get("score"))
        per_box = [vmap.get(oi) for oi in orig_idx]
        if any(x is not None for x in per_box):
            data["verdict_score"] = np.array(
                [float(x) if x is not None else float("nan") for x in per_box],
                dtype=float,
            )

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
        data=data,
    )


def from_supervision(dets: Any) -> list[dict]:
    """Convert a ``supervision.Detections`` back to detection-dicts.

    Inverse of :func:`to_supervision`: each box becomes a dict with ``box`` and
    ``label`` (from ``data["class_name"]`` when available, else the numeric
    ``class_id``), plus ``score`` / ``track_id`` / ``sub_label`` when present.
    """
    out: list[dict] = []
    names = dets.data.get("class_name") if getattr(dets, "data", None) else None
    subs = dets.data.get("sub_label") if getattr(dets, "data", None) else None
    for i in range(len(dets.xyxy)):
        x1, y1, x2, y2 = (float(c) for c in dets.xyxy[i])
        det: dict[str, Any] = {"box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}
        if names is not None:
            det["label"] = str(names[i])
        elif dets.class_id is not None:
            det["label"] = str(int(dets.class_id[i]))
        else:
            det["label"] = "object"
        if dets.confidence is not None:
            det["score"] = round(float(dets.confidence[i]), 4)
        if dets.tracker_id is not None and int(dets.tracker_id[i]) >= 0:
            det["track_id"] = int(dets.tracker_id[i])
        if subs is not None and subs[i]:
            det["sub_label"] = str(subs[i])
        if getattr(dets, "mask", None) is not None:
            det["mask"] = dets.mask[i]
        out.append(det)
    return out
