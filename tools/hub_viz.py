"""Push a detection/judged dataset to the Hub **and** publish a box-overlay
gallery alongside it.

``push_dataset_with_viz`` replaces a bare ``ds.push_to_hub(...)``: it pushes the
dataset, renders boxes (and judge scores, if present) onto a sample of images,
uploads them to a ``viz/`` folder in the same dataset repo, and embeds them in
the dataset README so they render on the dataset page — no need to ever run a
visualization script by hand.
"""

from __future__ import annotations

import io
import random
from typing import Any

from PIL import Image

from tools.bbox_viz import draw_detections

_GALLERY_START = "<!-- box-overlay-gallery:start -->"
_GALLERY_END = "<!-- box-overlay-gallery:end -->"


def _as_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value
    try:
        from tools.utils import load_image
        return load_image(value)
    except Exception:
        return None


def _build_gallery_md(repo_id: str, filenames: list[str], revision: str) -> str:
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}"
    imgs = "\n".join(
        f'<img src="{base}/viz/{fn}" width="320" style="margin:4px"/>'
        for fn in filenames
    )
    return (
        f"{_GALLERY_START}\n"
        "## Box-overlay preview\n\n"
        "Auto-generated sample of the labelled boxes (with judge scores when "
        "available). Regenerated on every push.\n\n"
        f"{imgs}\n"
        f"{_GALLERY_END}"
    )


def _upsert_gallery(readme: str, gallery_md: str) -> str:
    if _GALLERY_START in readme and _GALLERY_END in readme:
        pre = readme.split(_GALLERY_START)[0].rstrip()
        post = readme.split(_GALLERY_END, 1)[1].lstrip()
        parts = [p for p in (pre, gallery_md, post) if p]
        return "\n\n".join(parts) + "\n"
    return readme.rstrip() + "\n\n" + gallery_md + "\n"


def push_dataset_with_viz(
    ds: Any,
    repo_id: str,
    *,
    token: str | None,
    image_column: str = "image",
    detections_column: str = "detections",
    verdicts_column: str = "judge_verdicts",
    num_samples: int = 12,
    seed: int = 1337,
    revision: str = "main",
) -> None:
    """Push *ds* to *repo_id* and publish a box-overlay gallery to its README.

    Falls back gracefully: a failure to render/upload the gallery never blocks
    the dataset push itself.
    """
    # Clear any stale README first: pushing over an existing repo keeps the old
    # ``dataset_info`` features YAML, so a schema change (new struct fields) ends
    # up with parquet that no longer matches the declared features and
    # ``load_dataset`` fails to cast. Deleting it forces push_to_hub to write
    # fresh, matching metadata.
    try:
        from huggingface_hub import HfApi as _HfApi
        _HfApi(token=token).delete_file(
            "README.md", repo_id, repo_type="dataset",
            commit_message="Clear stale card before refresh",
        )
    except Exception:
        pass  # no existing README (new repo) — nothing to clear

    ds.push_to_hub(repo_id, token=token)
    print(f"Pushed dataset → https://huggingface.co/datasets/{repo_id}")

    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=token)

        cols = ds.column_names
        has_dets = (
            [i for i in range(len(ds)) if ds[i].get(detections_column)]
            if detections_column in cols else []
        )
        if not has_dets:
            print("  [viz] no detections to render — skipping gallery")
            return

        random.seed(seed)
        idxs = sorted(random.sample(has_dets, min(num_samples, len(has_dets))))

        filenames: list[str] = []
        for idx in idxs:
            row = ds[idx]
            img = _as_image(row[image_column])
            if img is None:
                continue
            dets = row[detections_column]
            verdicts = row.get(verdicts_column) if verdicts_column in cols else None
            viz = draw_detections(img, dets, verdicts)

            buf = io.BytesIO()
            viz.save(buf, format="PNG")
            fn = f"sample_{idx:05d}.png"
            api.upload_file(
                path_or_fileobj=buf.getvalue(),
                path_in_repo=f"viz/{fn}",
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            )
            filenames.append(fn)

        if not filenames:
            print("  [viz] nothing rendered — skipping gallery")
            return

        try:
            readme_path = hf_hub_download(
                repo_id, "README.md", repo_type="dataset",
                revision=revision, token=token,
            )
            with open(readme_path, encoding="utf-8") as f:
                readme = f.read()
        except Exception:
            readme = f"# {repo_id}\n"

        readme = _upsert_gallery(readme, _build_gallery_md(repo_id, filenames, revision))
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
        print(f"  [viz] published {len(filenames)} overlays → viz/ + README gallery")
    except Exception as e:  # noqa: BLE001 — viz is best-effort
        print(f"  [viz] gallery skipped ({type(e).__name__}: {e})")
