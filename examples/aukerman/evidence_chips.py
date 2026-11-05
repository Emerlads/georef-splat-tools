"""Evidence chips: for each flagged structure, crop the source drone photo that sees it best.

A screening tool should hand the reviewer the photo, not just a polygon. For
each structure, project its footprint into every registered image (ENU-aligned
COLMAP model), score views by how centred and how close to nadir it is, and
crop the best one around the footprint.

Usage: python evidence_chips.py MODEL_DIR IMAGES_DIR zones_summary.json zones.geojson OUT_DIR
(needs pycolmap)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image, ImageDraw


def main(model_dir, images_dir, summary_path, geojson_path, out_dir) -> None:
    rec = pycolmap.Reconstruction(model_dir)
    summary = json.load(open(summary_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ENU footprints are written next to the summary by `georef-splat zones`.
    enu = json.load(open(Path(summary_path).with_name("footprints_enu.json")))
    for st in summary["structures"]:
        hull = np.array(enu[str(st["id"])])
        z = st["roof_z_enu"] if st["roof_z_enu"] is not None else st["ground_z_enu"]
        pts3 = np.c_[hull, np.full(len(hull), z)]
        centre = pts3.mean(0)
        best = None
        for img in rec.images.values():
            cam = rec.cameras[img.camera_id]
            cam_pts = np.array([img.cam_from_world() * p for p in pts3])
            if (cam_pts[:, 2] <= 0).any():
                continue
            uv = np.array([np.asarray(cam.img_from_cam(p)).ravel()[:2] for p in cam_pts])
            if (uv < 0).any() or (uv[:, 0] >= cam.width).any() or (uv[:, 1] >= cam.height).any():
                continue
            off = np.linalg.norm(uv.mean(0) - [cam.width / 2, cam.height / 2]) / cam.width
            ray = centre - img.projection_center()
            nadir = abs(ray[2]) / np.linalg.norm(ray)
            score = nadir - off
            if best is None or score > best[0]:
                best = (score, img, uv)
        if best is None:
            print(f"structure {st['id']}: no image sees the whole footprint")
            continue
        _, img, uv = best
        im = Image.open(Path(images_dir) / img.name).convert("RGB")
        lo, hi = uv.min(0), uv.max(0)
        pad = max(hi - lo) * 0.9 + 80
        cx, cy = (lo + hi) / 2
        box = (int(cx - pad), int(cy - pad), int(cx + pad), int(cy + pad))
        d = ImageDraw.Draw(im)
        d.line([tuple(p) for p in np.vstack([uv, uv[:1]])], fill=(255, 80, 60), width=6)
        chip = im.crop(box).resize((480, 480), Image.LANCZOS)
        chip.save(out / f"structure_{st['id']}.jpg", quality=86)
        print(f"structure {st['id']}: {img.name}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
