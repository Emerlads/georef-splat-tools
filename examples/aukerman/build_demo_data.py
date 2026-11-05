"""Assemble the small JSON files the Splat to Earth viewer reads.

Usage: python build_demo_data.py georeference_manifest.json vertical_check.json OUT_DIR
(Run ``georef-splat tiles ... --out OUT_DIR/tiles`` separately for the splat.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(manifest_path: str, vcheck_path: str, out_dir: str) -> None:
    m = json.load(open(manifest_path))
    v = json.load(open(vcheck_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kept = [p for p in v["points"] if p["kept"]]
    fit = m["residuals_inliers"]
    meta = {
        "dataset": "OpenDroneMap odm_data_aukerman (CC0)",
        "origin": {"lat": m["origin"]["lat"], "lon": m["origin"]["lon"], "h": m["origin"]["height_ellipsoidal"]},
        "geoid_undulation_m": v["geoid_undulation_m"],
        "ground_enu_z": float(np.median([p["enu"][2] for p in kept])),
        "fit": {
            "cameras": m["inliers"],
            "horizontal_rmse_m": fit["horizontal"]["rmse_m"],
            "vertical_rmse_m": fit["vertical"]["rmse_m"],
            "note": m["note"],
        },
        "vertical": {
            "n": v["checkpoints"],
            "bias_m": v["as_captured_ellipsoidal"]["mean_dz_m"],
            "rmse_m": v["as_captured_ellipsoidal"]["rmse_m"],
            "method": v["method"],
        },
    }
    cams = [{"image": c["image"], "enu": c["enu_gps"], "residual_m": c["residual_m"]} for c in m["cameras"]]
    chk = [{"lat": p["lat"], "lon": p["lon"], "h_ref": p["h_ref"], "dz": p["dz"]} for p in kept]
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "cameras.json").write_text(json.dumps(cams))
    (out / "checkpoints.json").write_text(json.dumps(chk))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
