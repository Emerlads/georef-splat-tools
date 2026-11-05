"""Independent vertical check of an ENU reconstruction against USGS 3DEP terrain.

For each ground candidate (ENU x, y, z from the SfM point cloud):
  1. convert to lat/lon and ellipsoidal height with the manifest origin;
  2. ask USGS EPQS for the 3DEP bare-earth elevation (NAVD88, orthometric);
  3. ask NOAA NGS for the GEOID18 undulation N, so h_ref = H_3DEP + N;
  4. report reconstruction minus reference, both as captured and under the
     common mistake of treating the GPS altitude as orthometric (MSL).

Usage: python vertical_check.py candidates.json georeference_manifest.json out.json
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

import numpy as np

from georef_splat.geodesy import enu_to_geodetic
from georef_splat.sim3 import residual_stats


def get_json(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def main(cand_path: str, manifest_path: str, out_path: str) -> None:
    cands = json.load(open(cand_path))
    m = json.load(open(manifest_path))
    o = m["origin"]
    origin = (o["lat"], o["lon"], o["height_ellipsoidal"])
    n_geoid = get_json(f"https://geodesy.noaa.gov/api/geoid/ght?lat={origin[0]:.6f}&lon={origin[1]:.6f}&model=14")["geoidHeight"]

    rows = []
    for x, y, z in cands:
        lat, lon, h = enu_to_geodetic([x, y, z], origin)
        dem = float(get_json(
            f"https://epqs.nationalmap.gov/v1/json?x={lon:.7f}&y={lat:.7f}&wkid=4326&units=Meters&includeDate=false"
        )["value"])
        if dem < -1000:  # EPQS no-data
            continue
        h_ref = dem + n_geoid
        rows.append({"lat": lat, "lon": lon, "enu": [x, y, z], "h_recon": h, "H_3dep": dem, "h_ref": h_ref, "dz": h - h_ref})

    dz = np.array([r["dz"] for r in rows])
    # Robust: drop gross outliers (cells that still caught a roof or canopy).
    med = np.median(dz)
    mad = np.median(np.abs(dz - med)) * 1.4826
    keep = np.abs(dz - med) < max(3 * mad, 0.5)
    # Anchoring to 3DEP removes the bias; score it honestly on points not used to fit it.
    kept = dz[keep]
    rng = np.random.default_rng(0)
    holdout = []
    for _ in range(200):
        idx = rng.permutation(len(kept))
        fit, test = idx[: len(kept) // 2], idx[len(kept) // 2 :]
        holdout.append(float(np.sqrt(np.mean((kept[test] - kept[fit].mean()) ** 2))))
    result = {
        "method": "SfM ground points vs USGS 3DEP bare-earth (EPQS) + NOAA GEOID18",
        "anchored_holdout_rmse_m": {"median": round(float(np.median(holdout)), 3),
                                    "p90": round(float(np.percentile(holdout, 90)), 3),
                                    "splits": 200, "note": "bias fitted on a random half, RMSE on the other half"},
        "geoid_undulation_m": n_geoid,
        "checkpoints": int(keep.sum()),
        "rejected_outliers": int((~keep).sum()),
        "as_captured_ellipsoidal": {"mean_dz_m": float(dz[keep].mean()), **residual_stats(dz[keep])},
        # Wrongly treating an ellipsoidal altitude as MSL and "correcting" it: h = alt + N.
        "if_gps_altitude_treated_as_msl": {"mean_dz_m": float(dz[keep].mean() + n_geoid), **residual_stats(dz[keep] + n_geoid)},
        "points": [dict(r, kept=bool(k)) for r, k in zip(rows, keep)],
        "note": "3DEP is independent of the drone GPS; ground picks come from the sparse SfM cloud, so cells with "
                "vegetation can bias high. Horizontal accuracy is not assessed here.",
    }
    json.dump(result, open(out_path, "w"), indent=1)
    a, b = result["as_captured_ellipsoidal"], result["if_gps_altitude_treated_as_msl"]
    print(f"N = {n_geoid:.3f} m; {result['checkpoints']} checkpoints ({result['rejected_outliers']} rejected)")
    print(f"as captured (ellipsoidal): mean dz {a['mean_dz_m']:+.2f} m, RMSE {a['rmse_m']:.2f} m")
    print(f"treated as MSL (wrong):    mean dz {b['mean_dz_m']:+.2f} m, RMSE {b['rmse_m']:.2f} m")
    print(f"3DEP-anchored, held out:   RMSE {result['anchored_holdout_rmse_m']['median']:.2f} m (median of 200 splits)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
