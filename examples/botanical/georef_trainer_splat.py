"""Georeference a splat made by a third-party trainer (Postshot / RealityScan).

The photoreal splats people actually ship usually come from commercial
trainers that export in their own local frame, often with swapped axes. This
script puts such a splat into local ENU metres, using only what those tools
export:

  * the trainer's dense cloud in its "grid plane" frame (PLY),
  * per-camera positions in that grid frame (CSV), and
  * the same cameras' GPS positions (CSV, lon/lat in radians, as RealityScan writes).

Steps:
  1. Splat frame -> grid frame: test the 24 axis rotations against the dense cloud,
     then refine with trimmed ICP (similarity).
  2. Grid frame -> ENU: RANSAC Sim3 from grid camera centres to their GPS in ENU.
  3. Compose, transform the splat (positions, log-scales, quaternions) and write
     a PLY plus a manifest compatible with `georef-splat tiles/audit/zones`.

Usage:
  python georef_trainer_splat.py SPLAT.ply GRID_CLOUD.ply GRID_CAMS.csv GEO_CAMS.csv OUT_DIR
      [--altitude-datum orthometric --geoid-undulation N]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from georef_splat.geodesy import geodetic_to_enu
from georef_splat.ply import Splats, read_ply, read_points, write_ply
from georef_splat.sim3 import Sim3, ransac_umeyama, residual_report, umeyama

NOTE = ("Camera-prior fit residual (trainer camera centres vs their own GPS). "
        "Not independent checkpoint accuracy.")


def read_cams(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.reader(open(path, newline="")))
    return {r[0]: np.array([float(v) for v in r[1:4]]) for r in rows[1:] if r and not r[0].startswith("#")}


def axis_rotations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            r = np.zeros((3, 3))
            for i, (j, sg) in enumerate(zip(perm, signs)):
                r[i, j] = sg
            if np.linalg.det(r) > 0:
                yield r


def splat_to_grid(splat_xyz: np.ndarray, grid_xyz: np.ndarray, seed: int = 0) -> tuple[Sim3, float]:
    rng = np.random.default_rng(seed)
    ref = grid_xyz[rng.choice(len(grid_xyz), min(400_000, len(grid_xyz)), replace=False)]
    tree = cKDTree(ref)
    src = splat_xyz[rng.choice(len(splat_xyz), min(40_000, len(splat_xyz)), replace=False)]
    best = None
    for r in axis_rotations():
        q = src @ r.T
        t = np.median(ref, 0) - np.median(q, 0)
        d, _ = tree.query(q + t, k=1)
        if best is None or np.median(d) < best[0]:
            best = (float(np.median(d)), Sim3(1.0, r, t))
    sim = best[1]
    # Trimmed ICP with a similarity update: keep the closest 70% of matches.
    for _ in range(30):
        moved = sim.apply(src)
        d, idx = tree.query(moved, k=1)
        keep = d <= np.percentile(d, 70)
        step = umeyama(moved[keep], ref[idx[keep]])
        sim = Sim3(step.scale * sim.scale, step.rotation @ sim.rotation, step.scale * step.rotation @ sim.translation + step.translation)
    d, _ = tree.query(sim.apply(src), k=1)
    return sim, float(np.median(d))


def rotation_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return np.array([w, x, y, z])


def transform_splats(s: Splats, sim: Sim3) -> Splats:
    from georef_splat.tiles import quat_multiply_wxyz

    q = rotation_to_quat_wxyz(sim.rotation)
    return Splats(
        positions=sim.apply(s.positions),
        log_scales=s.log_scales + math.log(sim.scale),
        rotations=quat_multiply_wxyz(q, s.rotations),
        opacity_logits=s.opacity_logits,
        sh_dc=s.sh_dc,
        sh_rest=s.sh_rest,  # degree 0 only in the trainer export we handle; rotate SH before using >0
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splat", type=Path)
    ap.add_argument("grid_cloud", type=Path)
    ap.add_argument("grid_cams", type=Path)
    ap.add_argument("geo_cams", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--altitude-datum", choices=["ellipsoidal", "orthometric"], default="ellipsoidal")
    ap.add_argument("--geoid-undulation", type=float, default=0.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    s = read_ply(a.splat)
    if s.sh_rest is not None:
        raise SystemExit("splat has higher-order SH; this script only rotates degree-0 exports")
    grid_xyz, grid_rgb = read_points(a.grid_cloud)
    s2g, icp_med = splat_to_grid(s.positions[s.opacity > 0.5], grid_xyz)
    print(f"splat->grid: scale {s2g.scale:.5f}, ICP median NN {icp_med:.3f} m")

    gcam, qcam = read_cams(a.grid_cams), read_cams(a.geo_cams)
    names = sorted(set(gcam) & set(qcam))
    lla = []
    for n in names:
        lon, lat, alt = qcam[n]
        lat, lon = math.degrees(lat), math.degrees(lon)
        if a.altitude_datum == "orthometric":
            alt += a.geoid_undulation
        lla.append((lat, lon, alt))
    arr = np.array(lla)
    origin = (float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
    enu = np.array([geodetic_to_enu(*p, origin) for p in lla])
    grid = np.array([gcam[n] for n in names])
    g2e, inliers = ransac_umeyama(grid, enu, threshold=3.0)
    fitted = g2e.apply(grid)
    rep = residual_report(fitted[inliers], enu[inliers])
    print(f"grid->ENU: {inliers.sum()}/{len(names)} inliers, scale {g2e.scale:.5f}, "
          f"H RMSE {rep['horizontal']['rmse_m']:.3f} m, V RMSE {rep['vertical']['rmse_m']:.3f} m")

    total = Sim3(g2e.scale * s2g.scale, g2e.rotation @ s2g.rotation, g2e.scale * g2e.rotation @ s2g.translation + g2e.translation)
    out_splat = transform_splats(s, total)
    write_ply(a.out / "splat_enu.ply", out_splat)
    # Ground candidates for the independent vertical check: the trainer's dense cloud in ENU.
    gc = g2e.apply(grid_xyz)
    np.save(a.out / "grid_cloud_enu.npy", np.c_[gc, grid_rgb].astype(np.float32))

    manifest = {
        "schema": "georef_splat.manifest.v2",
        "frame": "local ENU metres (x=east, y=north, z=up)",
        "origin": {"lat": origin[0], "lon": origin[1], "height_ellipsoidal": origin[2]},
        "exif_altitude_datum": a.altitude_datum,
        "geoid_undulation_m": a.geoid_undulation if a.altitude_datum == "orthometric" else None,
        "source": {"splat": a.splat.name, "splat_to_grid_icp_median_m": round(icp_med, 3),
                   "splat_to_grid_scale": s2g.scale},
        "images_with_gps": len(names),
        "inliers": int(inliers.sum()),
        "sim3_grid_to_enu": {"scale": g2e.scale, "rotation": g2e.rotation.tolist(), "translation": g2e.translation.tolist()},
        "residuals_inliers": rep,
        "residuals_all": residual_report(fitted, enu),
        "note": NOTE,
        "cameras": [{"image": n, "enu_gps": e.round(3).tolist(), "residual_m": (f - e).round(3).tolist(), "inlier": bool(i)}
                    for n, e, f, i in zip(names, enu, fitted, inliers)],
    }
    (a.out / "georeference_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {a.out / 'splat_enu.ply'} and manifest")


if __name__ == "__main__":
    main()
