"""``georef-splat`` command line.

  georef-splat align  --model sparse/0 --images images --output sparse_enu/0 [--altitude-datum orthometric --geoid-undulation -33.9]
  georef-splat tiles  --ply splat.ply --manifest georeference_manifest.json --out tiles/ [--max-splats 1500000]
  georef-splat zones  --points fused.ply --manifest georeference_manifest.json --out zones/ [--ground 3dep --ground-offset -1.27]
  georef-splat audit  --ply splat.ply --points fused.ply --manifest georeference_manifest.json --out audit/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import __version__
from .exif import read_gps_folder
from .geodesy import geodetic_to_enu
from .ply import read_ply
from .sim3 import ransac_umeyama, residual_report
from .tiles import write_tileset

NOTE = (
    "Camera-prior fit residual: how well solved camera centres agree with the GPS "
    "positions they were fitted to. Not independent checkpoint accuracy."
)


def cmd_align(a: argparse.Namespace) -> None:
    try:
        import pycolmap
    except ImportError as e:  # pragma: no cover - optional heavy dependency
        raise SystemExit("align needs pycolmap: pip install 'georef-splat[colmap]'") from e

    rec = pycolmap.Reconstruction(str(a.model))
    gps = read_gps_folder(a.images)
    names, lla, centres = [], [], []
    for img in rec.images.values():
        g = gps.get(Path(img.name).name)
        if g is None:
            continue
        lat, lon, alt = g
        if a.altitude_datum == "orthometric":
            alt = alt + a.geoid_undulation  # h = H + N
        names.append(img.name)
        lla.append((lat, lon, alt))
        centres.append(np.asarray(img.projection_center(), dtype=np.float64))
    if len(names) < a.min_images:
        raise SystemExit(f"only {len(names)} registered images have GPS; need {a.min_images}")

    arr = np.array(lla)
    origin = (float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
    enu = np.array([geodetic_to_enu(*p, origin) for p in lla])
    sim, inliers = ransac_umeyama(np.array(centres), enu, threshold=a.ransac_threshold)

    rec.transform(pycolmap.Sim3d(sim.scale, pycolmap.Rotation3d(sim.rotation), sim.translation))
    if a.output.exists():
        if not a.force:
            raise SystemExit(f"{a.output} exists; pass --force")
        shutil.rmtree(a.output)
    a.output.mkdir(parents=True)
    rec.write(str(a.output))

    fitted = sim.apply(np.array(centres))
    per_camera = [
        {"image": n, "enu_gps": e.round(3).tolist(), "residual_m": (f - e).round(3).tolist(), "inlier": bool(i)}
        for n, e, f, i in zip(names, enu, fitted, inliers)
    ]
    manifest = {
        "schema": "georef_splat.manifest.v2",
        "tool_version": __version__,
        "frame": "local ENU metres (x=east, y=north, z=up)",
        "origin": {"lat": origin[0], "lon": origin[1], "height_ellipsoidal": origin[2]},
        "exif_altitude_datum": a.altitude_datum,
        "geoid_undulation_m": a.geoid_undulation if a.altitude_datum == "orthometric" else None,
        "registered_images": len(rec.images),
        "images_with_gps": len(names),
        "inliers": int(inliers.sum()),
        "sim3": {"scale": sim.scale, "rotation": sim.rotation.tolist(), "translation": sim.translation.tolist()},
        "residuals_inliers": residual_report(fitted[inliers], enu[inliers]),
        "residuals_all": residual_report(fitted, enu),
        "note": NOTE,
        "cameras": per_camera,
    }
    out = a.manifest or a.output.parent.parent / "georeference_manifest.json"
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    r = manifest["residuals_inliers"]
    print(f"{int(inliers.sum())}/{len(names)} cameras inlier; scale {sim.scale:.4f}")
    print(f"horizontal RMSE {r['horizontal']['rmse_m']:.3f} m, vertical RMSE {r['vertical']['rmse_m']:.3f} m, 3D RMSE {r['three_d']['rmse_m']:.3f} m")
    print(f"manifest: {out}")


def local_ground(pos: np.ndarray, cell: float, q: float = 0.02) -> np.ndarray:
    """Per-point ground height: low percentile of z in each cell (numpy only)."""
    from .zones import _group_percentile

    ij = np.floor((pos[:, :2] - pos[:, :2].min(0)) / cell).astype(int)
    nx = ij[:, 0].max() + 1
    key = ij[:, 1] * nx + ij[:, 0]
    n = int(key.max()) + 1
    g = _group_percentile(key, pos[:, 2], n, q)
    return g[key]


def cmd_tiles(a: argparse.Namespace) -> None:
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    o = manifest["origin"]
    origin = (o["lat"], o["lon"], o.get("height_ellipsoidal", o.get("height")))
    s = read_ply(a.ply)
    n0 = len(s)

    keep = s.opacity >= a.min_opacity
    keep &= np.exp(s.log_scales).max(1) <= a.max_scale
    if a.crop_radius:
        keep &= np.linalg.norm(s.positions[:, :2], axis=1) <= a.crop_radius
    if a.crop_to_cameras is not None:
        cams = np.array([c["enu_gps"] for c in manifest["cameras"]])
        lo, hi = cams[:, :2].min(0) - a.crop_to_cameras, cams[:, :2].max(0) + a.crop_to_cameras
        keep &= np.all((s.positions[:, :2] >= lo) & (s.positions[:, :2] <= hi), axis=1)
    if a.clip_above_ground:
        # Nadir captures leave "floaters" between cameras and ground. Estimate
        # ground per 10 m cell (2nd percentile) and drop anything far above it.
        ground = local_ground(s.positions[keep], 10.0)
        keep[keep] &= s.positions[keep, 2] <= ground + a.clip_above_ground
    s = s.subset(keep)
    if a.max_splats and len(s) > a.max_splats:
        # Opacity-weighted random sampling keeps coverage even. Hard ranking by
        # size keeps big blurry splats; hard ranking by opacity deletes surfaces
        # built from many faint splats (lawns go dark). Both were tried.
        w = s.opacity / s.opacity.sum()
        idx = np.random.default_rng(0).choice(len(s), a.max_splats, replace=False, p=w)
        s = s.subset(np.sort(idx))

    if s.sh_rest is not None:
        have = {9: 1, 24: 2, 45: 3}.get(s.sh_rest.shape[1], 0)
        keep_deg = min(a.sh_degree, have)
        dim_keep, dim_have = {0: 0, 1: 3, 2: 8, 3: 15}[keep_deg], s.sh_rest.shape[1] // 3
        s.sh_rest = s.sh_rest.reshape(len(s), 3, dim_have)[:, :, :dim_keep].reshape(len(s), -1) if dim_keep else None
    summary = write_tileset(s, origin, a.out, extras={"source_ply": Path(a.ply).name, "input_splats": n0})
    print(json.dumps(summary, indent=1))


def _ortho_chips(g, structures, out: Path, margin_m: float = 14.0, size: int = 480) -> None:
    """Per-structure evidence chips cut from the gridded orthophoto, with footprint and rings drawn."""
    from PIL import Image, ImageDraw

    from .zones import FT, buffer_convex

    out.mkdir(parents=True, exist_ok=True)
    rgb = (np.clip(g.rgb, 0, 1) * 255).astype(np.uint8)
    ny, nx = g.dsm.shape
    for st in structures:
        hull = np.array(st["footprint_enu"])
        lo, hi = hull.min(0) - margin_m, hull.max(0) + margin_m
        half = max(hi - lo) / 2
        c = (lo + hi) / 2
        x0, x1 = int((c[0] - half - g.x0) / g.cell), int((c[0] + half - g.x0) / g.cell)
        y0, y1 = int((c[1] - half - g.y0) / g.cell), int((c[1] + half - g.y0) / g.cell)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, nx), min(y1, ny)
        crop = rgb[y0:y1, x0:x1][::-1]  # north-up
        im = Image.fromarray(crop).resize((size, size), Image.LANCZOS)
        sx, sy = size / max(x1 - x0, 1), size / max(y1 - y0, 1)
        def px(pts):
            return [((x - g.x0) / g.cell - x0) * sx for x in pts[:, 0]], [(y1 - (y - g.y0) / g.cell) * sy for y in pts[:, 1]]
        d = ImageDraw.Draw(im)
        for r, col in ((30 * FT, (240, 170, 40)), (5 * FT, (230, 70, 60))):
            xs, ys = px(buffer_convex(hull, r))
            d.line(list(zip(xs, ys)) + [(xs[0], ys[0])], fill=col, width=2)
        xs, ys = px(hull)
        d.line(list(zip(xs, ys)) + [(xs[0], ys[0])], fill=(255, 255, 255), width=3)
        im.save(out / f"structure_{st['id']}.jpg", quality=85)


def cmd_zones(a: argparse.Namespace) -> None:
    from PIL import Image

    from .geodesy import enu_to_geodetic
    from .zones import CLASS_NAMES, ZONES, Params, buffer_convex, class_image, classify, rasterise, screen_structures

    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    o = manifest["origin"]
    origin = (o["lat"], o["lon"], o.get("height_ellipsoidal", o.get("height")))
    if (a.ply is None) == (a.points is None):
        raise SystemExit("give exactly one of --ply (splat) or --points (e.g. COLMAP MVS fused.ply)")
    if a.points is not None:
        from .ply import read_points

        pos, rgb = read_points(a.points)
    else:
        sp = read_ply(a.ply)
        pos = sp.positions
    keep = np.ones(len(pos), dtype=bool)
    if a.crop_radius:
        keep &= np.linalg.norm(pos[:, :2], axis=1) <= a.crop_radius
    if a.crop_to_cameras is not None:
        cams = np.array([c["enu_gps"] for c in manifest["cameras"]])
        lo, hi = cams[:, :2].min(0) - a.crop_to_cameras, cams[:, :2].max(0) + a.crop_to_cameras
        keep &= np.all((pos[:, :2] >= lo) & (pos[:, :2] <= hi), axis=1)
    if a.clip_above_ground:
        keep[keep] &= pos[keep, 2] <= local_ground(pos[keep], 10.0) + a.clip_above_ground
    p = Params(cell=a.cell, min_opacity=a.min_opacity, max_splat_size=a.max_splat_size)
    ground_fn = None
    if a.ground == "3dep":
        from .usgs import dep_ground_fn

        ground_fn = dep_ground_fn(origin, offset_m=a.ground_offset)
    if a.points is not None:
        from .zones import rasterise_points

        g = rasterise_points(pos[keep], rgb[keep], p, ground_fn=ground_fn)
    else:
        g = rasterise(sp.subset(keep), p, ground_fn=ground_fn)
    classify(g, p)
    structures = screen_structures(g, p)

    def ll(poly, z):
        pts = [enu_to_geodetic([x, y, z], origin) for x, y in poly]
        ring = [[round(lon, 8), round(lat, 8)] for lat, lon, _ in pts]
        return ring + [ring[0]]

    feats = []
    for st in structures:
        hull = np.array(st["footprint_enu"])
        z = st["ground_z_enu"]
        props = {k: v for k, v in st.items() if k not in ("footprint_enu", "zones")}
        props["flags"] = "; ".join(st["flags"])
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ll(hull, z)]},
                      "properties": {"layer": "structure", **props}})
        for name, r0, r1 in ZONES:
            outer = ll(buffer_convex(hull, r1), z)
            inner = ll(buffer_convex(hull, r0), z)[::-1]
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [outer, inner]},
                          "properties": {"layer": name, "structure_id": st["id"], **st["zones"][name]}})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "name": "defensible_space_screening", "features": feats}
    (out / "zones.geojson").write_text(json.dumps(fc), encoding="utf-8")
    Image.fromarray(class_image(g), "RGBA").save(out / "classes.png", optimize=True)

    ny, nx = g.classes.shape
    sw = enu_to_geodetic([g.x0, g.y0, 0], origin)
    ne = enu_to_geodetic([g.x0 + nx * g.cell, g.y0 + ny * g.cell, 0], origin)
    counts = {CLASS_NAMES[c]: round(100 * float((g.classes[g.valid] == c).mean()), 1) for c in CLASS_NAMES}
    summary = {
        "method": ("dense point cloud gridded at a high percentile (surface, colour)" if a.points is not None
                   else "splat rendered top-down by alpha compositing (surface, colour)")
                  + "; ground from " + ("USGS 3DEP lidar" if a.ground == "3dep" else "morphological opening")
                  + "; heuristic classes; CA defensible-space rings",
        "cell_m": a.cell,
        "overlay": {"png": "classes.png", "west": sw[1], "south": sw[0], "east": ne[1], "north": ne[0],
                    "ground_z_enu": float(np.median(g.dtm[g.valid]))},
        "class_cover_pct": counts,
        "structures": [{k: v for k, v in st.items() if k != "footprint_enu"} for st in structures],
        "note": "Screening heuristic to prioritise inspections. Not a compliance determination.",
    }
    (out / "zones_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "footprints_enu.json").write_text(json.dumps({st["id"]: st["footprint_enu"] for st in structures}), encoding="utf-8")
    if a.chips:
        _ortho_chips(g, structures, out / "chips")
    print(f"{len(structures)} structures; classes {counts}")
    for st in structures:
        print(f"  #{st['id']}: {st['footprint_area_m2']} m2, {st['height_m']} m tall, {st['priority']}: {'; '.join(st['flags']) or 'no flags'}")


def _load_cropped(a, manifest):
    """Splats cropped to the camera footprint and clipped of high floaters, per CLI flags."""
    s = read_ply(a.ply)
    keep = np.ones(len(s), dtype=bool)
    if a.crop_to_cameras is not None:
        cams = np.array([c["enu_gps"] for c in manifest["cameras"]])
        lo, hi = cams[:, :2].min(0) - a.crop_to_cameras, cams[:, :2].max(0) + a.crop_to_cameras
        keep &= np.all((s.positions[:, :2] >= lo) & (s.positions[:, :2] <= hi), axis=1)
    if getattr(a, "clip_above_ground", 0):
        keep[keep] &= s.positions[keep, 2] <= local_ground(s.positions[keep], 10.0) + a.clip_above_ground
    return s.subset(keep)


def _overlay_bounds(g, origin):
    from .geodesy import enu_to_geodetic

    ny, nx = g.dsm.shape
    sw = enu_to_geodetic([g.x0, g.y0, 0], origin)
    ne = enu_to_geodetic([g.x0 + nx * g.cell, g.y0 + ny * g.cell, 0], origin)
    return {"west": sw[1], "south": sw[0], "east": ne[1], "north": ne[0]}


def cmd_audit(a: argparse.Namespace) -> None:
    from PIL import Image

    from .audit import audit, verdict_image
    from .ply import read_points
    from .zones import Params

    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    o = manifest["origin"]
    origin = (o["lat"], o["lon"], o.get("height_ellipsoidal", o.get("height")))
    s = _load_cropped(a, manifest)
    pos, rgb = read_points(a.points)
    if a.crop_to_cameras is not None:
        cams = np.array([c["enu_gps"] for c in manifest["cameras"]])
        lo, hi = cams[:, :2].min(0) - a.crop_to_cameras, cams[:, :2].max(0) + a.crop_to_cameras
        k = np.all((pos[:, :2] >= lo) & (pos[:, :2] <= hi), axis=1)
        pos, rgb = pos[k], rgb[k]
    ground_fn = None
    if a.ground == "3dep":
        from .usgs import dep_ground_fn

        ground_fn = dep_ground_fn(origin, offset_m=a.ground_offset)
    r = audit(s, pos, rgb, Params(cell=a.cell, min_opacity=0.1, max_splat_size=4.0), ground_fn=ground_fn)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(verdict_image(r["verdict"]), "RGBA").save(out / "audit.png", optimize=True)
    summary = {
        "method": "splat rendered top-down (alpha compositing) vs COLMAP MVS surface on the same grid"
                  + ("; open ground also vs USGS 3DEP lidar" if ground_fn else ""),
        **r["summary"],
        "overlay": {"png": "audit.png", **_overlay_bounds(r["ref_grid"], origin),
                    "ground_z_enu": float(np.median(r["ref_grid"].dtm))},
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(r["summary"], indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="georef-splat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(required=True)

    al = sub.add_parser("align", help="Sim3-align a COLMAP model to EXIF GPS in local ENU metres")
    al.add_argument("--model", type=Path, required=True)
    al.add_argument("--images", type=Path, required=True)
    al.add_argument("--output", type=Path, required=True)
    al.add_argument("--manifest", type=Path)
    al.add_argument("--altitude-datum", choices=["ellipsoidal", "orthometric"], default="ellipsoidal",
                    help="what the EXIF GPSAltitude is referenced to")
    al.add_argument("--geoid-undulation", type=float, default=0.0, help="N in metres (h = H + N)")
    al.add_argument("--ransac-threshold", type=float, default=5.0)
    al.add_argument("--min-images", type=int, default=10)
    al.add_argument("--force", action="store_true")
    al.set_defaults(func=cmd_align)

    t = sub.add_parser("tiles", help="ENU splat PLY -> 3D Tiles 1.1 (KHR_gaussian_splatting + SPZ)")
    t.add_argument("--ply", type=Path, required=True)
    t.add_argument("--manifest", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--max-splats", type=int, default=0)
    t.add_argument("--min-opacity", type=float, default=0.02)
    t.add_argument("--max-scale", type=float, default=5.0, help="drop splats with any axis larger than this (m)")
    t.add_argument("--crop-radius", type=float, default=0.0, help="keep splats within this horizontal radius (m) of the origin")
    t.add_argument("--crop-to-cameras", type=float, default=None, metavar="MARGIN",
                   help="keep splats inside the camera bounding box plus MARGIN metres")
    t.add_argument("--clip-above-ground", type=float, default=0.0, metavar="H",
                   help="drop splats more than H metres above a locally estimated ground (removes floaters)")
    t.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3],
                   help="spherical-harmonic degree to keep (view-dependent colour); lower = smaller file")
    t.set_defaults(func=cmd_tiles)

    z = sub.add_parser("zones", help="defensible-space screening (Zone 0/1/2) from an ENU splat or point cloud")
    z.add_argument("--ply", type=Path, help="Gaussian splat PLY (ENU)")
    z.add_argument("--points", type=Path, help="coloured point cloud PLY (ENU), e.g. COLMAP MVS; preferred for heights")
    z.add_argument("--manifest", type=Path, required=True)
    z.add_argument("--out", type=Path, required=True)
    z.add_argument("--cell", type=float, default=0.5)
    z.add_argument("--crop-radius", type=float, default=0.0)
    z.add_argument("--crop-to-cameras", type=float, default=None, metavar="MARGIN")
    z.add_argument("--clip-above-ground", type=float, default=0.0, metavar="H")
    z.add_argument("--min-opacity", type=float, default=0.1)
    z.add_argument("--max-splat-size", type=float, default=4.0)
    z.add_argument("--ground", choices=["splat", "3dep"], default="splat",
                   help="bare-earth source: opening of the splat surface, or USGS 3DEP lidar (US only)")
    z.add_argument("--chips", action="store_true", help="write per-structure orthophoto evidence chips")
    z.add_argument("--ground-offset", type=float, default=0.0,
                   help="reconstruction minus 3DEP vertical bias (m), e.g. from vertical_check.py")
    z.set_defaults(func=cmd_zones)

    au = sub.add_parser("audit", help="where is the splat's geometry trustworthy? (vs MVS points / 3DEP)")
    au.add_argument("--ply", type=Path, required=True, help="Gaussian splat PLY (ENU)")
    au.add_argument("--points", type=Path, required=True, help="reference point cloud PLY in the same ENU frame (e.g. COLMAP fused.ply)")
    au.add_argument("--manifest", type=Path, required=True)
    au.add_argument("--out", type=Path, required=True)
    au.add_argument("--cell", type=float, default=0.5)
    au.add_argument("--crop-to-cameras", type=float, default=None, metavar="MARGIN")
    au.add_argument("--clip-above-ground", type=float, default=0.0, metavar="H",
                    help="apply the same floater clip used for delivery, so you audit what you ship")
    au.add_argument("--ground", choices=["none", "3dep"], default="none")
    au.add_argument("--ground-offset", type=float, default=0.0)
    au.set_defaults(func=cmd_audit)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
