"""Independent horizontal check: splat orthophoto vs USGS NAIP by phase correlation.

1. Render the splat top-down into a north-up colour raster (local ENU grid,
   which over a few hundred metres is a lat/lon grid up to a scale factor).
2. Fetch NAIP natural colour (public domain, ~0.6 m) for exactly that box.
3. Estimate the shift with FFT phase correlation on gradient-magnitude images
   (robust to the different dates, sun and colour), in overlapping tiles, and
   report the median shift and its spread.

NAIP is itself only accurate to a few metres (its spec is metre-level), so this
bounds gross horizontal error. It is not survey-grade validation.

Usage: python horizontal_check.py splat.ply georeference_manifest.json out_dir
"""

from __future__ import annotations

import io
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from georef_splat.geodesy import enu_to_geodetic
from georef_splat.ply import read_ply
from georef_splat.zones import Params, rasterise

NAIP_WMS = "https://imagery.nationalmap.gov/arcgis/services/USGSNAIPImagery/ImageServer/WMSServer"


def fetch_naip(west, south, east, north, w, h) -> np.ndarray:
    q = urllib.parse.urlencode({
        "service": "WMS", "request": "GetMap", "version": "1.1.1", "layers": "USGSNAIPImagery:NaturalColor",
        "styles": "", "srs": "EPSG:4326", "bbox": f"{west},{south},{east},{north}",
        "width": w, "height": h, "format": "image/png",
    })
    with urllib.request.urlopen(f"{NAIP_WMS}?{q}", timeout=180) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), dtype=np.float64) / 255.0


def grad(img: np.ndarray) -> np.ndarray:
    g = img.mean(-1)
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    return (m - m.mean()) / (m.std() + 1e-9)


def phase_correlate(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Shift (dy, dx) in pixels such that b ~= a shifted by (dy, dx); plus peak strength."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(a * win), np.fft.fft2(b * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-12
    c = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    peak = c[iy, ix] / (c.std() + 1e-12)
    # Sub-pixel: parabola through neighbours.
    def sub(cm, cp, c0):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d
    dy = iy + sub(c[iy - 1, ix], c[(iy + 1) % c.shape[0], ix], c[iy, ix])
    dx = ix + sub(c[iy, ix - 1], c[iy, (ix + 1) % c.shape[1]], c[iy, ix])
    if dy > a.shape[0] / 2: dy -= a.shape[0]
    if dx > a.shape[1] / 2: dx -= a.shape[1]
    return dy, dx, peak


def main(ply: str, manifest_path: str, out_dir: str) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    m = json.load(open(manifest_path))
    o = m["origin"]; origin = (o["lat"], o["lon"], o["height_ellipsoidal"])
    cams = np.array([c["enu_gps"] for c in m["cameras"]])
    s = read_ply(Path(ply))
    lo, hi = cams[:, :2].min(0) - 10, cams[:, :2].max(0) + 10
    s = s.subset(np.all((s.positions[:, :2] >= lo) & (s.positions[:, :2] <= hi), axis=1))

    cell = 0.6
    g = rasterise(s, Params(cell=cell, min_opacity=0.1, max_splat_size=4.0))
    ortho = g.rgb[::-1]                       # north-up
    valid = g.valid[::-1]
    ny, nx = valid.shape
    sw = enu_to_geodetic([g.x0, g.y0, 0], origin)
    ne = enu_to_geodetic([g.x0 + nx * cell, g.y0 + ny * cell, 0], origin)
    naip = fetch_naip(sw[1], sw[0], ne[1], ne[0], nx, ny)
    Image.fromarray((np.clip(ortho, 0, 1) * 255).astype(np.uint8)).save(out / "splat_ortho.png")
    Image.fromarray((naip * 255).astype(np.uint8)).save(out / "naip.png")

    ga, gb = grad(ortho), grad(naip)
    ga[~valid] = 0
    tile, step = 256, 128
    shifts = []
    for y in range(0, ny - tile + 1, step):
        for x in range(0, nx - tile + 1, step):
            if valid[y:y + tile, x:x + tile].mean() < 0.9:
                continue
            dy, dx, peak = phase_correlate(ga[y:y + tile, x:x + tile], gb[y:y + tile, x:x + tile])
            if peak > 8 and abs(dx) < 30 and abs(dy) < 30:
                # Image rows go south, so +row = -north. Report how far NAIP is from the splat.
                shifts.append((dx * cell, -dy * cell, float(peak)))
    if not shifts:
        raise SystemExit("no confident tiles; capture too small or imagery mismatch")
    sh = np.array(shifts)
    e, n = np.median(sh[:, 0]), np.median(sh[:, 1])
    spread = np.median(np.hypot(sh[:, 0] - e, sh[:, 1] - n))
    res = {
        "method": "phase correlation of gradient images, splat ortho vs USGS NAIP (WMS, EPSG:4326)",
        "cell_m": cell, "tiles_used": len(sh),
        "naip_minus_splat_m": {"east": round(float(e), 2), "north": round(float(n), 2), "horizontal": round(float(math.hypot(e, n)), 2)},
        "tile_spread_m": round(float(spread), 2),
        "tiles": sh.round(2).tolist(),
        "note": "NAIP is metre-level accurate itself; this bounds gross horizontal error, it is not survey validation.",
    }
    (out / "horizontal_check.json").write_text(json.dumps(res, indent=1))
    print(f"{len(sh)} tiles; NAIP - splat: E {e:+.2f} m, N {n:+.2f} m ({math.hypot(e, n):.2f} m), spread {spread:.2f} m")


if __name__ == "__main__":
    main(*sys.argv[1:4])
