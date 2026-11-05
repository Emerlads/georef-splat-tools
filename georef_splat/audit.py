"""Splat geometry audit: where can you trust a Gaussian splat's surface?

Splats are optimised for appearance, not geometry. On low-texture surfaces
(lawns, water, roofs) they often reproduce the right colours at the wrong
depth: floaters that look fine from the training views and are tens of metres
off in 3D. Anything that measures off the splat (heights, clearances, volumes)
inherits that error silently.

The audit renders the splat top-down (alpha compositing, as a viewer would)
and compares that surface with a geometric reference on the same grid:
a multi-view-stereo point cloud (photo-consistent depth) and, where available,
bare-earth lidar in open ground. Every cell gets a verdict.
"""

from __future__ import annotations

import numpy as np

from .ply import Splats
from .zones import Params, grid_extent, rasterise, rasterise_points

CONSISTENT, MARGINAL, FLOATING, SUNKEN, NO_REF = 0, 1, 2, 3, 4
VERDICT = {CONSISTENT: "consistent (|dz| <= 1 m)", MARGINAL: "marginal (1-2.5 m)", FLOATING: "floating above surface (> 2.5 m)",
           SUNKEN: "below surface (< -2.5 m)", NO_REF: "no reference"}
RGBA = {CONSISTENT: (60, 170, 110, 90), MARGINAL: (240, 190, 60, 170), FLOATING: (230, 60, 60, 200),
        SUNKEN: (70, 120, 230, 200), NO_REF: (0, 0, 0, 0)}


def audit(s: Splats, ref_pos: np.ndarray, ref_rgb: np.ndarray, p: Params | None = None, ground_fn=None) -> dict:
    p = p or Params(cell=0.5, min_opacity=0.1, max_splat_size=4.0)
    ext = grid_extent(ref_pos, p.cell)
    gs = rasterise(s, p, extent=ext)
    # No local ground correction here: it is fitted to the reference surface, which
    # would make "reference vs lidar" agree by construction.
    gr = rasterise_points(ref_pos, ref_rgb, p, ground_fn=ground_fn, extent=ext, local_correction=False)
    both = gs.valid & gr.valid
    dz = np.where(both, gs.dsm - gr.dsm, np.nan)

    v = np.full(dz.shape, NO_REF, dtype=np.uint8)
    a = np.abs(dz)
    v[both & (a <= 1.0)] = CONSISTENT
    v[both & (a > 1.0) & (a <= 2.5)] = MARGINAL
    v[both & (dz > 2.5)] = FLOATING
    v[both & (dz < -2.5)] = SUNKEN

    n = max(int(both.sum()), 1)
    pct = {VERDICT[k]: round(100 * float((v == k).sum()) / n, 1) for k in (CONSISTENT, MARGINAL, FLOATING, SUNKEN)}
    out = {
        "cell_m": p.cell,
        "cells_compared": int(both.sum()),
        "area_compared_m2": round(float(both.sum()) * p.cell**2),
        "verdict_pct": pct,
        "median_abs_dz_m": round(float(np.nanmedian(a)), 2),
        "p90_abs_dz_m": round(float(np.nanpercentile(a, 90)), 2),
        "worst_floaters_m": round(float(np.nanpercentile(dz, 99)), 1),
    }
    # Split by what the reference says is there: canopy vs open ground.
    for name, mask in (("over_canopy", both & (gr.ndsm >= 2.0)), ("over_open_ground", both & (gr.ndsm < 0.75))):
        if mask.any():
            out[name] = {
                "area_m2": round(float(mask.sum()) * p.cell**2),
                "floating_pct": round(100 * float((v[mask] == FLOATING).mean()), 1),
                "consistent_pct": round(100 * float((v[mask] == CONSISTENT).mean()), 1),
                "median_dz_m": round(float(np.nanmedian(dz[mask])), 2),
            }
    if ground_fn is not None:
        # Open ground per the reference (low and flat): compare both surfaces to lidar.
        open_ground = both & (gr.ndsm < 0.3)
        if open_ground.any():
            out["open_ground_vs_lidar"] = {
                "cells": int(open_ground.sum()),
                "splat_median_dz_m": round(float(np.median((gs.dsm - gr.dtm)[open_ground])), 2),
                "mvs_median_dz_m": round(float(np.median((gr.dsm - gr.dtm)[open_ground])), 2),
                "splat_p90_abs_dz_m": round(float(np.percentile(np.abs(gs.dsm - gr.dtm)[open_ground], 90)), 2),
                "mvs_p90_abs_dz_m": round(float(np.percentile(np.abs(gr.dsm - gr.dtm)[open_ground], 90)), 2),
            }
    return {"summary": out, "verdict": v, "dz": dz, "splat_grid": gs, "ref_grid": gr}


def verdict_image(v: np.ndarray) -> np.ndarray:
    img = np.zeros(v.shape + (4,), dtype=np.uint8)
    for k, c in RGBA.items():
        img[v == k] = c
    return img[::-1]
