"""Defensible-space screening from a georeferenced surface (MVS points or a splat).

No footprints and no trained imagery classifier. Bare earth comes from USGS
3DEP lidar when a ``ground_fn`` is given, otherwise from the surface itself.

1. Rasterise opaque splats (or a dense point cloud) onto an ENU grid -> surface
   model (DSM), ground model (DTM, by morphological opening or an external
   bare-earth model), height above ground (nDSM), and top-surface colour.
2. Classify cells: structure (tall, flat, not green), tree (tall, green),
   shrub / ladder fuel (0.3-2 m, green), low vegetation, ground.
3. Find structures as connected components; take each one's convex footprint.
4. Measure vegetation in the California defensible-space rings around each
   structure: Zone 0 (0-5 ft), Zone 1 (5-30 ft), Zone 2 (30-100 ft).

This is a screening heuristic for prioritising inspections, not a compliance
determination. Thresholds are parameters, not truths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ply import Splats

SH_C0 = 0.28209479177387814
FT = 0.3048
ZONES = (("zone0", 0.0, 5 * FT), ("zone1", 5 * FT, 30 * FT), ("zone2", 30 * FT, 100 * FT))

GROUND, LOWVEG, SHRUB, TREE, STRUCTURE, OBJECT = 0, 1, 2, 3, 4, 5
CLASS_NAMES = {GROUND: "ground", LOWVEG: "low vegetation", SHRUB: "shrub / tall grass", TREE: "tree canopy",
               STRUCTURE: "structure", OBJECT: "raised non-green (dry vegetation or objects)"}


@dataclass
class Params:
    cell: float = 0.5
    min_opacity: float = 0.35
    max_splat_size: float = 1.5
    ground_window_m: float = 24.0
    shrub_min_h: float = 0.75  # above the ~0.3-0.6 m noise floor of photogrammetric surface vs lidar
    tree_min_h: float = 2.0
    structure_min_h: float = 2.2
    structure_max_roughness: float = 0.35
    structure_min_area: float = 30.0
    green_exg: float = 0.07


@dataclass
class Grid:
    x0: float
    y0: float
    cell: float
    dsm: np.ndarray
    dtm: np.ndarray
    ndsm: np.ndarray
    rgb: np.ndarray
    exg: np.ndarray
    valid: np.ndarray
    classes: np.ndarray = field(default=None)

    def centres(self):
        ny, nx = self.dsm.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys)


def _group_percentile(keys: np.ndarray, values: np.ndarray, n_cells: int, q: float) -> np.ndarray:
    order = np.lexsort((values, keys))
    k, v = keys[order], values[order]
    starts = np.searchsorted(k, np.arange(n_cells), "left")
    ends = np.searchsorted(k, np.arange(n_cells), "right")
    counts = ends - starts
    out = np.full(n_cells, np.nan)
    has = counts > 0
    idx = starts[has] + np.floor((counts[has] - 1) * q).astype(int)
    out[has] = v[idx]
    return out


def horizontal_footprint(s: Splats) -> np.ndarray:
    """Area (m^2) of each Gaussian's 1-sigma ellipse projected onto the ground plane."""
    q = s.rotations / np.linalg.norm(s.rotations, axis=1, keepdims=True)
    w, x, y, z = q.T
    r = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
    ], 1)  # first two rows of R: (N, 2, 3)
    m = r * np.exp(s.log_scales)[:, None, :]
    cov = m @ np.transpose(m, (0, 2, 1))  # (N, 2, 2) horizontal covariance
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    return np.pi * np.sqrt(np.maximum(det, 0))


def _composite(key, z, alpha, rgb, n):
    """Front-to-back alpha compositing per cell, looking straight down.

    Returns surface height (where transmittance first drops below 0.5), colour,
    and accumulated opacity per cell. Sparse, faint floaters barely register,
    exactly as when the splat is viewed.
    """
    order = np.lexsort((-z, key))
    k, zs, a, c = key[order], z[order], alpha[order], rgb[order]
    logt = np.log1p(-a)
    cs = np.cumsum(logt)
    starts = np.searchsorted(k, k, "left")
    prev = np.where(starts > 0, cs[starts - 1], 0.0)
    cs_g = cs - prev
    w = np.exp(cs_g - logt) * a  # transmittance before this splat x its alpha
    acc = np.bincount(k, weights=w, minlength=n)
    col = np.stack([np.bincount(k, weights=w * c[:, i], minlength=n) for i in range(3)], 1)
    col /= np.maximum(acc, 1e-9)[:, None]
    zmean = np.bincount(k, weights=w * zs, minlength=n) / np.maximum(acc, 1e-9)
    hit = np.flatnonzero(cs_g < np.log(0.5))
    surf = zmean.copy()
    if hit.size:
        cells, first = np.unique(k[hit], return_index=True)
        surf[cells] = zs[hit[first]]
    return surf, col, acc


def grid_extent(pos: np.ndarray, cell: float) -> tuple[float, float, int, int]:
    x0, y0 = np.floor(pos[:, 0].min()), np.floor(pos[:, 1].min())
    nx = int(np.ceil((pos[:, 0].max() - x0) / cell)) + 1
    ny = int(np.ceil((pos[:, 1].max() - y0) / cell)) + 1
    return x0, y0, nx, ny


def local_ground_offset(resid: np.ndarray, valid: np.ndarray, cell: float, block_m: float = 4.0,
                        window_blocks: int = 9, clamp_m: float = 2.0) -> np.ndarray:
    """Smooth field of (reconstructed ground - reference ground).

    An external bare-earth model (lidar) and a photogrammetric surface rarely
    agree to better than ~0.5 m; the offset also drifts across a site. In every
    ~36 m neighbourhood the lowest visible surfaces are ground, so a low
    percentile of (surface - reference) estimates the local offset. Clamped so
    a neighbourhood that is all canopy cannot lift the ground into the trees.
    """
    from scipy import ndimage

    b = max(1, int(round(block_m / cell)))
    ny, nx = resid.shape
    by, bx = -(-ny // b), -(-nx // b)
    pad = np.full((by * b, bx * b), np.nan)
    pad[:ny, :nx] = np.where(valid, resid, np.nan)
    blocks = pad.reshape(by, b, bx, b).transpose(0, 2, 1, 3).reshape(by, bx, -1)
    with np.errstate(all="ignore"):
        low = np.nanpercentile(blocks, 10, axis=2)
    known = ~np.isnan(low)
    if not known.any():
        return np.zeros_like(resid)
    filled = np.where(known, low, np.nanmax(low))
    field = ndimage.percentile_filter(filled, 20, size=window_blocks, mode="nearest")
    field = np.clip(ndimage.gaussian_filter(field, 1.5), -clamp_m, clamp_m)
    up = np.repeat(np.repeat(field, b, 0), b, 1)[:ny, :nx]
    return ndimage.gaussian_filter(up, b / 2)


def _finish(x0, y0, nx, ny, p, dsm, col, valid, ground_fn, local_correction=True) -> Grid:
    from scipy import ndimage

    if ground_fn is not None:
        dtm = np.asarray(ground_fn(x0, y0, nx, ny, p.cell), dtype=np.float64)
        if local_correction:
            dtm = dtm + local_ground_offset(dsm - dtm, valid, p.cell)
    else:
        w = max(3, int(p.ground_window_m / p.cell) | 1)
        base = np.where(valid, dsm, np.nanmax(dsm[valid]) + 100)
        opened = np.where(valid, ndimage.grey_opening(base, size=(w, w)), np.nan)
        holes = np.isnan(opened)
        if holes.any():
            _, (iy_, ix_) = ndimage.distance_transform_edt(holes, return_indices=True)
            opened = opened[iy_, ix_]
        dtm = ndimage.gaussian_filter(opened, 2)
    tot = col.sum(-1) + 1e-6
    exg = (2 * col[..., 1] - col[..., 0] - col[..., 2]) / tot
    ndsm = np.where(valid, np.maximum(dsm - dtm, 0.0), 0.0)
    return Grid(x0, y0, p.cell, dsm, dtm, ndsm, col, exg, valid)


def rasterise_points(pos: np.ndarray, rgb: np.ndarray, p: Params, ground_fn=None, extent=None,
                     local_correction: bool = True) -> Grid:
    """Surface model from a dense point cloud (e.g. COLMAP MVS): high percentile per cell."""
    from scipy import ndimage

    x0, y0, nx, ny = extent or grid_extent(pos, p.cell)
    ix = ((pos[:, 0] - x0) / p.cell).astype(int)
    iy = ((pos[:, 1] - y0) / p.cell).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    pos, rgb, key = pos[inside], rgb[inside], (iy * nx + ix)[inside]
    n = nx * ny
    dsm = _group_percentile(key, pos[:, 2], n, 0.95)
    cnt = np.bincount(key, minlength=n)
    top = pos[:, 2] >= dsm[key] - 0.5
    col = np.stack([np.bincount(key[top], weights=rgb[top, i], minlength=n) for i in range(3)], 1)
    col /= np.maximum(np.bincount(key[top], minlength=n), 1)[:, None]
    dsm, col, valid = dsm.reshape(ny, nx), col.reshape(ny, nx, 3), (cnt >= 2).reshape(ny, nx)
    # Fill pinholes (<= 1.5 m) from neighbours; leave genuine gaps invalid.
    holes = ~valid
    near = ndimage.distance_transform_edt(holes, return_distances=True, return_indices=True)
    dist, (iy_, ix_) = near
    small = holes & (dist * p.cell <= 1.5)
    dsm = np.where(small, dsm[iy_, ix_], dsm)
    col = np.where(small[..., None], col[iy_, ix_], col)
    valid = valid | small
    return _finish(x0, y0, nx, ny, p, np.nan_to_num(dsm), col, valid, ground_fn, local_correction)


def rasterise(s: Splats, p: Params, ground_fn=None, extent=None) -> Grid:
    """Render the splat from directly above onto an ENU grid.

    ``ground_fn(x0, y0, nx, ny, cell) -> (ny, nx)`` may supply an external bare-earth
    model in the same ENU heights (e.g. USGS 3DEP). Without it, ground comes from a
    morphological opening of the rendered surface.
    """
    keep = (s.opacity >= p.min_opacity) & (np.exp(s.log_scales).max(1) <= p.max_splat_size)
    s = s.subset(keep)
    pos = s.positions
    rgb = np.clip(0.5 + SH_C0 * s.sh_dc, 0, 1)
    area = horizontal_footprint(s)
    x0, y0, nx, ny = extent or grid_extent(pos, p.cell)

    def at(cell):
        cx = nx if cell == p.cell else int(np.ceil(nx * p.cell / cell)) + 1
        cy = ny if cell == p.cell else int(np.ceil(ny * p.cell / cell)) + 1
        ix = ((pos[:, 0] - x0) / cell).astype(int)
        iy = ((pos[:, 1] - y0) / cell).astype(int)
        ins = (ix >= 0) & (ix < cx) & (iy >= 0) & (iy < cy)
        alpha = np.clip(s.opacity * np.minimum(1.0, area / cell**2), 0, 0.995)
        surf, col, acc = _composite((iy * cx + ix)[ins], pos[ins, 2], alpha[ins], rgb[ins], cx * cy)
        return surf.reshape(cy, cx), col.reshape(cy, cx, 3), acc.reshape(cy, cx)

    dsm, col, acc = at(p.cell)
    valid = acc >= 0.3
    # Big flat splats (lawns, roofs) are wider than a fine cell; fill fine-grid gaps
    # from a 4x coarser render instead of leaving holes.
    cs, ccol, cacc = at(p.cell * 4)
    up = lambda a: np.repeat(np.repeat(a, 4, 0), 4, 1)[:ny, :nx]
    fill = ~valid & (up(cacc) >= 0.3)
    dsm = np.where(fill, up(cs), dsm)
    col = np.where(fill[..., None], up(ccol), col)
    valid |= fill
    return _finish(x0, y0, nx, ny, p, dsm, col, valid, ground_fn)


def classify(g: Grid, p: Params) -> np.ndarray:
    from scipy import ndimage

    h = g.ndsm
    green = ndimage.median_filter(g.exg, 3) > p.green_exg
    d = np.where(g.valid, g.dsm, g.dtm)
    mean = ndimage.uniform_filter(d, 3, mode="nearest")
    rough = np.sqrt(np.maximum(ndimage.uniform_filter(d * d, 3, mode="nearest") - mean * mean, 0))
    cls = np.full(h.shape, GROUND, dtype=np.uint8)
    cls[green & (h < p.shrub_min_h)] = LOWVEG
    cls[green & (h >= p.shrub_min_h) & (h < p.tree_min_h)] = SHRUB
    cls[green & (h >= p.tree_min_h)] = TREE

    cand = (h >= p.structure_min_h) & ~green & (rough <= p.structure_max_roughness) & g.valid
    cand = ndimage.binary_opening(cand, iterations=1)
    cand = ndimage.binary_closing(cand, iterations=2)
    lab, n = ndimage.label(cand)
    min_cells = p.structure_min_area / (p.cell**2)
    sizes = ndimage.sum(cand, lab, range(1, n + 1))
    # Buildings are at least ~3 m wide: a component must survive a 1.5 m erosion,
    # which rejects path edges, fences and other thin slivers.
    core = ndimage.binary_erosion(cand, iterations=max(1, int(round(1.5 / p.cell))))
    has_core = ndimage.maximum(core, lab, range(1, n + 1))
    ring_iter = max(2, int(round(3.0 / p.cell)))
    for i, (sz, c) in enumerate(zip(sizes, has_core), start=1):
        if sz < min_cells or not c:
            continue
        comp = lab == i
        # A building stands above its immediate surroundings. Comparing with a
        # 3 m ring (its lower quartile, i.e. the ground beside it) rejects paths
        # and pads that only look raised because the ground model is off.
        ring = ndimage.binary_dilation(comp, iterations=ring_iter) & ~comp & g.valid
        if ring.sum() < 8 or np.median(g.dsm[comp]) - np.percentile(g.dsm[ring], 25) < p.structure_min_h * 0.8:
            continue
        cls[comp] = STRUCTURE
        # Eaves: edge cells mix roof and grass colours, so they read as "tall and
        # green". Fold cells within 1 m that sit at roof height into the structure.
        fringe = ndimage.binary_dilation(comp, iterations=max(1, int(round(1.0 / p.cell)))) & ~comp
        roof_h = np.median(h[comp])
        cls[fringe & (h >= 0.5 * roof_h)] = STRUCTURE
    # Tall non-green rough cells (dead/leafless canopy) count as tree fuel; other
    # raised non-green things (furniture, stored material, vehicles) are objects.
    cls[(cls == GROUND) & (h >= p.tree_min_h) & (rough > p.structure_max_roughness) & g.valid] = TREE
    cls[(cls == GROUND) & (h >= p.shrub_min_h) & g.valid] = OBJECT
    # Small "woody" patches whose average greenness is weak are objects picking
    # up grass colour at their edges (tables, bins, vehicles), not plants.
    wl, wn = ndimage.label(np.isin(cls, (SHRUB, TREE)))
    if wn:
        idx = np.arange(1, wn + 1)
        area = ndimage.sum(np.ones_like(h), wl, idx) * p.cell**2
        mexg = ndimage.mean(g.exg, wl, idx)
        bad = idx[(area < 10.0) & (mexg < 2 * p.green_exg)]
        cls[np.isin(wl, bad)] = OBJECT
    g.classes = cls
    return cls


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(pts, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        h = []
        for q in points:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def buffer_convex(hull: np.ndarray, r: float, segments: int = 24) -> np.ndarray:
    """Exact buffer of a convex polygon: hull of circles centred on its vertices."""
    if r <= 0:
        return hull
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], 1) * r
    return _convex_hull((hull[:, None, :] + ring[None]).reshape(-1, 2))


def _rasterise_polygon(poly: np.ndarray, g: Grid) -> np.ndarray:
    from PIL import Image, ImageDraw

    ny, nx = g.dsm.shape
    img = Image.new("1", (nx, ny), 0)
    px = [((x - g.x0) / g.cell, (y - g.y0) / g.cell) for x, y in poly]
    ImageDraw.Draw(img).polygon(px, fill=1)
    return np.array(img, dtype=bool)


def screen_structures(g: Grid, p: Params) -> list[dict]:
    from scipy import ndimage

    cls = g.classes
    lab, n = ndimage.label(cls == STRUCTURE)
    cx, cy = g.centres()
    results = []
    for i in range(1, n + 1):
        m = lab == i
        # Eaves-fringe merging can leave detached slivers labelled as structure.
        if m.sum() * g.cell**2 < p.structure_min_area:
            continue
        cells = np.stack([cx[m], cy[m]], 1)
        # Footprint corners rather than centres, so the hull covers whole cells.
        off = g.cell / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        hull = _convex_hull((cells[:, None, :] + off[None]).reshape(-1, 2))
        foot = _rasterise_polygon(hull, g)
        dist = ndimage.distance_transform_edt(~foot) * g.cell
        zones = {}
        for name, r0, r1 in ZONES:
            z = (dist > r0) & (dist <= r1) & g.valid if r0 > 0 else (dist > 0) & (dist <= r1) & g.valid
            area = z.sum() * g.cell**2
            veg = np.isin(cls, (SHRUB, TREE)) & z
            vh = g.ndsm[veg]
            zones[name] = {
                "inner_ft": round(r0 / FT), "outer_ft": round(r1 / FT),
                "area_m2": round(float(area), 1),
                "woody_cover_pct": round(100 * float(veg.sum()) / max(z.sum(), 1), 1),
                "tree_cover_pct": round(100 * float((cls[z] == TREE).sum()) / max(z.sum(), 1), 1),
                "shrub_cover_pct": round(100 * float((cls[z] == SHRUB).sum()) / max(z.sum(), 1), 1),
                "object_cover_pct": round(100 * float((cls[z] == OBJECT).sum()) / max(z.sum(), 1), 1),
                "max_veg_height_m": round(float(vh.max()), 1) if vh.size else 0.0,
                "observed_pct": round(100 * float(z.sum()) / max(((dist > r0) & (dist <= r1)).sum(), 1), 1),
            }
        z0, z1 = zones["zone0"], zones["zone1"]
        flags = []
        if z0["woody_cover_pct"] > 2:
            flags.append(f"Woody vegetation in Zone 0 ({z0['woody_cover_pct']}% of 0-5 ft ring)")
        if z1["tree_cover_pct"] > 40:
            flags.append(f"Dense canopy in Zone 1 ({z1['tree_cover_pct']}%)")
        if z1["shrub_cover_pct"] > 15:
            flags.append(f"Shrub / tall grass in Zone 1 ({z1['shrub_cover_pct']}%)")
        if z0["object_cover_pct"] > 2:
            flags.append(f"Raised non-green material in Zone 0 ({z0['object_cover_pct']}%): dry vegetation or stored objects; check for combustibles")
        if z0["observed_pct"] < 60:
            flags.append("Zone 0 poorly observed; capture more obliques")
        roof = g.dsm[foot & g.valid]
        results.append({
            "id": len(results) + 1,
            # Ranking score for triage (not a risk model): Zone 0 fuel counts most.
            "score": round(3 * z0["woody_cover_pct"] + 2 * z0["object_cover_pct"]
                           + 1.0 * z1["tree_cover_pct"] + 1.0 * z1["shrub_cover_pct"]
                           + 0.25 * (zones["zone2"]["tree_cover_pct"] + zones["zone2"]["shrub_cover_pct"]), 1),
            "footprint_enu": hull.round(2).tolist(),
            "footprint_area_m2": round(float(foot.sum()) * g.cell**2, 1),
            "height_m": round(float(np.percentile(g.ndsm[m], 90)), 1),
            "roof_z_enu": round(float(np.median(roof)), 2) if roof.size else None,
            "ground_z_enu": round(float(np.median(g.dtm[m])), 2),
            "zones": zones,
            "flags": flags,
            "priority": ("review" if any(f.startswith("Woody vegetation in Zone 0") for f in flags)
                         or sum(not f.startswith(("Zone 0 poorly", "Raised non-green")) for f in flags) >= 2
                         else ("watch" if flags else "ok")),
        })
    return results


CLASS_RGBA = {
    GROUND: (0, 0, 0, 0),
    LOWVEG: (140, 190, 90, 70),
    SHRUB: (240, 170, 40, 200),
    TREE: (40, 140, 70, 170),
    STRUCTURE: (230, 70, 60, 210),
    OBJECT: (150, 90, 210, 220),
}


def class_image(g: Grid) -> np.ndarray:
    """RGBA image, north-up (row 0 = north edge)."""
    out = np.zeros(g.classes.shape + (4,), dtype=np.uint8)
    for c, rgba in CLASS_RGBA.items():
        out[g.classes == c] = rgba
    return out[::-1]
