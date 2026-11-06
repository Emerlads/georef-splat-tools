#!/usr/bin/env python3
"""
Build the static data behind the "Fire Spread Replay & Exposure Timeline" page.

Input: NASA FEDS (Fire Event Data Suite) large-fire NRT layers for one fire,
downloaded from the openveda.cloud OGC API as GeoJSON:
    lf_perimeter_nrt.geojson   12-hourly perimeter snapshots
    lf_fireline_nrt.geojson    active fire line per snapshot
    lf_newfirepix_nrt.geojson  new VIIRS fire pixels per snapshot

External (fetched once, cached in <input>/external/):
    HIFLD Electric Power Transmission Lines (ArcGIS FeatureServer)
    OpenStreetMap power=substation via Overpass (HIFLD substations are no
    longer served publicly, see README/notes on the page)
    NIFC WFIGS Interagency Perimeters, YearToDate (for the honesty check)

Output (all small, meant to be committed to a GitHub Pages folder):
    steps.json          per-snapshot readouts + growth + rate of spread runs
    perimeters.json     simplified perimeter per snapshot (GeoJSON)
    firelines.json      simplified active fire line per snapshot (GeoJSON)
    newpix.json         new fire pixels per snapshot, [[lon,lat],...]
    arrival.png/json    first-burned grid as colour PNG + bounds + legend
    arrival_days.json   the same grid as integer days (for hover readout)
    exposure.json       segments + substations with distance/timing metrics
    nifc.json           simplified NIFC perimeter + area comparison
    inspection_queue.kml / .csv   top-N corridors for the Capture Planner
    meta.json           fire facts, sources, summary numbers

Everything is computed in a local equirectangular metre frame centred on the
fire (fine for a 60 km area: scale error < 0.1 %). Geometry via shapely 2.

Usage:
    python build_fire_data.py --input path/to/fire_cache --out path/to/site/fire-spread/data
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import collections
from collections import OrderedDict

import numpy as np
import shapely
from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                              MultiPolygon, Point, Polygon, box, mapping, shape)
from shapely.ops import nearest_points, unary_union
from PIL import Image

UA = "fire-spread-portfolio/1.0 (github.com/Emerlads/georef-splat-tools)"

FIRE = {
    "name": "Little Giant Fire",
    "state": "Washington",
    "feds_fireid": 63369,
    "feds_region": "CONUS",
    "nifc_name": "Little Giant",
}

HIFLD_TL = ("https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/"
            "Electric_Power_Transmission_Lines/FeatureServer/0")
NIFC_YTD = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
            "WFIGS_Interagency_Perimeters_YearToDate/FeatureServer/0")
OVERPASS = "https://overpass-api.de/api/interpreter"

MARGIN_KM = 10.0          # search margin around the fire for infrastructure
SEG_LEN_M = 1000.0        # target segment length
GRID_M = 250.0            # arrival grid cell size
CORRIDOR_M = 60.0         # buffer each side of a segment for the KML corridor
TOP_N = 30                # inspection queue export size
SIMPLIFY_PERIM_M = 60.0
SIMPLIFY_LINE_M = 40.0

# Sequential single-hue ramp (light -> dark) for arrival time. Kept warm so it
# reads as "fire" on satellite imagery; one hue, so lightness alone carries time.
RAMP = ["#fff1c8", "#ffd27a", "#f6a33c", "#e2711d", "#b84a14", "#7a2a0c", "#3d1206"]


# ----------------------------------------------------------------- utilities
def log(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def fetch_cached(url: str, path: str, data: bytes | None = None) -> bytes:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            return f.read()
    log("fetching", url[:100], "...")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(body)
    return body


class Local:
    """Equirectangular metres about (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = 111320.0 * math.cos(math.radians(lat0))
        self.ky = 110574.0

    def fwd(self, g):
        return shapely.transform(g, lambda c: np.c_[(c[:, 0] - self.lon0) * self.kx, (c[:, 1] - self.lat0) * self.ky])

    def inv(self, g):
        return shapely.transform(g, lambda c: np.c_[c[:, 0] / self.kx + self.lon0, c[:, 1] / self.ky + self.lat0])

    def inv_xy(self, x, y):
        return (x / self.kx + self.lon0, y / self.ky + self.lat0)


def round_geom(g, nd=5):
    return shapely.set_precision(g, 10 ** -nd) if False else shapely.transform(g, lambda c: np.round(c, nd))


def bearing_deg(x0, y0, x1, y1) -> float:
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0


def compass(b: float) -> str:
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((b + 11.25) // 22.5) % 16]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ramp_color(f: float):
    f = min(max(f, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(f), len(RAMP) - 2)
    a, b = hex_to_rgb(RAMP[i]), hex_to_rgb(RAMP[i + 1])
    t = f - i
    return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


# ------------------------------------------------------------------ loading
def load_feds(input_dir: str):
    def load(name):
        with open(os.path.join(input_dir, name), encoding="utf-8") as f:
            return json.load(f)["features"]

    per_raw = load("lf_perimeter_nrt.geojson")
    fl_raw = load("lf_fireline_nrt.geojson")
    px_raw = load("lf_newfirepix_nrt.geojson")

    # FEDS republishes the same state every 12 h while nothing new is detected:
    # `t` is the time of the last new detection, the primarykey carries the
    # snapshot time. Deduplicate on `t`, keep the earliest snapshot.
    def dedupe(feats):
        out: "OrderedDict[str, dict]" = OrderedDict()
        for f in sorted(feats, key=lambda f: f["properties"]["primarykey"]):
            t = f["properties"]["t"]
            if t not in out:
                out[t] = f
        return OrderedDict(sorted(out.items()))

    per = dedupe(per_raw)
    fl = dedupe(fl_raw)
    px = dedupe(px_raw)
    log(f"perimeters {len(per_raw)} -> {len(per)} unique snapshots; firelines {len(fl)}; newpix {len(px)}")
    return per, fl, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                                  "Emerlads.github.io", "fire-spread", "data"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    ext = os.path.join(args.input, "external")

    per, fl, px = load_feds(args.input)
    times = list(per.keys())
    t_dt = [parse_t(t) for t in times]
    t0 = t_dt[0]

    # Fire centroid -> local frame
    final_ll = shape(per[times[-1]]["geometry"])
    c = final_ll.centroid
    loc = Local(c.x, c.y)
    log(f"centroid {c.y:.4f} N {c.x:.4f} W; final FEDS area {per[times[-1]]['properties']['farea']:.1f} km2")

    perims_m = [shapely.make_valid(loc.fwd(shape(per[t]["geometry"]))) for t in times]
    union_m = unary_union(perims_m)
    minx, miny, maxx, maxy = union_m.bounds

    # ------------------------------------------------------------ step table
    steps = []
    prev_area = 0.0
    for i, t in enumerate(times):
        p = per[t]["properties"]
        dt_days = (t_dt[i] - t_dt[i - 1]).total_seconds() / 86400 if i else None
        area = float(p["farea"])
        steps.append({
            "i": i,
            "t": t + "Z",
            "day": round((t_dt[i] - t0).total_seconds() / 86400, 3),
            "dt_days": round(dt_days, 3) if dt_days else None,
            "area_km2": round(area, 2),
            "growth_km2": round(area - prev_area, 2),
            "growth_km2_day": round((area - prev_area) / dt_days, 2) if dt_days else None,
            "perim_km": round(float(p["fperim"]), 1),
            "fline_km": round(float(p["flinelen"] or 0), 1),
            "mean_frp_mw": round(float(p["meanfrp"]), 1) if p["meanfrp"] is not None else None,
            "n_pixels": int(p["n_pixels"]),
            "n_newpixels": int(p["n_newpixels"]),
            "duration_days": p["duration"],
        })
        prev_area = area

    # Growth per calendar day (UTC): area at the last snapshot of each day minus
    # the previous day's. This is what "peak growth day" means on the page.
    by_day: "OrderedDict[str, float]" = OrderedDict()
    for i, t in enumerate(times):
        by_day[t[:10]] = steps[i]["area_km2"]
    daily = []
    prev = 0.0
    prev_day = None
    for d, a in by_day.items():
        dd = dt.date.fromisoformat(d)
        ndays = (dd - prev_day).days if prev_day else 1
        daily.append({"date": d, "area_km2": a, "growth_km2": round(a - prev, 2),
                      "growth_km2_day": round((a - prev) / max(ndays, 1), 2), "span_days": ndays})
        prev, prev_day = a, dd
    peak = max(daily, key=lambda r: r["growth_km2"])
    log(f"peak growth day {peak['date']}: +{peak['growth_km2']} km2")

    # ---------------------------------------------------------- rate of spread
    # For each step, the perimeter vertex farthest outside the previous
    # perimeter; its distance / elapsed time is the max outward advance.
    runs = []
    for i in range(1, len(times)):
        prev_g, cur_g = perims_m[i - 1], perims_m[i]
        if cur_g.is_empty or prev_g.is_empty:
            continue
        polys = list(cur_g.geoms) if isinstance(cur_g, MultiPolygon) else [cur_g]
        best = None
        for poly in polys:
            detached = not poly.intersects(prev_g)
            xy = np.asarray(poly.exterior.coords)
            pts = shapely.points(xy)
            d = shapely.distance(pts, prev_g)
            k = int(np.argmax(d))
            if best is None or d[k] > best[0]:
                best = (float(d[k]), xy[k], detached)
        dist, (x1, y1), detached = best
        if dist < 200:        # below one VIIRS pixel: not a run, just jitter
            continue
        near = nearest_points(prev_g, Point(x1, y1))[0]
        dtd = steps[i]["dt_days"]
        b = bearing_deg(near.x, near.y, x1, y1)
        lon0, lat0 = loc.inv_xy(near.x, near.y)
        lon1, lat1 = loc.inv_xy(x1, y1)
        runs.append({
            "i": i, "t": times[i] + "Z", "dt_days": dtd,
            "advance_m": round(dist), "rate_m_day": round(dist / dtd),
            "bearing_deg": round(b), "compass": compass(b), "detached": bool(detached),
            "from": [round(lon0, 5), round(lat0, 5)], "to": [round(lon1, 5), round(lat1, 5)],
        })
    runs_sorted = sorted(runs, key=lambda r: -r["rate_m_day"])
    log("top runs:", [(r["t"][:13], r["rate_m_day"], r["compass"], r["detached"]) for r in runs_sorted[:5]])

    # --------------------------------------------------- shipped geometries
    def simplify_ll(g_m, tol):
        s = g_m.simplify(tol, preserve_topology=True)
        return round_geom(loc.inv(s), 5)

    perim_fc = {"type": "FeatureCollection", "features": []}
    for i, t in enumerate(times):
        perim_fc["features"].append({"type": "Feature", "properties": {"i": i},
                                     "geometry": mapping(simplify_ll(perims_m[i], SIMPLIFY_PERIM_M))})

    fl_fc = {"type": "FeatureCollection", "features": []}
    fl_index = {t: j for j, t in enumerate(fl.keys())}
    for i, t in enumerate(times):
        if t in fl:
            g = loc.fwd(shape(fl[t]["geometry"]))
            fl_fc["features"].append({"type": "Feature", "properties": {"i": i},
                                      "geometry": mapping(simplify_ll(g, SIMPLIFY_LINE_M))})

    newpix = {}
    for i, t in enumerate(times):
        if t in px:
            g = shape(px[t]["geometry"])
            pts = [g] if isinstance(g, Point) else list(g.geoms)
            newpix[str(i)] = [[round(p.x, 4), round(p.y, 4)] for p in pts]

    # --------------------------------------------------------- arrival grid
    pad = 2 * GRID_M
    gx0, gy0, gx1, gy1 = minx - pad, miny - pad, maxx + pad, maxy + pad
    nx, ny = int(math.ceil((gx1 - gx0) / GRID_M)), int(math.ceil((gy1 - gy0) / GRID_M))
    xs = gx0 + (np.arange(nx) + 0.5) * GRID_M
    ys = gy1 - (np.arange(ny) + 0.5) * GRID_M          # row 0 = north
    X, Y = np.meshgrid(xs, ys)
    arrival = np.full((ny, nx), -1, dtype=np.int16)    # step index of first containment
    flat_x, flat_y = X.ravel(), Y.ravel()
    unassigned = np.ones(flat_x.size, dtype=bool)
    for i, g in enumerate(perims_m):
        idx = np.flatnonzero(unassigned)
        if idx.size == 0:
            break
        inside = shapely.contains_xy(g, flat_x[idx], flat_y[idx])
        hit = idx[inside]
        arrival.ravel()[hit] = i
        unassigned[hit] = False
    log(f"arrival grid {nx}x{ny} at {GRID_M:.0f} m, burned cells {(arrival >= 0).sum()}")

    day_of_step = np.array([s["day"] for s in steps])
    arr_days = np.where(arrival >= 0, day_of_step[np.clip(arrival, 0, None)], -1.0)
    dmax = float(day_of_step[-1])
    # Legend breaks: most of the growth is in the first weeks, so use the days
    # of actual growth rather than a linear 0..dmax stretch. Map colour by the
    # rank of the arrival day among burned cells (quantile stretch), and ship
    # the breakpoints so the legend is honest about it.
    burned_days = arr_days[arr_days >= 0]
    qs = np.quantile(burned_days, np.linspace(0, 1, len(RAMP)))
    qs[0], qs[-1] = 0.0, dmax

    def day_to_f(d):
        return float(np.interp(d, qs, np.linspace(0, 1, len(RAMP))))

    img = np.zeros((ny, nx, 4), dtype=np.uint8)
    for r in range(ny):
        for cidx in range(nx):
            d = arr_days[r, cidx]
            if d >= 0:
                img[r, cidx, :3] = ramp_color(day_to_f(d))
                img[r, cidx, 3] = 205
    Image.fromarray(img, "RGBA").save(os.path.join(out, "arrival.png"), optimize=True)
    west, south = loc.inv_xy(gx0, gy0)
    east, north = loc.inv_xy(gx1, gy1)
    legend = []
    for q in qs:
        di = int(round(q))
        legend.append({"day": round(float(q), 1), "date": (t0 + dt.timedelta(days=float(q))).strftime("%Y-%m-%d"),
                       "color": "#%02x%02x%02x" % ramp_color(day_to_f(q))})
    with open(os.path.join(out, "arrival.json"), "w") as f:
        json.dump({"west": round(west, 6), "south": round(south, 6), "east": round(east, 6), "north": round(north, 6),
                   "nx": nx, "ny": ny, "cell_m": GRID_M, "ramp": RAMP, "stops": legend,
                   "stretch": "quantile over burned cells", "t0": times[0] + "Z"}, f)
    with open(os.path.join(out, "arrival_days.json"), "w") as f:
        # tenths of a day, -1 = never burned; row-major from the north-west corner
        json.dump({"nx": nx, "ny": ny, "scale": 0.1,
                   "v": [int(round(v * 10)) if v >= 0 else -1 for v in arr_days.ravel().tolist()]}, f,
                  separators=(",", ":"))

    # ----------------------------------------------------- infrastructure
    mk = MARGIN_KM * 1000
    bb_m = box(minx - mk, miny - mk, maxx + mk, maxy + mk)
    bb_ll = loc.inv(bb_m).bounds   # west, south, east, north

    q = urllib.parse.urlencode({
        "where": "1=1", "geometry": ",".join(f"{v:.5f}" for v in bb_ll), "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": "4326", "f": "geojson"})
    tl = json.loads(fetch_cached(f"{HIFLD_TL}/query?{q}", os.path.join(ext, "hifld_transmission_lines.geojson")))
    if tl.get("properties", {}).get("exceededTransferLimit"):
        log("WARNING: HIFLD transfer limit exceeded; paging not implemented")
    log(f"HIFLD transmission lines in bbox: {len(tl['features'])}")

    ov = ("[out:json][timeout:60];("
          f"node[\"power\"=\"substation\"]({bb_ll[1]:.4f},{bb_ll[0]:.4f},{bb_ll[3]:.4f},{bb_ll[2]:.4f});"
          f"way[\"power\"=\"substation\"]({bb_ll[1]:.4f},{bb_ll[0]:.4f},{bb_ll[3]:.4f},{bb_ll[2]:.4f});"
          f"relation[\"power\"=\"substation\"]({bb_ll[1]:.4f},{bb_ll[0]:.4f},{bb_ll[3]:.4f},{bb_ll[2]:.4f});"
          f"way[\"power\"=\"line\"]({bb_ll[1]:.4f},{bb_ll[0]:.4f},{bb_ll[3]:.4f},{bb_ll[2]:.4f});"
          f"way[\"power\"=\"minor_line\"]({bb_ll[1]:.4f},{bb_ll[0]:.4f},{bb_ll[3]:.4f},{bb_ll[2]:.4f});"
          ");out tags center;")
    osm = json.loads(fetch_cached(OVERPASS, os.path.join(ext, "osm_power_overpass.json"),
                                  data=urllib.parse.urlencode({"data": ov}).encode()))
    subs_raw = [e for e in osm["elements"] if e.get("tags", {}).get("power") == "substation"]
    log(f"OSM substations in bbox: {len(subs_raw)}")

    final_m = perims_m[-1]

    def timing(geom_m):
        """first step at which the perimeter is within 0 / 1000 / 5000 m, and min distance."""
        first = {0: None, 1000: None, 5000: None, 10000: None}
        dmin, dmin_i = float("inf"), None
        series = []
        for i, g in enumerate(perims_m):
            d = geom_m.distance(g)
            series.append(int(round(d / 10.0)) * 10)
            if d < dmin:
                dmin, dmin_i = d, i
            for thr in first:
                if first[thr] is None and d <= thr:
                    first[thr] = i
        return first, dmin, dmin_i, series

    def step_t(i):
        return None if i is None else times[i] + "Z"

    segments = []
    seg_geoms_ll = []
    for f in tl["features"]:
        p = f["properties"]
        g_ll = shape(f["geometry"])
        g_m = loc.fwd(g_ll).intersection(bb_m)
        parts = list(g_m.geoms) if isinstance(g_m, MultiLineString) else ([g_m] if not g_m.is_empty else [])
        k = 0
        for part in parts:
            L = part.length
            n = max(1, int(round(L / SEG_LEN_M)))
            for j in range(n):
                seg = shapely.ops.substring(part, j * L / n, (j + 1) * L / n)
                if seg.length < 1:
                    continue
                k += 1
                first, dmin, dmin_i, series = timing(seg)
                inside_len = seg.intersection(final_m).length
                sid = f"L{p['ID']}-{k:02d}"
                segments.append({
                    "id": sid, "line_id": str(p["ID"]), "owner": (p.get("OWNER") or "").title(),
                    "kv": p.get("VOLTAGE"), "type": p.get("TYPE"), "from": p.get("SUB_1"), "to": p.get("SUB_2"),
                    "len_m": round(seg.length),
                    "first_0m": step_t(first[0]), "first_1km": step_t(first[1000]), "first_5km": step_t(first[5000]),
                    "first_10km": step_t(first[10000]),
                    "min_dist_m": round(dmin), "min_dist_t": step_t(dmin_i),
                    "inside_frac": round(inside_len / seg.length, 2), "ever_inside": bool(first[0] is not None),
                    "dist_series_m": series,
                })
                seg_geoms_ll.append(round_geom(loc.inv(seg), 5))
    log(f"segments: {len(segments)}")

    substations = []
    sub_pts_ll = []
    for e in subs_raw:
        tags = e.get("tags", {})
        lat, lon = (e["center"]["lat"], e["center"]["lon"]) if "center" in e else (e["lat"], e["lon"])
        pt_m = loc.fwd(Point(lon, lat))
        first, dmin, dmin_i, series = timing(pt_m)
        substations.append({
            "id": f"osm-{e['type'][0]}{e['id']}", "name": tags.get("name") or "(unnamed)",
            "operator": tags.get("operator"), "kv": tags.get("voltage"), "kind": tags.get("substation"),
            "first_0m": step_t(first[0]), "first_1km": step_t(first[1000]), "first_5km": step_t(first[5000]),
            "first_10km": step_t(first[10000]),
            "min_dist_m": round(dmin), "min_dist_t": step_t(dmin_i), "ever_inside": bool(first[0] is not None),
            "dist_series_m": series,
            "lon": round(lon, 5), "lat": round(lat, 5),
            "osm": f"https://www.openstreetmap.org/{e['type']}/{e['id']}",
        })

    def exposure_class(r):
        if r["first_0m"]:
            return 0
        if r["first_1km"]:
            return 1
        if r["first_5km"]:
            return 2
        if r["first_10km"]:
            return 3
        return 4

    def rank_key(r):
        cls = exposure_class(r)
        tkey = r["first_0m"] or r["first_1km"] or r["first_5km"] or r["first_10km"] or "9999"
        return (cls, tkey, r["min_dist_m"])

    for lst in (segments, substations):
        for r in lst:
            r["class"] = exposure_class(r)
    ranked = sorted(segments, key=rank_key)
    for n, r in enumerate(ranked):
        r["rank"] = n + 1 if r["class"] < 4 else None

    exposure_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": s, "geometry": mapping(g)} for s, g in zip(segments, seg_geoms_ll)]}
    with open(os.path.join(out, "exposure.json"), "w") as f:
        json.dump({"segments": exposure_fc, "substations": substations,
                   "classes": ["inside the burn", "within 1 km", "within 5 km", "within 10 km", "beyond 10 km"],
                   "bbox": [round(v, 5) for v in bb_ll]}, f, separators=(",", ":"))

    n_in = sum(1 for s in segments if s["class"] == 0)
    n_1k = sum(1 for s in segments if s["class"] == 1)
    n_5k = sum(1 for s in segments if s["class"] == 2)
    n_10k = sum(1 for s in segments if s["class"] == 3)
    closest = min(segments, key=lambda s: s["min_dist_m"]) if segments else None
    log(f"segments inside {n_in}, within 1 km {n_1k}, within 5 km {n_5k}, within 10 km {n_10k}; "
        f"closest {closest['id'] if closest else None} {closest['min_dist_m'] if closest else None} m; "
        f"substations by class {[sum(1 for s in substations if s['class']==k) for k in range(5)]}")

    # ----------------------------------------------------- inspection queue
    queue = [r for r in ranked if r["class"] < 4][:TOP_N]
    seg_by_id = {s["id"]: g for s, g in zip(segments, seg_geoms_ll)}
    kml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f'<name>Little Giant Fire: post-fire line inspection corridors (top {len(queue)})</name>',
           '<description>Screening product from NASA FEDS perimeters and HIFLD transmission lines. '
           f'Each polygon is a {CORRIDOR_M:.0f} m corridor each side of a ~1 km line segment. Not operational guidance.</description>',
           '<Style id="c0"><LineStyle><color>ff1a1aaa</color><width>2</width></LineStyle><PolyStyle><color>551a1aaa</color></PolyStyle></Style>',
           '<Style id="c1"><LineStyle><color>ff1d71e2</color><width>2</width></LineStyle><PolyStyle><color>551d71e2</color></PolyStyle></Style>',
           '<Style id="c2"><LineStyle><color>ff3ca3f6</color><width>2</width></LineStyle><PolyStyle><color>553ca3f6</color></PolyStyle></Style>',
           '<Style id="c3"><LineStyle><color>ff7ad2ff</color><width>2</width></LineStyle><PolyStyle><color>557ad2ff</color></PolyStyle></Style>']
    csv_rows = []
    for r in queue:
        g_m = loc.fwd(seg_by_id[r["id"]])
        corridor = loc.inv(g_m.buffer(CORRIDOR_M, cap_style="flat", join_style="round").simplify(2))
        if isinstance(corridor, MultiPolygon):
            corridor = max(corridor.geoms, key=lambda p: p.area)
        coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in corridor.exterior.coords)
        cname = ["inside burn", "within 1 km", "within 5 km", "within 10 km"][r["class"]]
        desc = (f"{r['owner']} {r['kv']} kV, {r['from']} to {r['to']}. {cname}. "
                f"First within 10 km {r['first_10km'] or '-'}; within 5 km {r['first_5km'] or '-'}; within 1 km {r['first_1km'] or '-'}; reached {r['first_0m'] or '-'}. "
                f"Min distance {r['min_dist_m']} m. Segment length {r['len_m']} m.")
        kml.append(f'<Placemark><name>#{r["rank"]} {r["id"]} ({cname})</name><styleUrl>#c{r["class"]}</styleUrl>'
                   f'<description>{desc}</description>'
                   f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
        cen = seg_by_id[r["id"]].centroid
        csv_rows.append([r["rank"], r["id"], cname, r["owner"], r["kv"], r["from"], r["to"], r["len_m"],
                         r["first_10km"] or "", r["first_5km"] or "", r["first_1km"] or "", r["first_0m"] or "", r["min_dist_m"],
                         r["inside_frac"], f"{cen.y:.5f}", f"{cen.x:.5f}"])
    kml.append("</Document></kml>")
    with open(os.path.join(out, "little-giant_inspection_corridors.kml"), "w", encoding="utf-8") as f:
        f.write("\n".join(kml))
    with open(os.path.join(out, "little-giant_inspection_queue.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "segment_id", "exposure_class", "owner", "kv", "from_sub", "to_sub", "segment_len_m",
                    "first_within_10km_utc", "first_within_5km_utc", "first_within_1km_utc", "first_reached_utc", "min_dist_m",
                    "inside_fraction", "centroid_lat", "centroid_lon"])
        w.writerows(csv_rows)

    # ------------------------------------------------------- NIFC comparison
    q = urllib.parse.urlencode({
        "where": f"poly_IncidentName LIKE '%{FIRE['nifc_name']}%'",
        "outFields": "poly_IncidentName,poly_GISAcres,poly_DateCurrent,poly_PolygonDateTime,attr_IncidentSize,"
                     "attr_FireDiscoveryDateTime,attr_ContainmentDateTime,attr_UniqueFireIdentifier,poly_MapMethod,"
                     "poly_Source,attr_POOState",
        "outSR": "4326", "f": "geojson"})
    nifc = None
    try:
        nj = json.loads(fetch_cached(f"{NIFC_YTD}/query?{q}", os.path.join(ext, "nifc_wfigs_little_giant.geojson")))
        feats = [f for f in nj.get("features", []) if f["properties"].get("attr_POOState") == "US-WA"]
        if feats:
            f0 = max(feats, key=lambda f: f["properties"].get("poly_GISAcres") or 0)
            ng_m = shapely.make_valid(loc.fwd(shape(f0["geometry"])))
            inter = ng_m.intersection(final_m).area
            union = ng_m.union(final_m).area
            ms = lambda v: dt.datetime.fromtimestamp(v / 1000, dt.timezone.utc).strftime("%Y-%m-%d") if v else None
            nifc = {
                "name": f0["properties"]["poly_IncidentName"], "unique_id": f0["properties"].get("attr_UniqueFireIdentifier"),
                "gis_acres": f0["properties"].get("poly_GISAcres"),
                "area_km2": round(ng_m.area / 1e6, 1), "feds_final_km2": round(final_m.area / 1e6, 1),
                "feds_reported_km2": steps[-1]["area_km2"],
                "intersection_km2": round(inter / 1e6, 1), "union_km2": round(union / 1e6, 1), "iou": round(inter / union, 3),
                "feds_only_km2": round((final_m.area - inter) / 1e6, 1), "nifc_only_km2": round((ng_m.area - inter) / 1e6, 1),
                "polygon_date": ms(f0["properties"].get("poly_PolygonDateTime")),
                "discovery_date": ms(f0["properties"].get("attr_FireDiscoveryDateTime")),
                "containment_date": ms(f0["properties"].get("attr_ContainmentDateTime")),
                "map_method": f0["properties"].get("poly_MapMethod"), "source": f0["properties"].get("poly_Source"),
                "geometry": mapping(round_geom(loc.inv(ng_m.simplify(SIMPLIFY_PERIM_M, preserve_topology=True)), 5)),
            }
            log(f"NIFC {nifc['gis_acres']} ac = {nifc['area_km2']} km2 vs FEDS {nifc['feds_final_km2']} km2, IoU {nifc['iou']}")
    except Exception as e:  # keep the build going if NIFC is down
        log("NIFC fetch failed:", e)
    with open(os.path.join(out, "nifc.json"), "w") as f:
        json.dump(nifc, f, separators=(",", ":"))
    nifc_m = ng_m if nifc else None

    # ------------------------------------------------------------ buildings
    # OSM building footprints as centroids, fetched in four quadrants because a
    # single bbox query times out on the public Overpass instance.
    bpath = os.path.join(ext, "osm_buildings_overpass.json")
    if not os.path.exists(bpath):
        s_, w_, n_, e_ = bb_ll[1], bb_ll[0], bb_ll[3], bb_ll[2]
        mlat, mlon = (s_ + n_) / 2, (w_ + e_) / 2
        els, seen = [], set()
        for k, (a, b2, c2, d2) in enumerate([(s_, w_, mlat, mlon), (s_, mlon, mlat, e_), (mlat, w_, n_, mlon), (mlat, mlon, n_, e_)]):
            qq = (f"[out:json][timeout:120];(way[\"building\"]({a:.5f},{b2:.5f},{c2:.5f},{d2:.5f});"
                  f"relation[\"building\"]({a:.5f},{b2:.5f},{c2:.5f},{d2:.5f}););out center tags;")
            part = json.loads(fetch_cached(OVERPASS, os.path.join(ext, f"osm_buildings_q{k+1}.json"),
                                           data=urllib.parse.urlencode({"data": qq}).encode()))
            for e in part["elements"]:
                if e["id"] not in seen:
                    seen.add(e["id"]); els.append(e)
        with open(bpath, "w") as f:
            json.dump({"elements": els}, f)
    with open(bpath, encoding="utf-8") as f:
        bl = json.load(f)["elements"]
    b_lon = np.array([(e.get("center") or e)["lon"] for e in bl])
    b_lat = np.array([(e.get("center") or e)["lat"] for e in bl])
    b_x = (b_lon - loc.lon0) * loc.kx
    b_y = (b_lat - loc.lat0) * loc.ky
    nb = b_lon.size
    log(f"OSM buildings: {nb}")

    # First step inside / within 1 km / within 5 km. Buffers are built on a
    # lightly simplified perimeter (30 m, far below the buffer distance).
    first_in = np.full(nb, -1, dtype=np.int16)
    first_1k = np.full(nb, -1, dtype=np.int16)
    first_5k = np.full(nb, -1, dtype=np.int16)
    for i, g in enumerate(perims_m):
        gs = g.simplify(30, preserve_topology=True)
        for arr_, geom in ((first_in, gs), (first_1k, gs.buffer(1000, quad_segs=4)), (first_5k, gs.buffer(5000, quad_segs=4))):
            idx = np.flatnonzero(arr_ < 0)
            if idx.size == 0:
                continue
            hit = idx[shapely.contains_xy(geom, b_x[idx], b_y[idx])]
            arr_[hit] = i
    # distance to the union of all perimeters == min over steps of the distance
    b_pts = shapely.points(np.c_[b_x, b_y])
    b_dmin = shapely.distance(b_pts, union_m)
    # same against the NIFC final perimeter (0 inside, 1 within 1 km, 2 within 5 km, 3 beyond)
    b_nifc = np.full(nb, 3, dtype=np.int8)
    if nifc_m is not None:
        ns = nifc_m.simplify(30, preserve_topology=True)
        b_nifc[shapely.contains_xy(ns.buffer(5000, quad_segs=4), b_x, b_y)] = 2
        b_nifc[shapely.contains_xy(ns.buffer(1000, quad_segs=4), b_x, b_y)] = 1
        b_nifc[shapely.contains_xy(ns, b_x, b_y)] = 0
    cum_in = [int((first_in[(first_in >= 0)] <= i).sum()) for i in range(len(times))]
    cum_1k = [int((first_1k[(first_1k >= 0)] <= i).sum()) for i in range(len(times))]
    cum_5k = [int((first_5k[(first_5k >= 0)] <= i).sum()) for i in range(len(times))]
    bsum = {
        "n": int(nb), "source": "OpenStreetMap building footprints (centroids) via Overpass",
        "feds": {"inside": cum_in[-1], "within_1km": cum_1k[-1], "within_5km": cum_5k[-1]},
        "nifc": {"inside": int((b_nifc == 0).sum()), "within_1km": int((b_nifc <= 1).sum()), "within_5km": int((b_nifc <= 2).sum())} if nifc_m is not None else None,
        "first_inside_t": step_t(int(first_in[first_in >= 0].min())) if (first_in >= 0).any() else None,
        "tags": dict(collections.Counter(e.get("tags", {}).get("building", "yes") for e in bl).most_common(6)),
    }
    log(f"buildings FEDS inside {bsum['feds']['inside']}, <=1km {bsum['feds']['within_1km']}, <=5km {bsum['feds']['within_5km']}; NIFC {bsum['nifc']}")

    # 500 m clusters for the inspection queue
    CELL = 500.0
    cx0, cy0 = math.floor(b_x.min() / CELL) * CELL, math.floor(b_y.min() / CELL) * CELL
    cix = ((b_x - cx0) // CELL).astype(int); ciy = ((b_y - cy0) // CELL).astype(int)
    cells = {}
    for k in range(nb):
        key = (int(cix[k]), int(ciy[k]))
        c_ = cells.setdefault(key, {"n": 0, "inside": 0, "k1": 0, "nifc_in": 0, "t_in": None, "t_1k": None})
        c_["n"] += 1
        if first_in[k] >= 0:
            c_["inside"] += 1; c_["t_in"] = int(first_in[k]) if c_["t_in"] is None else min(c_["t_in"], int(first_in[k]))
        if first_1k[k] >= 0:
            c_["k1"] += 1; c_["t_1k"] = int(first_1k[k]) if c_["t_1k"] is None else min(c_["t_1k"], int(first_1k[k]))
        if b_nifc[k] == 0:
            c_["nifc_in"] += 1
    ranked_cells = sorted([(k, v) for k, v in cells.items() if v["inside"] + v["k1"] > 0],
                          key=lambda kv: (-kv[1]["inside"], -kv[1]["k1"], kv[1]["t_in"] if kv[1]["t_in"] is not None else 9999, kv[1]["t_1k"] if kv[1]["t_1k"] is not None else 9999))
    clusters = []
    for rank, ((ix, iy), v) in enumerate(ranked_cells[:20], 1):
        x0, y0 = cx0 + ix * CELL, cy0 + iy * CELL
        poly_ll = loc.inv(box(x0, y0, x0 + CELL, y0 + CELL))
        w_, s_, e_, n_ = poly_ll.bounds
        clusters.append({"rank": rank, "id": f"C{ix:03d}-{iy:03d}", "n": v["n"], "inside": v["inside"], "within_1km": v["k1"],
                         "nifc_inside": v["nifc_in"], "first_inside": step_t(v["t_in"]), "first_1km": step_t(v["t_1k"]),
                         "bounds": [round(w_, 5), round(s_, 5), round(e_, 5), round(n_, 5)]})
    kmlc = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
            f'<name>Little Giant Fire: building clusters for post-fire inspection (top {len(clusters)}, {CELL:.0f} m cells)</name>',
            '<description>Screening product from NASA FEDS perimeters and OpenStreetMap building centroids. Ranked by buildings inside the satellite perimeter, then within 1 km. Not operational guidance; inside the satellite perimeter does not mean damaged.</description>',
            '<Style id="cl"><LineStyle><color>ff1a1aaa</color><width>2</width></LineStyle><PolyStyle><color>401a1aaa</color></PolyStyle></Style>']
    for c_ in clusters:
        w_, s_, e_, n_ = c_["bounds"]
        coords = f"{w_},{s_},0 {e_},{s_},0 {e_},{n_},0 {w_},{n_},0 {w_},{s_},0"
        kmlc.append(f'<Placemark><name>#{c_["rank"]} {c_["id"]}: {c_["inside"]} inside, {c_["within_1km"]} within 1 km</name><styleUrl>#cl</styleUrl>'
                    f'<description>{c_["n"]} OSM buildings in cell. Inside FEDS perimeter: {c_["inside"]} (first {c_["first_inside"] or "-"}). Within 1 km: {c_["within_1km"]} (first {c_["first_1km"] or "-"}). Inside NIFC final perimeter: {c_["nifc_inside"]}.</description>'
                    f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
    kmlc.append("</Document></kml>")
    with open(os.path.join(out, "little-giant_building_clusters.kml"), "w", encoding="utf-8") as f:
        f.write("\n".join(kmlc))
    with open(os.path.join(out, "little-giant_building_clusters.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "cell_id", "buildings_in_cell", "inside_feds_perimeter", "within_1km_feds", "inside_nifc_perimeter",
                    "first_inside_utc", "first_within_1km_utc", "west", "south", "east", "north"])
        for c_ in clusters:
            w.writerow([c_["rank"], c_["id"], c_["n"], c_["inside"], c_["within_1km"], c_["nifc_inside"], c_["first_inside"] or "", c_["first_1km"] or ""] + c_["bounds"])
    with open(os.path.join(out, "buildings.json"), "w") as f:
        json.dump({"n": int(nb), "lon": [round(float(v), 4) for v in b_lon], "lat": [round(float(v), 4) for v in b_lat],
                   "i0": first_in.tolist(), "i1": first_1k.tolist(), "i5": first_5k.tolist(),
                   "dmin": [int(v) for v in np.round(b_dmin / 10) * 10], "nifc": b_nifc.tolist(),
                   "cum": {"inside": cum_in, "within_1km": cum_1k, "within_5km": cum_5k},
                   "clusters": clusters, "cell_m": CELL}, f, separators=(",", ":"))

    # ------------------------------------------------------------- writes
    with open(os.path.join(out, "steps.json"), "w") as f:
        json.dump({"steps": steps, "daily": daily, "runs": runs}, f, separators=(",", ":"))
    with open(os.path.join(out, "perimeters.json"), "w") as f:
        json.dump(perim_fc, f, separators=(",", ":"))
    with open(os.path.join(out, "firelines.json"), "w") as f:
        json.dump(fl_fc, f, separators=(",", ":"))
    with open(os.path.join(out, "newpix.json"), "w") as f:
        json.dump(newpix, f, separators=(",", ":"))

    meta = {
        "fire": FIRE,
        "centroid": [round(c.x, 4), round(c.y, 4)],
        "bounds": [round(v, 5) for v in loc.inv(union_m).bounds],
        "t_first": times[0] + "Z", "t_last": times[-1] + "Z", "n_snapshots": len(times),
        "n_raw_snapshots": 182,
        "final_area_km2": steps[-1]["area_km2"], "final_perim_km": steps[-1]["perim_km"],
        "n_pixels_total": steps[-1]["n_pixels"],
        "peak_growth": peak,
        "max_run": runs_sorted[0] if runs_sorted else None,
        "top_runs": runs_sorted[:5],
        "exposure": {"segments": len(segments), "lines": len(tl["features"]),
                     "inside": n_in, "within_1km": n_1k, "within_5km": n_5k, "within_10km": n_10k,
                     "closest": {k: closest[k] for k in ("id", "owner", "kv", "min_dist_m", "min_dist_t")} if closest else None,
                     "substations": len(substations),
                     "substations_by_class": [sum(1 for s in substations if s["class"] == k) for k in range(5)],
                     "closest_substation": min(substations, key=lambda s: s["min_dist_m"])["name"] if substations else None,
                     "closest_substation_m": min(s["min_dist_m"] for s in substations) if substations else None,
                     "queue_n": len(queue), "corridor_m": CORRIDOR_M, "segment_m": SEG_LEN_M, "margin_km": MARGIN_KM},
        "nifc": {k: v for k, v in (nifc or {}).items() if k != "geometry"} if nifc else None,
        "buildings": bsum, "clusters_n": len(clusters), "cluster_cell_m": CELL,
        "grid_m": GRID_M, "simplify_m": SIMPLIFY_PERIM_M,
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "sources": {
            "feds": "https://openveda.cloud/api/features/collections/public.eis_fire_lf_perimeter_nrt",
            "feds_paper": "https://doi.org/10.1038/s41597-022-01343-0",
            "hifld_tl": HIFLD_TL,
            "osm_overpass": OVERPASS,
            "nifc": NIFC_YTD,
        },
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    total = sum(os.path.getsize(os.path.join(out, fn)) for fn in os.listdir(out))
    log(f"wrote {out}: {total/1e6:.2f} MB total")
    for fn in sorted(os.listdir(out)):
        log(f"  {fn:40s} {os.path.getsize(os.path.join(out, fn))/1e3:8.1f} kB")


if __name__ == "__main__":
    main()
