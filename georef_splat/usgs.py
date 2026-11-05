"""US public reference data: USGS 3DEP bare-earth elevation and NOAA GEOID18."""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request

import numpy as np

from .geodesy import enu_to_geodetic

DEP_EXPORT = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
NGS_GEOID = "https://geodesy.noaa.gov/api/geoid/ght"


def _get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "georef-splat"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def geoid18(lat: float, lon: float) -> float:
    """GEOID18 undulation N (m) from NOAA NGS; ellipsoidal h = orthometric H + N."""
    return float(json.loads(_get(f"{NGS_GEOID}?lat={lat:.6f}&lon={lon:.6f}&model=14"))["geoidHeight"])


def dep_bare_earth(west: float, south: float, east: float, north: float, width: int, height: int) -> np.ndarray:
    """3DEP elevation (NAVD88 metres), north-up array of shape (height, width)."""
    from PIL import Image

    q = urllib.parse.urlencode({
        "bbox": f"{west},{south},{east},{north}", "bboxSR": 4326, "imageSR": 4326,
        "size": f"{width},{height}", "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    })
    return np.asarray(Image.open(io.BytesIO(_get(f"{DEP_EXPORT}?{q}"))), dtype=np.float64)


def dep_ground_fn(origin: tuple[float, float, float], offset_m: float = 0.0, n_geoid: float | None = None):
    """A ``ground_fn`` for :func:`zones.rasterise` that samples 3DEP in ENU heights.

    ``offset_m`` is the measured vertical bias of the reconstruction relative to
    3DEP (reconstruction minus reference), so ground lands in the splat's frame.
    """
    n = geoid18(origin[0], origin[1]) if n_geoid is None else n_geoid

    def fn(x0, y0, nx, ny, cell):
        sw = enu_to_geodetic([x0, y0, 0.0], origin)
        ne = enu_to_geodetic([x0 + nx * cell, y0 + ny * cell, 0.0], origin)
        dem = dep_bare_earth(sw[1], sw[0], ne[1], ne[0], nx, ny)[::-1]  # south-up like the grid
        return dem + n - origin[2] + offset_m

    return fn
