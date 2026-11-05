"""WGS84 geodesy: geodetic <-> ECEF <-> local East-North-Up.

Heights here are always *ellipsoidal* (height above the WGS84 ellipsoid).
Orthometric heights (above mean sea level / a geoid) must be converted with
``h = H + N`` first, where ``N`` is the geoid undulation. Mixing the two is
the classic way to put a model tens of metres above or below the terrain.
"""

from __future__ import annotations

import math

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B = WGS84_A * (1.0 - WGS84_F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + h) * cos_lat * math.cos(lon),
            (n + h) * cos_lat * math.sin(lon),
            (n * (1.0 - WGS84_E2) + h) * sin_lat,
        ],
        dtype=np.float64,
    )


def ecef_to_geodetic(xyz) -> tuple[float, float, float]:
    """Inverse of :func:`geodetic_to_ecef` (Bowring start + Newton refinement)."""
    x, y, z = (float(v) for v in xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    ep2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    lat = math.atan2(
        z + ep2 * WGS84_B * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    for _ in range(5):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n if abs(lat) < math.radians(89) else z / math.sin(lat) - n * (1 - WGS84_E2)
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n if abs(lat) < math.radians(89) else z / math.sin(lat) - n * (1 - WGS84_E2)
    return math.degrees(lat), math.degrees(lon), h


def ecef_to_enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """3x3 rotation taking ECEF difference vectors into local ENU at (lat, lon)."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(lat: float, lon: float, h: float, origin: tuple[float, float, float]) -> np.ndarray:
    o_lat, o_lon, o_h = origin
    d = geodetic_to_ecef(lat, lon, h) - geodetic_to_ecef(o_lat, o_lon, o_h)
    return ecef_to_enu_rotation(o_lat, o_lon) @ d


def enu_to_geodetic(enu, origin: tuple[float, float, float]) -> tuple[float, float, float]:
    o_lat, o_lon, o_h = origin
    ecef = geodetic_to_ecef(o_lat, o_lon, o_h) + ecef_to_enu_rotation(o_lat, o_lon).T @ np.asarray(enu, dtype=np.float64)
    return ecef_to_geodetic(ecef)


def enu_to_ecef_matrix(origin: tuple[float, float, float]) -> np.ndarray:
    """4x4 matrix mapping local ENU metres at ``origin`` to ECEF (row-major numpy).

    Equivalent to ``Cesium.Transforms.eastNorthUpToFixedFrame``. For a 3D Tiles
    ``transform`` use ``m.T.flatten()`` (column-major).
    """
    o_lat, o_lon, o_h = origin
    m = np.eye(4)
    m[:3, :3] = ecef_to_enu_rotation(o_lat, o_lon).T
    m[:3, 3] = geodetic_to_ecef(o_lat, o_lon, o_h)
    return m
