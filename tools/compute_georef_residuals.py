#!/usr/bin/env python3
"""Compute camera-center residuals between an ENU COLMAP model and EXIF GPS priors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image
import pycolmap


Image.MAX_IMAGE_PIXELS = None
GPS_TAGS = ExifTags.GPSTAGS


def rational_to_float(value) -> float:
    try:
        return float(value)
    except TypeError:
        return float(value[0]) / float(value[1])


def dms_to_degrees(values, ref: str) -> float:
    deg = rational_to_float(values[0])
    minute = rational_to_float(values[1])
    second = rational_to_float(values[2])
    sign = -1.0 if ref in ("S", "W") else 1.0
    return sign * (deg + minute / 60.0 + second / 3600.0)


def read_gps(path: Path) -> tuple[float, float, float] | None:
    with Image.open(path) as im:
        exif = im.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo) if exif else {}
    if not gps_ifd:
        return None
    gps = {GPS_TAGS.get(k, k): v for k, v in gps_ifd.items()}
    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        return None
    lat = dms_to_degrees(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
    lon = dms_to_degrees(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    alt = rational_to_float(gps.get("GPSAltitude", 0.0))
    if gps.get("GPSAltitudeRef", 0) == 1:
        alt = -alt
    return lat, lon, alt


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h) * cos_lat * math.cos(lon)
    y = (n + h) * cos_lat * math.sin(lon)
    z = (n * (1.0 - e2) + h) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(lat: float, lon: float, h: float, origin: tuple[float, float, float]) -> np.ndarray:
    origin_lat, origin_lon, origin_h = origin
    return ecef_to_enu_matrix(origin_lat, origin_lon) @ (
        geodetic_to_ecef(lat, lon, h) - geodetic_to_ecef(origin_lat, origin_lon, origin_h)
    )


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse_m": math.sqrt(float(np.mean(values**2))),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "max_m": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    origin = (
        float(manifest["origin"]["lat"]),
        float(manifest["origin"]["lon"]),
        float(manifest["origin"]["height"]),
    )
    reconstruction = pycolmap.Reconstruction(args.model)

    errors: list[np.ndarray] = []
    missing: list[str] = []
    for image in reconstruction.images.values():
        image_path = args.images / image.name
        gps = read_gps(image_path)
        if not gps:
            missing.append(image.name)
            continue
        errors.append(image.projection_center() - geodetic_to_enu(*gps, origin))

    if not errors:
        raise SystemExit("No registered images with readable GPS found.")

    arr = np.asarray(errors, dtype=np.float64)
    payload = {
        "schema": "geosplat.georef_residuals.v1",
        "method": "COLMAP camera centers fit to EXIF GPS priors",
        "model": str(args.model.resolve()),
        "images": str(args.images.resolve()),
        "manifest": str(args.manifest.resolve()),
        "registered_cameras_with_gps": len(errors),
        "missing_gps": len(missing),
        "residuals": {
            "three_d": stats(np.linalg.norm(arr, axis=1)),
            "horizontal": stats(np.linalg.norm(arr[:, :2], axis=1)),
            "vertical": stats(np.abs(arr[:, 2])),
        },
        "note": "Camera-prior fit residual, not independent surveyed GCP/checkpoint error.",
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
