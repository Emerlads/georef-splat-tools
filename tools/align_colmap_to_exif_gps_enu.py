#!/usr/bin/env python3
"""
Align a native COLMAP reconstruction to image EXIF GPS in a local ENU meter frame.

This produces a geospatially meaningful COLMAP model without routing through any
cloud service. GPS is used as a weak similarity alignment target; RTK/GCPs should
replace or refine this later when available.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
    # WGS84 ellipsoid.
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
    ecef = geodetic_to_ecef(lat, lon, h)
    origin_ecef = geodetic_to_ecef(origin_lat, origin_lon, origin_h)
    return ecef_to_enu_matrix(origin_lat, origin_lon) @ (ecef - origin_ecef)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/example_site/local_colmap/sparse/0"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("runs/example_site/local_colmap/images"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/example_site/local_colmap/sparse_enu/0"),
    )
    parser.add_argument("--min-common-images", type=int, default=20)
    parser.add_argument("--max-error", type=float, default=25.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    images = args.images.resolve()
    output = args.output.resolve()
    if not model.exists():
        raise SystemExit(f"COLMAP model does not exist: {model}")
    if not images.exists():
        raise SystemExit(f"Image folder does not exist: {images}")
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {output}. Use --force.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(model)

    gps_by_name: dict[str, tuple[float, float, float]] = {}
    for image_path in sorted(images.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        gps = read_gps(image_path)
        if gps:
            gps_by_name[image_path.name] = gps
            gps_by_name[str(image_path.relative_to(images)).replace("\\", "/")] = gps

    registered = []
    lla = []
    for image in reconstruction.images.values():
        gps = gps_by_name.get(image.name)
        if gps:
            registered.append(image.name)
            lla.append(gps)

    if len(registered) < args.min_common_images:
        raise SystemExit(f"Only {len(registered)} registered images have GPS; need {args.min_common_images}.")

    arr = np.array(lla, dtype=np.float64)
    origin = (float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
    enu = np.array([geodetic_to_enu(lat, lon, alt, origin) for lat, lon, alt in lla], dtype=np.float64)

    ransac = pycolmap.RANSACOptions()
    ransac.max_error = args.max_error
    sim = pycolmap.align_reconstruction_to_locations(
        reconstruction,
        registered,
        enu,
        args.min_common_images,
        ransac,
    )
    if sim is None:
        raise SystemExit("Failed to align COLMAP reconstruction to EXIF GPS.")

    reconstruction.transform(sim)
    reconstruction.write(output)
    txt_output = output.parent.parent / "sparse_enu_txt"
    if txt_output.exists():
        shutil.rmtree(txt_output)
    txt_output.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(txt_output)

    manifest = {
        "pipeline": "native_pycolmap_to_exif_gps_enu",
        "crs": "local ENU meters from WGS84 EXIF GPS",
        "origin": {
            "lat": origin[0],
            "lon": origin[1],
            "height": origin[2],
        },
        "registered_images_with_gps": len(registered),
        "sim3_colmap_to_enu": jsonable(sim.todict()),
        "ransac_max_error_m": args.max_error,
        "source_model": str(model),
        "output_model": str(output),
    }
    manifest_path = output.parent.parent / "georeference_manifest_enu.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Registered images with GPS: {len(registered)}")
    print(f"ENU origin: lat={origin[0]:.12f}, lon={origin[1]:.12f}, h={origin[2]:.3f}")
    print(f"Scale: {sim.scale}")
    print(f"Translation: {sim.translation}")
    print(f"Aligned binary model: {output}")
    print(f"Aligned text model: {txt_output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
