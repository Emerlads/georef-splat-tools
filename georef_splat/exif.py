"""Read GPS position from image EXIF."""

from __future__ import annotations

from pathlib import Path

from PIL import ExifTags, Image

Image.MAX_IMAGE_PIXELS = None
_GPS_TAGS = ExifTags.GPSTAGS


def _rational(value) -> float:
    try:
        return float(value)
    except TypeError:
        return float(value[0]) / float(value[1])


def dms_to_degrees(values, ref: str) -> float:
    deg, minute, second = (_rational(v) for v in values)
    sign = -1.0 if ref in ("S", "W") else 1.0
    return sign * (deg + minute / 60.0 + second / 3600.0)


def read_gps(path: Path) -> tuple[float, float, float] | None:
    """Return (lat, lon, altitude) or None.

    The altitude is returned exactly as stored. EXIF does not say which vertical
    datum it uses: many drones write height above mean sea level, some write
    ellipsoidal height, and some write barometric height above takeoff. Treat it
    as unknown until it is checked against independent terrain.
    """
    with Image.open(path) as im:
        exif = im.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo) if exif else {}
    if not gps_ifd:
        return None
    gps = {_GPS_TAGS.get(k, k): v for k, v in gps_ifd.items()}
    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        return None
    lat = dms_to_degrees(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
    lon = dms_to_degrees(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    alt = _rational(gps.get("GPSAltitude", 0.0))
    if gps.get("GPSAltitudeRef", 0) in (1, b"\x01"):
        alt = -alt
    return lat, lon, alt


def read_gps_folder(folder: Path) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}:
            g = read_gps(p)
            if g:
                out[p.name] = g
    return out
