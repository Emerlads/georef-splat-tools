#!/usr/bin/env python3
"""Select a GPS-local, view-diverse image subset for high-fidelity splat runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}
EXIF_TAGS = [
    "-SourceFile",
    "-FileName",
    "-Directory",
    "-ImageWidth",
    "-ImageHeight",
    "-GPSLatitude",
    "-GPSLongitude",
    "-GPSAltitude",
    "-RelativeAltitude",
    "-AbsoluteAltitude",
    "-GimbalPitchDegree",
    "-GimbalYawDegree",
    "-FlightYawDegree",
    "-DateTimeOriginal",
    "-CreateDate",
]


def clean_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace("+", "").strip())
    except ValueError:
        return None


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371008.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def list_images(roots: list[Path], name_regex: str | None, exclude_regex: str | None) -> list[Path]:
    include = re.compile(name_regex, re.IGNORECASE) if name_regex else None
    exclude = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
    images: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            text = str(path)
            if not path.is_file() or path.suffix not in IMAGE_EXTS:
                continue
            if include and not include.search(path.name):
                continue
            if exclude and exclude.search(text):
                continue
            images.append(path)
    return sorted(images)


def run_exiftool(paths: list[Path], batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        cmd = ["exiftool", "-n", "-json", "-s", *EXIF_TAGS, *[str(p) for p in batch]]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        rows.extend(json.loads(proc.stdout))
    return rows


def normalize(row: dict[str, Any], target_lat: float, target_lon: float) -> dict[str, Any] | None:
    lat = clean_number(first(row, "GpsLatitude", "GPSLatitude"))
    lon = clean_number(first(row, "GpsLongitude", "GPSLongitude"))
    if lat is None or lon is None:
        return None
    source = str(row.get("SourceFile"))
    return {
        "source_file": source,
        "file_name": row.get("FileName") or Path(source).name,
        "directory": row.get("Directory") or str(Path(source).parent),
        "longitude_deg": lon,
        "latitude_deg": lat,
        "gps_altitude_m": clean_number(row.get("GPSAltitude")),
        "absolute_altitude_m": clean_number(row.get("AbsoluteAltitude")),
        "relative_altitude_m": clean_number(row.get("RelativeAltitude")),
        "gimbal_pitch_deg": clean_number(row.get("GimbalPitchDegree")),
        "gimbal_yaw_deg": clean_number(row.get("GimbalYawDegree")),
        "flight_yaw_deg": clean_number(row.get("FlightYawDegree")),
        "image_width": clean_number(row.get("ImageWidth")),
        "image_height": clean_number(row.get("ImageHeight")),
        "datetime": first(row, "DateTimeOriginal", "CreateDate"),
        "distance_m": haversine_m(target_lat, target_lon, lat, lon),
    }


def pitch_bin(value: float | None) -> str:
    if value is None:
        return "pitch_unknown"
    if value <= -78:
        return "nadir"
    if value <= -50:
        return "oblique"
    return "shallow"


def altitude_bin(value: float | None) -> str:
    if value is None:
        return "alt_unknown"
    if value < 35:
        return "low"
    if value < 80:
        return "mid"
    return "high"


def yaw_bin(value: float | None, sectors: int = 8) -> str:
    if value is None:
        return "yaw_unknown"
    normalized = (value + 360.0) % 360.0
    return f"yaw_{int(normalized / (360.0 / sectors)):02d}"


def row_bin(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        pitch_bin(row.get("gimbal_pitch_deg")),
        altitude_bin(row.get("relative_altitude_m")),
        yaw_bin(row.get("gimbal_yaw_deg") or row.get("flight_yaw_deg")),
    )


def far_enough(row: dict[str, Any], selected: list[dict[str, Any]], min_spacing_m: float) -> bool:
    if min_spacing_m <= 0:
        return True
    lat = float(row["latitude_deg"])
    lon = float(row["longitude_deg"])
    return all(
        haversine_m(lat, lon, float(other["latitude_deg"]), float(other["longitude_deg"])) >= min_spacing_m
        for other in selected
    )


def select_diverse(rows: list[dict[str, Any]], count: int, min_spacing_m: float) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: float(r["distance_m"])):
        buckets[row_bin(row)].append(row)

    # Start with the bins that have the closest useful images, then round-robin
    # through view groups so one orbit or yaw direction does not dominate.
    bin_order = sorted(buckets, key=lambda key: float(buckets[key][0]["distance_m"]))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    spacing = min_spacing_m

    while len(selected) < count and bin_order:
        added_this_round = False
        for key in list(bin_order):
            bucket = buckets[key]
            while bucket:
                row = bucket.pop(0)
                source = str(row["source_file"])
                if source in seen:
                    continue
                if not far_enough(row, selected, spacing):
                    continue
                selected.append(row)
                seen.add(source)
                added_this_round = True
                break
            if not bucket:
                bin_order.remove(key)
            if len(selected) >= count:
                break
        if not added_this_round:
            if spacing <= 0.05:
                break
            spacing *= 0.5

    if len(selected) < count:
        for row in sorted(rows, key=lambda r: float(r["distance_m"])):
            source = str(row["source_file"])
            if source in seen:
                continue
            selected.append(row)
            seen.add(source)
            if len(selected) >= count:
                break
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "source_file",
        "file_name",
        "directory",
        "longitude_deg",
        "latitude_deg",
        "gps_altitude_m",
        "absolute_altitude_m",
        "relative_altitude_m",
        "gimbal_pitch_deg",
        "gimbal_yaw_deg",
        "flight_yaw_deg",
        "image_width",
        "image_height",
        "datetime",
        "distance_m",
        "selection_bin",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["selection_bin"] = "/".join(row_bin(row))
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--target-lat", required=True, type=float)
    parser.add_argument("--target-lon", required=True, type=float)
    parser.add_argument("--name-regex", default=None)
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--max-distance-m", type=float, default=75.0)
    parser.add_argument(
        "--max-gimbal-pitch-deg",
        type=float,
        default=None,
        help="Keep only images with gimbal pitch <= this value, e.g. -20 to reject near-horizon frames.",
    )
    parser.add_argument("--count", type=int, default=360)
    parser.add_argument("--min-spacing-m", type=float, default=0.75)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    if args.copy_images:
        if image_dir.exists():
            if not args.force:
                raise SystemExit(f"Output image folder exists: {image_dir}. Use --force.")
            shutil.rmtree(image_dir)
        image_dir.mkdir(parents=True)

    images = list_images(args.root, args.name_regex, args.exclude_regex)
    print(f"Candidate images: {len(images)}")
    records = [normalize(r, args.target_lat, args.target_lon) for r in run_exiftool(images, args.batch_size)]
    rows = [r for r in records if r is not None]
    rows.sort(key=lambda r: float(r["distance_m"]))
    nearby = [r for r in rows if float(r["distance_m"]) <= args.max_distance_m]
    if args.max_gimbal_pitch_deg is not None:
        nearby = [
            r
            for r in nearby
            if r.get("gimbal_pitch_deg") is None
            or float(r["gimbal_pitch_deg"]) <= args.max_gimbal_pitch_deg
        ]
    selected = select_diverse(nearby, args.count, args.min_spacing_m)

    for radius in [10, 25, 50, 75, 100, 120, 150]:
        print(f"within {radius:>3} m: {sum(1 for r in rows if float(r['distance_m']) <= radius)}")
    print(f"Selected: {len(selected)} within {args.max_distance_m} m")
    for row in selected[:12]:
        print(
            f"  {row['distance_m']:.2f} m  pitch={row.get('gimbal_pitch_deg')} "
            f"alt={row.get('relative_altitude_m')}  {row['source_file']}"
        )

    write_csv(args.output / "gps_candidates.csv", rows)
    write_csv(args.output / "selected_images.csv", selected)

    if args.copy_images:
        for idx, row in enumerate(selected, start=1):
            src = Path(str(row["source_file"]))
            shutil.copy2(src, image_dir / f"{idx:05d}_{src.name}")
        print(f"Copied selected images to: {image_dir}")
    print(f"Wrote: {args.output / 'selected_images.csv'}")


if __name__ == "__main__":
    main()
