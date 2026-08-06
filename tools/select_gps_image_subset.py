#!/usr/bin/env python3
"""Select GPS-tagged images near a target lon/lat for controlled splat runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}
TAGS = [
    "-SourceFile",
    "-FileName",
    "-Directory",
    "-GPSLatitude",
    "-GPSLongitude",
    "-XMP-drone-dji:GPSLatitude",
    "-XMP-drone-dji:GPSLongitude",
    "-XMP-drone-dji:AbsoluteAltitude",
    "-XMP-drone-dji:RelativeAltitude",
]


def clean_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
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


def list_images(roots: list[Path], name_regex: str, exclude_regex: str | None) -> list[Path]:
    include = re.compile(name_regex, re.IGNORECASE)
    exclude = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
    images: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            text = str(path)
            if not path.is_file() or path.suffix not in IMAGE_EXTS:
                continue
            if not include.search(path.name):
                continue
            if exclude and exclude.search(text):
                continue
            images.append(path)
    return sorted(images)


def run_exiftool(paths: list[Path], batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        cmd = ["exiftool", "-n", "-json", *TAGS, *[str(p) for p in batch]]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        rows.extend(json.loads(proc.stdout))
    return rows


def normalize(row: dict[str, Any], target_lat: float, target_lon: float) -> dict[str, Any] | None:
    lon = clean_number(first(row, "XMP-drone-dji:GPSLongitude", "GPSLongitude", "GPS Longitude"))
    lat = clean_number(first(row, "XMP-drone-dji:GPSLatitude", "GPSLatitude", "GPS Latitude"))
    if lon is None or lat is None:
        return None
    source = str(row.get("SourceFile"))
    distance = haversine_m(target_lat, target_lon, lat, lon)
    return {
        "source_file": source,
        "file_name": row.get("FileName") or Path(source).name,
        "directory": row.get("Directory") or str(Path(source).parent),
        "longitude_deg": lon,
        "latitude_deg": lat,
        "absolute_altitude_m": clean_number(first(row, "XMP-drone-dji:AbsoluteAltitude", "AbsoluteAltitude")),
        "relative_altitude_m": clean_number(first(row, "XMP-drone-dji:RelativeAltitude", "RelativeAltitude")),
        "distance_m": distance,
    }


def greedy_spacing(rows: list[dict[str, Any]], count: int, min_spacing_m: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if len(selected) >= count:
            break
        lat = float(row["latitude_deg"])
        lon = float(row["longitude_deg"])
        if all(
            haversine_m(lat, lon, float(other["latitude_deg"]), float(other["longitude_deg"])) >= min_spacing_m
            for other in selected
        ):
            selected.append(row)
    if len(selected) < count:
        seen = {r["source_file"] for r in selected}
        for row in rows:
            if len(selected) >= count:
                break
            if row["source_file"] not in seen:
                selected.append(row)
                seen.add(row["source_file"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--target-lat", required=True, type=float)
    parser.add_argument("--target-lon", required=True, type=float)
    parser.add_argument("--name-regex", default="_V")
    parser.add_argument("--exclude-regex", default=r"\\Multispec\\")
    parser.add_argument("--max-distance-m", type=float, default=120.0)
    parser.add_argument("--count", type=int, default=220)
    parser.add_argument("--min-spacing-m", type=float, default=1.0)
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
    selected = greedy_spacing(nearby, args.count, args.min_spacing_m)

    for radius in [25, 50, 75, 100, 120, 150, 200, 300]:
        print(f"within {radius:>3} m: {sum(1 for r in rows if float(r['distance_m']) <= radius)}")
    print(f"Selected: {len(selected)} within {args.max_distance_m} m")
    if selected:
        print("Closest selected:")
        for row in selected[:10]:
            print(f"  {row['distance_m']:.2f} m  {row['source_file']}")

    all_csv = args.output / "gps_candidates.csv"
    selected_csv = args.output / "selected_images.csv"
    columns = [
        "source_file",
        "file_name",
        "directory",
        "longitude_deg",
        "latitude_deg",
        "absolute_altitude_m",
        "relative_altitude_m",
        "distance_m",
    ]
    for path, data in [(all_csv, rows), (selected_csv, selected)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)

    if args.copy_images:
        for idx, row in enumerate(selected, start=1):
            src = Path(str(row["source_file"]))
            shutil.copy2(src, image_dir / f"{idx:05d}_{src.name}")
        print(f"Copied selected images to: {image_dir}")
    print(f"Wrote: {selected_csv}")


if __name__ == "__main__":
    main()
