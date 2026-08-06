"""Build a viewer-ready source image index for click-to-nearest-photo lookup.

The Cesium viewer should not parse large pipeline CSVs directly. This tool
turns a run's selected image CSV into a compact JSON sidecar that can be copied
beside a splat/tileset asset under ``cesium-gaussian-splatting/public``.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-images-csv", required=True, help="CSV with source_file/file_name/lon/lat/alt columns")
    parser.add_argument("--output", required=True, help="Output JSON path for the web viewer")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--georeference-manifest", help="Optional manifest path copied into metadata")
    args = parser.parse_args()

    selected_csv = Path(args.selected_images_csv).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with selected_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            lon = parse_float(row.get("longitude_deg"))
            lat = parse_float(row.get("latitude_deg"))
            if lon is None or lat is None:
                continue

            records.append(
                {
                    "id": idx,
                    "fileName": row.get("file_name") or Path(row.get("source_file", "")).name,
                    "sourcePath": row.get("source_file"),
                    "directory": row.get("directory"),
                    "lon": lon,
                    "lat": lat,
                    "heightM": parse_float(row.get("absolute_altitude_m")),
                    "relativeHeightM": parse_float(row.get("relative_altitude_m")),
                    "selectionDistanceM": parse_float(row.get("distance_m")),
                }
            )

    payload = {
        "schema": "geosplat.source_image_index.v1",
        "siteId": args.site_id,
        "runId": args.run_id,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceCsv": str(selected_csv),
        "georeferenceManifest": str(Path(args.georeference_manifest).resolve()) if args.georeference_manifest else None,
        "clickLookup": {
            "strategy": "nearest_camera_prior_horizontal_distance",
            "note": "Uses EXIF/GPS camera prior locations. Orientation-aware visibility scoring can be added once camera quaternions are exported to the index.",
        },
        "images": records,
    }

    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "images": len(records)}, indent=2))


if __name__ == "__main__":
    main()
