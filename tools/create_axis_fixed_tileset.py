"""Create a corrected 3D Tiles wrapper when a converter used the wrong source axes.

The current local 3DGS -> 3D Tiles converter defaulted to
``camera_y_down_z_forward`` for some ENU-trained PLYs because the PLY comment
did not declare a coordinate system. That converts ENU content approximately as:

    converter_output = [east, up, -north]

Cesium then receives a valid ENU-to-ECEF root transform, but the content is
already axis-swapped, so it appears tilted/vertical. This tool writes a sibling
tileset JSON that reuses the existing tile GLBs and applies:

    ENU = [output.x, -output.z, output.y]

at the root transform.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


AXIS_FIX_OUTPUT_TO_ENU_COL_MAJOR = [
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, -1, 0, 0,
    0, 0, 0, 1,
]


def matmul_col_major(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return out


def rewrite_content_uris(tile: dict[str, Any], prefix: str) -> None:
    content = tile.get("content")
    if isinstance(content, dict) and isinstance(content.get("uri"), str):
        content["uri"] = f"{prefix.rstrip('/')}/{content['uri']}"

    for child in tile.get("children", []) or []:
        rewrite_content_uris(child, prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tileset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--content-prefix",
        default=None,
        help="Prefix from output tileset folder to source tile contents. Defaults to ../<source-folder-name>",
    )
    args = parser.parse_args()

    source_tileset = args.source_tileset.resolve()
    source_dir = source_tileset.parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(source_tileset.read_text(encoding="utf-8"))
    fixed = copy.deepcopy(data)
    root = fixed["root"]
    old_transform = root["transform"]
    root["transform"] = matmul_col_major(old_transform, AXIS_FIX_OUTPUT_TO_ENU_COL_MAJOR)

    prefix = args.content_prefix or f"../{source_dir.name}"
    rewrite_content_uris(root, prefix)

    fixed.setdefault("extras", {})
    fixed["extras"]["geosplatAxisFix"] = {
        "reason": "Source PLY was ENU, but converter assumed camera_y_down_z_forward and emitted output axes [east, up, -north].",
        "outputToEnuColumnMajor": AXIS_FIX_OUTPUT_TO_ENU_COL_MAJOR,
        "sourceTileset": str(source_tileset),
    }

    (output_dir / "tileset.json").write_text(json.dumps(fixed, separators=(",", ":")), encoding="utf-8")

    summary_path = source_dir / "build_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["axis_fix"] = fixed["extras"]["geosplatAxisFix"]
        summary["source_coordinate_system_corrected_to"] = "enu_x_east_y_north_z_up"
        (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_dir / "tileset.json"), "content_prefix": prefix}, indent=2))


if __name__ == "__main__":
    main()
