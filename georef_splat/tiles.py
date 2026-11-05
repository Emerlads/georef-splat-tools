"""Write ENU Gaussian splats as a georeferenced 3D Tiles 1.1 tileset.

Content is glTF 2.0 using ``KHR_gaussian_splatting`` with
``KHR_gaussian_splatting_compression_spz_2`` (the form CesiumJS renders).

Frames, explicitly:
  * splats arrive in local ENU metres (x=east, y=north, z=up) about ``origin``;
  * glTF is y-up, and 3D Tiles rotates glTF content y-up -> z-up on load, so
    content is stored as (east, up, -north);
  * the tileset root ``transform`` maps ENU -> ECEF at ``origin``, whose height
    must be ellipsoidal (WGS84), not orthometric.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np

from .geodesy import enu_to_ecef_matrix
from .ply import Splats

SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_COLOR_SCALE = 0.15
_HALF = math.sqrt(0.5)
# Rotation of -90 deg about X: ENU (e, n, u) -> glTF (e, u, -n).
ENU_TO_GLTF = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
ENU_TO_GLTF_QUAT_WXYZ = np.array([_HALF, -_HALF, 0.0, 0.0])


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.broadcast_to(a, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.broadcast_to(b, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def encode_spz_v2(positions, log_scales, rotations_wxyz, opacity, sh_dc, sh_rest=None, fractional_bits: int = 12) -> bytes:
    """Pack splats into a gzipped SPZ v2 stream, with spherical harmonics if given."""
    n = len(positions)
    fixed = np.round(np.asarray(positions, dtype=np.float64) * (1 << fractional_bits)).astype(np.int64)
    limit = 1 << 23
    if np.any(fixed >= limit) or np.any(fixed < -limit):
        raise ValueError("positions exceed 24-bit fixed-point range; recentre or lower fractional_bits")
    fixed &= 0xFFFFFF
    pos_bytes = np.stack([(fixed >> s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.uint8)

    alpha = _u8(np.asarray(opacity) * 255.0)
    color = _u8(np.asarray(sh_dc) * (SPZ_COLOR_SCALE * 255.0) + 0.5 * 255.0)
    scale = _u8((np.asarray(log_scales) + 10.0) * 16.0)

    q = np.asarray(rotations_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(q[:, :1] < 0, -q, q)
    rot = _u8(q[:, 1:4] * 127.5 + 127.5)

    sh_degree, sh_bytes = 0, b""
    if sh_rest is not None and sh_rest.shape[1] > 0:
        dim = sh_rest.shape[1] // 3                      # coefficients per channel (3, 8 or 15)
        sh_degree = {3: 1, 8: 2, 15: 3}[dim]
        # PLY f_rest is channel-major (all R, all G, all B); SPZ wants coefficient-major, RGB interleaved.
        sh = np.asarray(sh_rest, dtype=np.float64).reshape(n, 3, dim).transpose(0, 2, 1)
        q8 = np.round(sh * 128.0) + 128.0
        bucket = np.full(dim, 16.0)                       # 4 bits for degree 2-3 (SPZ default)
        bucket[:3] = 8.0                                  # 5 bits for degree 1
        q8 = np.floor((q8 + bucket[None, :, None] / 2) / bucket[None, :, None]) * bucket[None, :, None]
        sh_bytes = np.clip(q8, 0, 255).astype(np.uint8).reshape(-1).tobytes()

    header = struct.pack("<IIIBBBB", SPZ_MAGIC, 2, n, sh_degree, fractional_bits, 0, 0)
    body = b"".join(a.tobytes() for a in (pos_bytes.reshape(-1), alpha, color.reshape(-1), scale.reshape(-1), rot.reshape(-1)))
    return gzip.compress(header + body + sh_bytes, compresslevel=9, mtime=0)


def build_glb(splats: Splats, node_matrix) -> bytes:
    """GLB with one SPZ-compressed splat primitive.

    ``node_matrix`` is a 4x4 (row-major numpy) placed on the glTF node. Keeping
    the axis change there, rather than baking it into the data, means the
    renderer rotates positions, orientations *and* spherical harmonics
    consistently, with no Wigner-D rotation needed here.
    """
    n = len(splats)
    spz = encode_spz_v2(splats.positions, splats.log_scales, splats.rotations, splats.opacity, splats.sh_dc, splats.sh_rest)
    pad = (-len(spz)) % 4
    bin_chunk = spz + b"\x00" * pad
    pmin = splats.positions.min(0).tolist()
    pmax = splats.positions.max(0).tolist()
    attributes = {
        "POSITION": 0,
        "COLOR_0": 1,
        "KHR_gaussian_splatting:SCALE": 2,
        "KHR_gaussian_splatting:ROTATION": 3,
    }
    accessors = [
        {"componentType": 5126, "count": n, "type": "VEC3", "min": pmin, "max": pmax},
        {"componentType": 5121, "count": n, "type": "VEC4", "normalized": True},
        {"componentType": 5126, "count": n, "type": "VEC3"},
        {"componentType": 5126, "count": n, "type": "VEC4"},
    ]
    if splats.sh_rest is not None and splats.sh_rest.shape[1] > 0:
        dim = splats.sh_rest.shape[1] // 3
        names = [(1, i) for i in range(3)] + [(2, i) for i in range(5)] + [(3, i) for i in range(7)]
        for lvl, i in names[:dim]:
            attributes[f"KHR_gaussian_splatting:SH_DEGREE_{lvl}_COEF_{i}"] = len(accessors)
            accessors.append({"componentType": 5126, "count": n, "type": "VEC3"})
    gltf = {
        "asset": {"version": "2.0", "generator": "georef-splat"},
        "extensionsUsed": ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2", "KHR_materials_unlit"],
        "extensionsRequired": ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "matrix": [float(v) for v in np.asarray(node_matrix, dtype=np.float64).T.reshape(-1)]}],
        "materials": [{"extensions": {"KHR_materials_unlit": {}}}],
        "meshes": [{"primitives": [{
            "mode": 0,
            "material": 0,
            "attributes": attributes,
            "extensions": {"KHR_gaussian_splatting": {
                "kernel": "ellipse",
                "colorSpace": "srgb_rec709_display",
                "projection": "perspective",
                "sortingMethod": "cameraDistance",
                "extensions": {"KHR_gaussian_splatting_compression_spz_2": {"bufferView": 0}},
            }},
        }]}],
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(spz)}],
        "accessors": accessors,
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    return b"".join([
        struct.pack("<III", 0x46546C67, 2, total),
        struct.pack("<II", len(js), 0x4E4F534A), js,
        struct.pack("<II", len(bin_chunk), 0x004E4942), bin_chunk,
    ])


def enu_splats_to_gltf_frame(s: Splats) -> Splats:
    return Splats(
        positions=s.positions @ ENU_TO_GLTF.T,
        log_scales=s.log_scales,
        rotations=quat_multiply_wxyz(ENU_TO_GLTF_QUAT_WXYZ, s.rotations),
        opacity_logits=s.opacity_logits,
        sh_dc=s.sh_dc,
        sh_rest=None,  # higher-order SH would need a Wigner-D rotation; write_tileset avoids this path.
    )


def write_tileset(enu: Splats, origin: tuple[float, float, float], out_dir: Path, extras: dict | None = None,
                  sh: bool = True) -> dict:
    """Write ``tileset.json`` + ``splats.glb``; return a small build summary.

    Splat data stays in local ENU (recentred for 24-bit SPZ positions). The glTF
    node carries ENU -> glTF axes plus the recentring, so higher-order spherical
    harmonics survive without being rotated here.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = enu.positions.min(0), enu.positions.max(0)
    centre = (lo + hi) / 2.0
    local = Splats(enu.positions - centre, enu.log_scales, enu.rotations, enu.opacity_logits, enu.sh_dc,
                   enu.sh_rest if sh else None)
    node = np.eye(4)
    node[:3, :3] = ENU_TO_GLTF
    node[:3, 3] = ENU_TO_GLTF @ centre
    glb = build_glb(local, node)
    (out_dir / "splats.glb").write_bytes(glb)

    elo, ehi = enu.positions.min(0), enu.positions.max(0)
    ec, half = (elo + ehi) / 2.0, (ehi - elo) / 2.0 + 1.0
    gltf_ext = ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"]
    tileset = {
        "asset": {"version": "1.1", "generator": "georef-splat"},
        # CesiumJS only routes content to its splat renderer when the tileset
        # declares the glTF extensions; without this it draws a point model.
        "extensionsUsed": ["3DTILES_content_gltf"],
        "extensions": {"3DTILES_content_gltf": {"extensionsUsed": gltf_ext, "extensionsRequired": gltf_ext}},
        "geometricError": float(np.linalg.norm(half) * 2),
        "root": {
            "transform": enu_to_ecef_matrix(origin).T.flatten().tolist(),
            "boundingVolume": {"box": [*ec.tolist(), half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]},
            "geometricError": 0.0,
            "refine": "ADD",
            "content": {"uri": "splats.glb"},
        },
        "extras": {"origin_lat_lon_h_ellipsoidal": list(origin), **(extras or {})},
    }
    (out_dir / "tileset.json").write_text(json.dumps(tileset, indent=1), encoding="utf-8")
    return {"splats": len(enu), "glb_bytes": len(glb), "tileset": str(out_dir / "tileset.json")}
