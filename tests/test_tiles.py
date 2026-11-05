import gzip
import json
import struct

import numpy as np
import pytest

from georef_splat.geodesy import enu_to_ecef_matrix
from georef_splat.ply import Splats, read_ply, write_ply
from georef_splat.tiles import (
    ENU_TO_GLTF,
    ENU_TO_GLTF_QUAT_WXYZ,
    encode_spz_v2,
    quat_multiply_wxyz,
    write_tileset,
)


def make_splats(n=500, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return Splats(
        positions=rng.uniform(-80, 80, size=(n, 3)),
        log_scales=rng.uniform(-5, 0.5, size=(n, 3)),
        rotations=q,
        opacity_logits=rng.normal(size=n),
        sh_dc=rng.uniform(-1.5, 1.5, size=(n, 3)),
    )


def decode_spz_v2(blob):
    """Independent decoder following Niantic's reference unpack formulas."""
    raw = gzip.decompress(blob)
    magic, version, n, sh_deg, frac, flags, _ = struct.unpack("<IIIBBBB", raw[:16])
    assert magic == 0x5053474E and version == 2
    sh_dim = {0: 0, 1: 3, 2: 8, 3: 15}[sh_deg]
    o = 16
    p = np.frombuffer(raw, np.uint8, n * 9, o).reshape(n, 3, 3).astype(np.int32); o += n * 9
    fixed = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    alpha = np.frombuffer(raw, np.uint8, n, o) / 255.0; o += n
    color = (np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3) / 255.0 - 0.5) / 0.15; o += n * 3
    scale = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3) / 16.0 - 10.0; o += n * 3
    xyz = np.frombuffer(raw, np.uint8, n * 3, o).reshape(n, 3) / 127.5 - 1.0; o += n * 3
    sh = (np.frombuffer(raw, np.uint8, n * sh_dim * 3, o).reshape(n, sh_dim, 3) - 128.0) / 128.0; o += n * sh_dim * 3
    assert o == len(raw)
    w = np.sqrt(np.clip(1 - (xyz**2).sum(1), 0, None))
    decode_spz_v2.sh = sh
    return fixed / (1 << frac), alpha, color, scale, np.c_[w, xyz]


def test_spz_roundtrip_within_quantisation():
    s = make_splats()
    pos, alpha, color, scale, quat = decode_spz_v2(
        encode_spz_v2(s.positions, s.log_scales, s.rotations, s.opacity, s.sh_dc)
    )
    np.testing.assert_allclose(pos, s.positions, atol=1 / 4096)
    np.testing.assert_allclose(alpha, s.opacity, atol=1 / 255)
    np.testing.assert_allclose(color, s.sh_dc, atol=0.5 / 255 / 0.15 + 1e-9)
    np.testing.assert_allclose(scale, s.log_scales, atol=1 / 32 + 1e-9)
    qn = s.rotations / np.linalg.norm(s.rotations, axis=1, keepdims=True)
    qn *= np.sign(qn[:, :1])
    assert np.abs(np.abs((quat * qn).sum(1)) - 1).max() < 0.02


def test_spz_rejects_out_of_range_positions():
    s = make_splats(10)
    with pytest.raises(ValueError):
        encode_spz_v2(s.positions + 5000, s.log_scales, s.rotations, s.opacity, s.sh_dc)


def test_enu_to_gltf_quaternion_matches_matrix():
    w, x, y, z = ENU_TO_GLTF_QUAT_WXYZ
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    np.testing.assert_allclose(r, ENU_TO_GLTF, atol=1e-12)
    # Up in ENU becomes +Y (glTF up); north becomes -Z.
    np.testing.assert_allclose(ENU_TO_GLTF @ [0, 0, 1], [0, 1, 0])
    np.testing.assert_allclose(ENU_TO_GLTF @ [0, 1, 0], [0, 0, -1])
    ident = quat_multiply_wxyz(np.array([1.0, 0, 0, 0]), ENU_TO_GLTF_QUAT_WXYZ)
    np.testing.assert_allclose(ident, ENU_TO_GLTF_QUAT_WXYZ)


def read_glb(path):
    b = path.read_bytes()
    magic, version, total = struct.unpack("<III", b[:12])
    assert magic == 0x46546C67 and version == 2 and total == len(b)
    jl, jt = struct.unpack("<II", b[12:20])
    assert jt == 0x4E4F534A
    gltf = json.loads(b[20 : 20 + jl])
    bl, bt = struct.unpack("<II", b[20 + jl : 28 + jl])
    assert bt == 0x004E4942
    return gltf, b[28 + jl : 28 + jl + bl]


def test_tileset_places_content_at_true_position(tmp_path):
    s = make_splats(300)
    origin = (41.3038, -81.7505, 312.0)
    write_tileset(s, origin, tmp_path)
    ts = json.loads((tmp_path / "tileset.json").read_text())
    np.testing.assert_allclose(np.array(ts["root"]["transform"]).reshape(4, 4).T, enu_to_ecef_matrix(origin))
    # Regression: Cesium falls back to a plain point model without this declaration.
    assert "3DTILES_content_gltf" in ts["extensionsUsed"]
    assert "KHR_gaussian_splatting_compression_spz_2" in ts["extensions"]["3DTILES_content_gltf"]["extensionsRequired"]

    gltf, binary = read_glb(tmp_path / "splats.glb")
    prim = gltf["meshes"][0]["primitives"][0]
    assert prim["mode"] == 0
    spz_ext = prim["extensions"]["KHR_gaussian_splatting"]["extensions"]["KHR_gaussian_splatting_compression_spz_2"]
    bv = gltf["bufferViews"][spz_ext["bufferView"]]
    pos, *_ = decode_spz_v2(binary[bv["byteOffset"] : bv["byteOffset"] + bv["byteLength"]])
    m = np.array(gltf["nodes"][0]["matrix"]).reshape(4, 4).T
    # Apply the glTF node, then 3D Tiles' y-up -> z-up rotation: (x, y, z) -> (x, -z, y).
    g = pos @ m[:3, :3].T + m[:3, 3]
    enu_back = np.c_[g[:, 0], -g[:, 2], g[:, 1]]
    np.testing.assert_allclose(enu_back, s.positions, atol=1e-3)
    # The node is a pure rotation + translation (decomposable, det +1), as the spec requires.
    np.testing.assert_allclose(np.linalg.det(m[:3, :3]), 1.0)


def test_spherical_harmonics_survive_packing():
    s = make_splats(200, seed=4)
    rng = np.random.default_rng(5)
    s.sh_rest = rng.uniform(-0.6, 0.6, size=(200, 45))   # PLY layout: 15 R, 15 G, 15 B
    pos, *_ = decode_spz_v2(encode_spz_v2(s.positions, s.log_scales, s.rotations, s.opacity, s.sh_dc, s.sh_rest))
    sh = decode_spz_v2.sh                                  # (n, 15 coefficients, RGB)
    want = s.sh_rest.reshape(200, 3, 15).transpose(0, 2, 1)
    np.testing.assert_allclose(sh[:, :3], want[:, :3], atol=8 / 128 / 2 + 1 / 128)    # 5-bit degree 1
    np.testing.assert_allclose(sh[:, 3:], want[:, 3:], atol=16 / 128 / 2 + 1 / 128)   # 4-bit degree 2-3


def test_ply_roundtrip(tmp_path):
    s = make_splats(50)
    write_ply(tmp_path / "s.ply", s)
    r = read_ply(tmp_path / "s.ply")
    for a, b in [(r.positions, s.positions), (r.rotations, s.rotations), (r.sh_dc, s.sh_dc)]:
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
