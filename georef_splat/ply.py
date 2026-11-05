"""Read/write Graphdeco-style 3D Gaussian splat PLY files (binary little-endian)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


@dataclass
class Splats:
    """Gaussians in *training* parameterisation.

    positions: (N,3) metres; log_scales: (N,3) natural-log scales;
    rotations: (N,4) quaternion w,x,y,z (not necessarily normalised);
    opacity_logits: (N,) pre-sigmoid opacity; sh_dc: (N,3) degree-0 SH coefficients;
    sh_rest: (N,K) higher-order SH (channel-major, as in the PLY) or None.
    """

    positions: np.ndarray
    log_scales: np.ndarray
    rotations: np.ndarray
    opacity_logits: np.ndarray
    sh_dc: np.ndarray
    sh_rest: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.positions)

    def subset(self, mask_or_idx) -> "Splats":
        return Splats(
            self.positions[mask_or_idx],
            self.log_scales[mask_or_idx],
            self.rotations[mask_or_idx],
            self.opacity_logits[mask_or_idx],
            self.sh_dc[mask_or_idx],
            None if self.sh_rest is None else self.sh_rest[mask_or_idx],
        )

    @property
    def opacity(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logits))


def _read_header(f) -> tuple[int, list[tuple[str, str]], int]:
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    count, props, in_vertex = 0, [], False
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unterminated PLY header")
        tok = line.decode("ascii", "replace").split()
        if not tok:
            continue
        if tok[0] == "format" and tok[1] != "binary_little_endian":
            raise ValueError(f"unsupported PLY format {tok[1]}")
        if tok[0] == "element":
            in_vertex = tok[1] == "vertex"
            if in_vertex:
                count = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            if tok[1] == "list":
                raise ValueError("list properties on vertices are not supported")
            props.append((tok[2], _PLY_TYPES[tok[1]]))
        elif tok[0] == "end_header":
            return count, props, f.tell()


def read_ply(path: Path) -> Splats:
    with open(path, "rb") as f:
        count, props, offset = _read_header(f)
    dtype = np.dtype([(n, "<" + t) for n, t in props])
    v = np.fromfile(path, dtype=dtype, count=count, offset=offset)
    names = set(dtype.names)

    def cols(*keys):
        return np.stack([v[k].astype(np.float64) for k in keys], axis=1)

    rest_keys = sorted((n for n in names if n.startswith("f_rest_")), key=lambda s: int(s[7:]))
    return Splats(
        positions=cols("x", "y", "z"),
        log_scales=cols("scale_0", "scale_1", "scale_2"),
        rotations=cols("rot_0", "rot_1", "rot_2", "rot_3"),
        opacity_logits=v["opacity"].astype(np.float64),
        sh_dc=cols("f_dc_0", "f_dc_1", "f_dc_2"),
        sh_rest=cols(*rest_keys) if rest_keys else None,
    )


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Coloured point cloud (e.g. COLMAP fused.ply) -> positions (N,3), rgb in [0,1] (N,3)."""
    with open(path, "rb") as f:
        count, props, offset = _read_header(f)
    dtype = np.dtype([(n, "<" + t) for n, t in props])
    v = np.fromfile(path, dtype=dtype, count=count, offset=offset)
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    if "red" in dtype.names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64) / 255.0
    else:
        rgb = np.full((count, 3), 0.5)
    return pos, rgb


def write_ply(path: Path, s: Splats) -> None:
    fields = [("x", s.positions[:, 0]), ("y", s.positions[:, 1]), ("z", s.positions[:, 2])]
    fields += [(f"f_dc_{i}", s.sh_dc[:, i]) for i in range(3)]
    if s.sh_rest is not None:
        fields += [(f"f_rest_{i}", s.sh_rest[:, i]) for i in range(s.sh_rest.shape[1])]
    fields += [("opacity", s.opacity_logits)]
    fields += [(f"scale_{i}", s.log_scales[:, i]) for i in range(3)]
    fields += [(f"rot_{i}", s.rotations[:, i]) for i in range(4)]
    arr = np.empty(len(s), dtype=[(n, "<f4") for n, _ in fields])
    for n, col in fields:
        arr[n] = col
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(s)}"]
    header += [f"property float {n}" for n, _ in fields] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        arr.tofile(f)
