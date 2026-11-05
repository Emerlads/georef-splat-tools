import numpy as np
import pytest

pytest.importorskip("scipy")

from georef_splat.audit import CONSISTENT, FLOATING, audit
from georef_splat.ply import Splats
from georef_splat.zones import Params


def test_audit_finds_the_floating_half():
    rng = np.random.default_rng(0)
    # Reference: dense points on a flat ground plane, 60 x 40 m.
    ref = np.c_[rng.uniform(0, 60, 200_000), rng.uniform(0, 40, 200_000), rng.normal(0, 0.02, 200_000)]
    # Splat: west half on the ground, east half a floating layer 10 m up (same colours).
    n = 120_000
    xy = np.c_[rng.uniform(0, 60, n), rng.uniform(0, 40, n)]
    z = np.where(xy[:, 0] < 30, 0.0, 10.0) + rng.normal(0, 0.05, n)
    s = Splats(
        positions=np.c_[xy, z],
        log_scales=np.tile(np.log([0.2, 0.2, 0.02]), (n, 1)),
        rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
        opacity_logits=np.full(n, 4.0),
        sh_dc=np.zeros((n, 3)),
    )
    r = audit(s, ref, np.full((len(ref), 3), 0.5), Params(cell=0.5))
    v = r["verdict"]
    west, east = v[:, : v.shape[1] // 2 - 2], v[:, v.shape[1] // 2 + 2 :]
    assert (west == CONSISTENT).mean() > 0.95
    assert (east == FLOATING).mean() > 0.95
    assert r["summary"]["worst_floaters_m"] == pytest.approx(10, abs=0.5)
