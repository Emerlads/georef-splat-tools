import numpy as np
import pytest

pytest.importorskip("scipy")

from georef_splat.ply import Splats
from georef_splat.zones import (
    FT, SHRUB, STRUCTURE, TREE, Params, buffer_convex, classify, rasterise, screen_structures,
)

SH_C0 = 0.28209479177387814


def rgb_to_dc(rgb):
    return (np.asarray(rgb, dtype=float) - 0.5) / SH_C0


def synthetic_scene(seed=0):
    """Flat sloping ground, a 12x8 m flat-roofed house 5 m tall, a 7 m tree
    touching its east wall, and a shrub patch 6 m west of it."""
    rng = np.random.default_rng(seed)
    parts = []

    def add(xy, z, rgb):
        parts.append((np.c_[xy, z], np.tile(rgb_to_dc(rgb), (len(xy), 1))))

    g = rng.uniform([-40, -40], [40, 40], size=(160_000, 2))
    add(g, 0.02 * g[:, 0] + rng.normal(0, 0.03, len(g)), [0.55, 0.5, 0.42])          # bare soil, gentle slope
    r = rng.uniform([-6, -4], [6, 4], size=(20_000, 2))
    add(r, 5.0 + 0.02 * r[:, 0] + rng.normal(0, 0.02, len(r)), [0.6, 0.6, 0.62])    # grey roof
    c = rng.normal([9.0, 0.0], 1.6, size=(12_000, 2))
    add(c, 7.0 + 0.02 * c[:, 0] - 0.4 * np.linalg.norm(c - [9, 0], axis=1) + rng.normal(0, 0.3, len(c)), [0.2, 0.5, 0.15])
    sh = rng.uniform([-12, -3], [-9, 3], size=(6_000, 2))
    add(sh, 1.0 + 0.02 * sh[:, 0] + rng.normal(0, 0.15, len(sh)), [0.3, 0.55, 0.2])

    pos = np.concatenate([p for p, _ in parts])
    dc = np.concatenate([d for _, d in parts])
    n = len(pos)
    return Splats(
        positions=pos,
        log_scales=np.full((n, 3), np.log(0.08)),
        rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
        opacity_logits=np.full(n, 3.0),
        sh_dc=dc,
    )


@pytest.fixture(scope="module")
def screened():
    p = Params(cell=0.5)
    g = rasterise(synthetic_scene(), p)
    classify(g, p)
    return g, screen_structures(g, p)


def test_finds_exactly_one_structure_of_the_right_size(screened):
    g, res = screened
    assert len(res) == 1
    assert res[0]["footprint_area_m2"] == pytest.approx(96, rel=0.15)
    assert res[0]["height_m"] == pytest.approx(5.0, abs=0.6)


def test_ground_model_ignores_the_house(screened):
    g, _ = screened
    j = int((0 - g.x0) / g.cell)
    i = int((0 - g.y0) / g.cell)
    assert abs(g.dtm[i, j] - 0.0) < 0.5


def test_classes_land_where_expected(screened):
    g, _ = screened
    def cls_at(x, y):
        return g.classes[int((y - g.y0) / g.cell), int((x - g.x0) / g.cell)]
    assert cls_at(0, 0) == STRUCTURE
    assert cls_at(9, 0) == TREE
    assert cls_at(-10.5, 0) == SHRUB


def test_zone0_tree_is_flagged_and_shrubs_land_in_zone1(screened):
    _, res = screened
    z = res[0]["zones"]
    assert z["zone0"]["tree_cover_pct"] > 10          # canopy touches the wall
    assert z["zone1"]["shrub_cover_pct"] > 1          # shrubs are 3-6 m from the west wall
    assert z["zone0"]["woody_cover_pct"] > z["zone0"]["shrub_cover_pct"]  # mostly the tree, not shrubs
    assert res[0]["priority"] == "review"
    assert any("Zone 0" in f for f in res[0]["flags"])


def test_zone_ring_widths_are_in_feet():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    b = buffer_convex(sq, 30 * FT)
    assert b[:, 0].max() == pytest.approx(10 + 30 * FT, abs=0.01)
    assert b[:, 1].min() == pytest.approx(-30 * FT, abs=0.01)
