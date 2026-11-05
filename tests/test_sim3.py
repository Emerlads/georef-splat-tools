import numpy as np
import pytest

from georef_splat.sim3 import ransac_umeyama, residual_report, umeyama


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3)) * 10
    r, s, t = random_rotation(rng), 37.5, np.array([100.0, -20.0, 5.0])
    dst = s * src @ r.T + t
    est = umeyama(src, dst)
    assert est.scale == pytest.approx(s, rel=1e-9)
    np.testing.assert_allclose(est.rotation, r, atol=1e-9)
    np.testing.assert_allclose(est.translation, t, atol=1e-6)


def test_umeyama_never_returns_a_reflection():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(20, 3))
    dst = src * np.array([1, 1, -1])  # mirrored target
    assert np.linalg.det(umeyama(src, dst).rotation) == pytest.approx(1.0)


def test_ransac_ignores_bad_gps_fixes():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(60, 3)) * 20
    r, s, t = random_rotation(rng), 2.0, np.array([5.0, 6.0, 7.0])
    dst = s * src @ r.T + t + rng.normal(scale=0.05, size=(60, 3))
    dst[:6] += rng.normal(scale=40, size=(6, 3))  # 10% outliers
    est, inliers = ransac_umeyama(src, dst, threshold=0.5)
    assert not inliers[:6].any() and inliers[6:].all()
    rep = residual_report(est.apply(src[inliers]), dst[inliers])
    assert rep["three_d"]["rmse_m"] < 0.15
    assert rep["horizontal"]["n"] == 54
