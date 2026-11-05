"""Similarity (Sim3) fitting and residual statistics.

``umeyama`` is the closed-form least-squares similarity between two point sets
(Umeyama 1991). ``ransac_umeyama`` wraps it for GPS priors that contain the odd
bad fix. Neither needs pycolmap, so they are unit-testable on synthetic data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Sim3:
    scale: float
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3

    def apply(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return self.scale * pts @ self.rotation.T + self.translation

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m


def umeyama(src, dst, with_scale: bool = True) -> Sim3:
    """Find s, R, t minimising sum ||dst - (s R src + t)||^2."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need matching (N>=3, 3) point sets")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1.0
    r = u @ s_fix @ vt
    var_s = (xs**2).sum() / len(src)
    scale = float(np.trace(np.diag(d) @ s_fix) / var_s) if with_scale else 1.0
    t = mu_d - scale * r @ mu_s
    return Sim3(scale, r, t)


def ransac_umeyama(src, dst, threshold: float, iterations: int = 500, seed: int = 0) -> tuple[Sim3, np.ndarray]:
    """Robust Sim3: sample triplets, keep the largest inlier set, refit on it."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best = np.zeros(len(src), dtype=bool)
    for _ in range(iterations):
        idx = rng.choice(len(src), 3, replace=False)
        try:
            model = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        inliers = np.linalg.norm(model.apply(src) - dst, axis=1) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        raise RuntimeError("RANSAC found fewer than 3 inliers")
    return umeyama(src[best], dst[best]), best


def residual_stats(values) -> dict[str, float]:
    v = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "rmse_m": math.sqrt(float(np.mean(v**2))),
        "median_m": float(np.median(v)),
        "p90_m": float(np.percentile(v, 90)),
        "max_m": float(np.max(v)),
        "n": int(len(v)),
    }


def residual_report(predicted, observed) -> dict[str, dict[str, float]]:
    """3D / horizontal / vertical stats for ENU residuals (predicted - observed)."""
    d = np.asarray(predicted, dtype=np.float64) - np.asarray(observed, dtype=np.float64)
    return {
        "three_d": residual_stats(np.linalg.norm(d, axis=1)),
        "horizontal": residual_stats(np.linalg.norm(d[:, :2], axis=1)),
        "vertical": residual_stats(d[:, 2]),
    }
