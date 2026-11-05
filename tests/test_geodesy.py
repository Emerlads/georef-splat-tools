import math

import numpy as np
import pytest

from georef_splat.geodesy import (
    ecef_to_geodetic,
    enu_to_ecef_matrix,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
)


def test_equator_prime_meridian_is_semi_major_axis():
    np.testing.assert_allclose(geodetic_to_ecef(0, 0, 0), [6378137.0, 0, 0], atol=1e-6)


def test_north_pole_is_semi_minor_axis():
    np.testing.assert_allclose(geodetic_to_ecef(90, 0, 0), [0, 0, 6356752.314245], atol=1e-3)


@pytest.mark.parametrize("lat,lon,h", [(41.3038, -81.7505, 346.3), (-33.9, 151.2, -20.0), (0.0, 179.9, 8848.0), (60.0, 10.0, 0.0)])
def test_ecef_roundtrip(lat, lon, h):
    la, lo, hh = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert la == pytest.approx(lat, abs=1e-10)
    assert lo == pytest.approx(lon, abs=1e-10)
    assert hh == pytest.approx(h, abs=1e-4)


def test_enu_axes_point_the_right_way():
    origin = (41.3, -81.75, 300.0)
    north = geodetic_to_enu(41.3 + 0.001, -81.75, 300.0, origin)
    east = geodetic_to_enu(41.3, -81.75 + 0.001, 300.0, origin)
    up = geodetic_to_enu(41.3, -81.75, 310.0, origin)
    assert north[1] == pytest.approx(111.06, abs=0.1) and abs(north[0]) < 1e-6
    assert east[0] == pytest.approx(83.6, abs=0.2) and abs(east[1]) < 1e-3
    np.testing.assert_allclose(up, [0, 0, 10.0], atol=1e-6)


def test_enu_roundtrip_and_matrix_agree():
    origin = (41.3038, -81.7505, 312.0)
    enu = np.array([123.4, -56.7, 8.9])
    lat, lon, h = enu_to_geodetic(enu, origin)
    np.testing.assert_allclose(geodetic_to_enu(lat, lon, h, origin), enu, atol=1e-6)
    m = enu_to_ecef_matrix(origin)
    np.testing.assert_allclose((m @ np.r_[enu, 1])[:3], geodetic_to_ecef(lat, lon, h), atol=1e-6)


def test_vertical_datum_shift_is_pure_up_translation():
    """Changing only the origin height moves the whole model straight up/down."""
    a = enu_to_ecef_matrix((41.3, -81.75, 312.0))
    b = enu_to_ecef_matrix((41.3, -81.75, 312.0 - 33.9))
    np.testing.assert_allclose(a[:3, :3], b[:3, :3])
    assert np.linalg.norm(a[:3, 3] - b[:3, 3]) == pytest.approx(33.9, abs=1e-6)
    assert math.isclose(float(a[:3, 2] @ (a[:3, 3] - b[:3, 3])), 33.9, abs_tol=1e-6)
