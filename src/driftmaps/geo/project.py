"""Local map projection: WGS84 lon/lat <-> metres east/north around a route origin.

A transverse Mercator (scale factor 1) whose central meridian and origin latitude are the route
origin, as chosen in docs/research/geo.md §3.1 (`+proj=tmerc +lat_0 +lon_0 +k=1 +ellps=WGS84`).
The scale error is about x^2 / (2 R^2): 5 ppm 20 km from the origin.

The maths is Krueger's series to sixth order in the third flattening n (Karney, "Transverse
Mercator with an accuracy of a few nanometers", J. Geodesy 85, 2011, eqs. 7, 9, 35-36), so no
pyproj is needed. It is accurate to well under a millimetre within a few thousand km of the
central meridian; tests check it against the quarter meridian and a numerically integrated
meridian arc.

`lonlat_to_ac` chains this with `driftmaps.ac.coords.enu_to_ac` (AC = (E, U, -N)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from driftmaps.ac.coords import enu_to_ac

__all__ = ["WGS84_A", "WGS84_F", "LocalProjection"]

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563

_N = WGS84_F / (2.0 - WGS84_F)  # third flattening
_E = np.sqrt(WGS84_F * (2.0 - WGS84_F))  # first eccentricity
# Rectifying radius: meridian arc length = _A_RECT * rectifying latitude.
_A_RECT = WGS84_A / (1.0 + _N) * (1.0 + _N**2 / 4.0 + _N**4 / 64.0 + _N**6 / 256.0)


def _series(coeffs: list[list[float]]) -> np.ndarray:
    """Evaluate Krueger coefficient polynomials in n: coeffs[j][p] multiplies n^(p+1)."""
    return np.array([sum(c * _N ** (p + 1) for p, c in enumerate(row)) for row in coeffs])


# alpha_j (forward) and beta_j (reverse), Karney 2011 eqs. 35 and 36.
_ALPHA = _series(
    [
        [1 / 2, -2 / 3, 5 / 16, 41 / 180, -127 / 288, 7891 / 37800],
        [0, 13 / 48, -3 / 5, 557 / 1440, 281 / 630, -1983433 / 1935360],
        [0, 0, 61 / 240, -103 / 140, 15061 / 26880, 167603 / 181440],
        [0, 0, 0, 49561 / 161280, -179 / 168, 6601661 / 7257600],
        [0, 0, 0, 0, 34729 / 80640, -3418889 / 1995840],
        [0, 0, 0, 0, 0, 212378941 / 319334400],
    ]
)
_BETA = _series(
    [
        [1 / 2, -2 / 3, 37 / 96, -1 / 360, -81 / 512, 96199 / 604800],
        [0, 1 / 48, 1 / 15, -437 / 1440, 46 / 105, -1118711 / 3870720],
        [0, 0, 17 / 480, -37 / 840, -209 / 4480, 5569 / 90720],
        [0, 0, 0, 4397 / 161280, -11 / 504, -830251 / 7257600],
        [0, 0, 0, 0, 4583 / 161280, -108847 / 3991680],
        [0, 0, 0, 0, 0, 20648693 / 638668800],
    ]
)
_J2 = 2.0 * np.arange(1, 7)  # 2j for j = 1..6


def _tm_forward(dlon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radians (longitude from the central meridian, latitude) -> TM (x, y) metres, k0 = 1."""
    sin_lat = np.sin(lat)
    # tan of the conformal latitude.
    t = np.sinh(np.arctanh(sin_lat) - _E * np.arctanh(_E * sin_lat))
    xi_p = np.arctan2(t, np.cos(dlon))
    eta_p = np.arctanh(np.sin(dlon) / np.sqrt(1.0 + t * t))
    a2 = _J2 * xi_p[..., None]
    b2 = _J2 * eta_p[..., None]
    xi = xi_p + np.sum(_ALPHA * np.sin(a2) * np.cosh(b2), axis=-1)
    eta = eta_p + np.sum(_ALPHA * np.cos(a2) * np.sinh(b2), axis=-1)
    return _A_RECT * eta, _A_RECT * xi


def _tm_reverse(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TM (x, y) metres -> radians (longitude from the central meridian, latitude)."""
    xi = y / _A_RECT
    eta = x / _A_RECT
    a2 = _J2 * xi[..., None]
    b2 = _J2 * eta[..., None]
    xi_p = xi - np.sum(_BETA * np.sin(a2) * np.cosh(b2), axis=-1)
    eta_p = eta - np.sum(_BETA * np.cos(a2) * np.sinh(b2), axis=-1)
    chi = np.arcsin(np.clip(np.sin(xi_p) / np.cosh(eta_p), -1.0, 1.0))  # conformal latitude
    dlon = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    # Conformal -> geodetic latitude by fixed-point iteration (converges ~e^2 per step).
    lat = chi.copy()
    q = np.tan(np.pi / 4.0 + chi / 2.0)
    for _ in range(8):
        es = _E * np.sin(lat)
        lat = 2.0 * np.arctan(q * ((1.0 + es) / (1.0 - es)) ** (_E / 2.0)) - np.pi / 2.0
    return dlon, lat


@dataclass(frozen=True)
class LocalProjection:
    """Transverse Mercator centred on (lat0, lon0): the origin maps to (0, 0).

    All methods are vectorised: pass scalars or arrays of matching shape. East/north are metres
    on the local grid (grid north = true north only along the central meridian; the grid
    convergence is about dlon * sin(lat), a fraction of a degree within 20 km).
    """

    lat0: float
    lon0: float
    _y0: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (-90.0 < self.lat0 < 90.0 and -180.0 <= self.lon0 <= 180.0):
            raise ValueError(f"origin out of range: lat={self.lat0}, lon={self.lon0}")
        _, y0 = _tm_forward(np.zeros(1), np.radians(np.array([self.lat0], dtype=np.float64)))
        object.__setattr__(self, "_y0", float(y0[0]))

    def to_enu(self, lon: npt.ArrayLike, lat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Degrees lon/lat -> (east, north) metres from the origin."""
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        dlon = np.radians((lon - self.lon0 + 180.0) % 360.0 - 180.0)
        x, y = _tm_forward(dlon, np.radians(lat))
        return x, y - self._y0

    def to_lonlat(self, east: npt.ArrayLike, north: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """(east, north) metres from the origin -> degrees (lon, lat)."""
        east = np.asarray(east, dtype=np.float64)
        north = np.asarray(north, dtype=np.float64)
        dlon, lat = _tm_reverse(east, north + self._y0)
        lon = (self.lon0 + np.degrees(dlon) + 180.0) % 360.0 - 180.0
        return lon, np.degrees(lat)

    def to_xy(self, lonlat: npt.ArrayLike) -> np.ndarray:
        """(..., 2) lon/lat degrees -> (..., 2) east/north metres."""
        ll = np.asarray(lonlat, dtype=np.float64)
        e, n = self.to_enu(ll[..., 0], ll[..., 1])
        return np.stack([e, n], axis=-1)

    def from_xy(self, xy: npt.ArrayLike) -> np.ndarray:
        """(..., 2) east/north metres -> (..., 2) lon/lat degrees."""
        xy = np.asarray(xy, dtype=np.float64)
        lon, lat = self.to_lonlat(xy[..., 0], xy[..., 1])
        return np.stack([lon, lat], axis=-1)

    def lonlat_to_ac(
        self, lon: npt.ArrayLike, lat: npt.ArrayLike, up: npt.ArrayLike = 0.0
    ) -> np.ndarray:
        """Degrees lon/lat + height (metres, already relative to the track datum) -> AC (..., 3)."""
        e, n = self.to_enu(lon, lat)
        u = np.broadcast_to(np.asarray(up, dtype=np.float64), e.shape)
        return enu_to_ac(np.stack([e, n, u], axis=-1))
