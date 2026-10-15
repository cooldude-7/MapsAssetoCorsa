"""Road height profile: raw DEM samples along the road -> a smooth, driveable z(s).

Pipeline (docs/research/geo.md §2.4, PLAN §5.2):

1. Robust cross-road median of the samples taken across the road at each station.
2. Stations with weight 0 (bridges, tunnels, NoData) are not fitted at all: the road runs
   STRAIGHT (constant grade) across each such run, between the nearest weighted stations.
3. Robust asymmetric Whittaker smoother: minimise
       sum_i c_i (z_i - y_i)^2 + lam * sum_i e_i (z''_i)^2
   with c_i = weight x station length and z'' the second divided difference, re-weighting a few
   times. Points above the fit get weight 2p and points below 2(1-p) (p = 0.1 for surface
   models such as GLO-30, so the fit hugs the lower envelope and rejects trees and buildings),
   times Tukey bisquare weights that drop outliers in either direction. lam sets the cut-off
   wavelength L (half-power): lam = (L / 2 pi)^4; L grows with the DEM's pixel size because
   lidar DTMs need far less smoothing than 30 m DSMs (`cutoff_wavelength_m`).
4. The same objective (final weights) under hard limits |z'| <= max_grade and
   |z''| <= max_vertical_curvature, solved by a small primal-dual interior-point method on the
   banded system (scipy only; osqp isn't a dependency). Every iterate stays strictly inside
   the limits, so the returned profile satisfies them up to floating-point rounding.

Stations can be unevenly spaced; all derivatives are divided differences in metres.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded

__all__ = [
    "cross_road_median",
    "cutoff_wavelength_m",
    "road_profile",
    "whittaker_smooth",
]

FloatArray = npt.NDArray[np.float64]

DEFAULT_MAX_GRADE = 0.20
# R_v,min ~ 150 m (sag) / 200 m (crest) for mountain roads (geo.md §2.4, ASSUMED); one limit
# for both keeps the API simple: 1/150 1/m.
DEFAULT_MAX_VERTICAL_CURVATURE = 1.0 / 150.0
# A source this coarse or coarser is treated as a surface model (lower-envelope fit).
DSM_RESOLUTION_M = 20.0


def cutoff_wavelength_m(resolution_m: float) -> float:
    """Smoother cut-off wavelength for a DEM pixel size: ~45 m for 1 m lidar, ~80 m for 10 m,
    ~200 m for 30 m GLO-30 (geo.md §2.4 starting values, ASSUMED; tune on the test routes)."""
    r = float(np.clip(resolution_m, 1.0, 90.0))
    anchors_res, anchors_cutoff = [1.0, 10.0, 30.0, 90.0], [45.0, 80.0, 200.0, 400.0]
    return float(np.interp(math.log(r), np.log(anchors_res), anchors_cutoff))


def cross_road_median(z_samples: npt.ArrayLike) -> FloatArray:
    """(N,) or (N, K) samples -> (N,) robust per-station height (NaN-aware median; NaN where a
    station has no finite sample)."""
    z = np.asarray(z_samples, np.float64)
    if z.ndim == 1:
        return z.copy()
    if z.ndim != 2:
        raise ValueError(f"z_samples must be (N,) or (N, K), got {z.shape}")
    out = np.full(z.shape[0], np.nan)
    ok = np.isfinite(z).any(axis=1)
    if ok.any():
        out[ok] = np.nanmedian(z[ok], axis=1)
    return out


# ---------------------------------------------------------------------------------------------
# Banded linear algebra
# ---------------------------------------------------------------------------------------------


class _BandedSPD:
    """Cholesky factor of a sparse symmetric positive-definite banded matrix."""

    def __init__(self, m: sparse.sparray) -> None:
        m = sparse.csr_array(m)
        n = m.shape[0]
        coo = m.tocoo()
        bw = int(np.abs(coo.row - coo.col).max()) if coo.nnz else 0
        ab = np.zeros((bw + 1, n))
        for k in range(bw + 1):
            ab[bw - k, k:] = m.diagonal(k)
        self.factor = (cholesky_banded(ab, lower=False), False)

    def solve(self, rhs: FloatArray) -> FloatArray:
        return cho_solve_banded(self.factor, rhs)


def _operators(s: FloatArray, kept: npt.NDArray[np.intp]):
    """Divided-difference operators on the kept stations, with straight runs between them.

    Returns (D1, D2, e): D1 (nK-1, nK) grades between consecutive kept stations; D2 (nK-2, nK)
    the curvature at each interior kept station, which is the kink between its two straight
    neighbours spread over the station's own fine spacing (so a kink at a bridge end is judged
    like any other); e (nK-2,) the station lengths that weight the curvature penalty.
    """
    sk = s[kept]
    nk = sk.size
    big_h = np.diff(sk)
    d1 = sparse.diags_array(
        [-1.0 / big_h, 1.0 / big_h], offsets=[0, 1], shape=(nk - 1, nk), format="csr"
    )
    if nk < 3:
        return d1, sparse.csr_array((0, nk)), np.zeros(0)
    h = np.diff(s)
    i = kept[1:-1]
    hf = h[i - 1] + h[i]  # fine spacing around each interior kept station
    a, b = 1.0 / big_h[:-1], 1.0 / big_h[1:]
    k = 2.0 / hf
    d2 = sparse.diags_array(
        [k * a, -k * (a + b), k * b], offsets=[0, 1, 2], shape=(nk - 2, nk), format="csr"
    )
    return d1, d2, hf / 2.0


def _penalised_matrix(c: FloatArray, lam: float, d2: sparse.sparray, e: FloatArray):
    ridge = 1e-9 * max(float(c.max(initial=0.0)), 1e-12)
    p = sparse.diags_array(c + ridge) + lam * (d2.T @ sparse.diags_array(e) @ d2)
    return sparse.csr_array(p)


def whittaker_smooth(
    y: FloatArray, c: FloatArray, lam: float, d2: sparse.sparray, e: FloatArray
) -> FloatArray:
    """argmin sum c (z - y)^2 + lam sum e (D2 z)^2 (one banded solve)."""
    return _BandedSPD(_penalised_matrix(c, lam, d2, e)).solve(c * y)


# ---------------------------------------------------------------------------------------------
# Box-constrained QP: min 1/2 x'Px + q'x  s.t.  -1 <= A x <= 1
# ---------------------------------------------------------------------------------------------


def _max_step(v: FloatArray, dv: FloatArray) -> float:
    neg = dv < 0
    if not neg.any():
        return 1.0
    return float(min(1.0, np.min(-v[neg] / dv[neg])))


def _newton(fac, a, at, rd, ru, rl, su, sl, lu, ll, rcu, rcl):
    """One Newton direction (dx, dsu, dsl, dlu, dll) for complementarity targets rcu, rcl."""
    tu = (rcu + lu * ru) / su
    tl = (rcl + ll * rl) / sl
    dx = fac.solve(-rd - at @ (tu - tl))
    adx = a @ dx
    dsu, dsl = -ru - adx, -rl + adx
    return dx, dsu, dsl, (rcu - lu * dsu) / su, (rcl - ll * dsl) / sl


def _box_qp(
    p: sparse.sparray,
    q: FloatArray,
    a: sparse.sparray,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> FloatArray:
    """Mehrotra predictor-corrector interior point from x = 0 (strictly feasible because every
    row of A is a difference operator, so A 0 = 0). Slacks stay positive and the primal
    residual stays zero, so every iterate satisfies the box strictly."""
    n, m = p.shape[0], a.shape[0]
    if m == 0:
        return _BandedSPD(p).solve(-q)
    a = sparse.csr_array(a)
    at = sparse.csr_array(a.T)
    x = np.zeros(n)
    su, sl = np.ones(m), np.ones(m)  # 1 - A x, 1 + A x
    lu, ll = np.ones(m), np.ones(m)
    qn = 1.0 + float(np.abs(q).max(initial=0.0))
    converged = False
    for _ in range(max_iter):
        ax = a @ x
        ru = ax + su - 1.0
        rl = -ax + sl - 1.0
        rd = p @ x + q + at @ (lu - ll)
        mu = (su @ lu + sl @ ll) / (2 * m)
        if mu < tol and np.abs(rd).max() < tol * qn:
            converged = True
            break
        fac = _BandedSPD(p + at @ sparse.diags_array(lu / su + ll / sl) @ a)
        state = (fac, a, at, rd, ru, rl, su, sl, lu, ll)
        _, dsu, dsl, dlu, dll = _newton(*state, -su * lu, -sl * ll)  # predictor
        alpha = min(
            _max_step(su, dsu), _max_step(sl, dsl), _max_step(lu, dlu), _max_step(ll, dll)
        )
        mu_aff = ((su + alpha * dsu) @ (lu + alpha * dlu)
                  + (sl + alpha * dsl) @ (ll + alpha * dll)) / (2 * m)
        sigma = (mu_aff / mu) ** 3
        dx, dsu2, dsl2, dlu2, dll2 = _newton(  # corrector
            *state, -su * lu - dsu * dlu + sigma * mu, -sl * ll - dsl * dll + sigma * mu
        )
        alpha = 0.99 * min(
            _max_step(su, dsu2), _max_step(sl, dsl2), _max_step(lu, dlu2), _max_step(ll, dll2)
        )
        alpha = min(alpha, 1.0)
        x += alpha * dx
        su += alpha * dsu2
        sl += alpha * dsl2
        lu += alpha * dlu2
        ll += alpha * dll2
    if not converged:
        warnings.warn("road profile QP stopped before converging; limits still hold",
                      stacklevel=3)
    return x


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------


def road_profile(
    s: npt.ArrayLike,
    z_samples: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    source_resolution_m: float,
    max_grade: float = DEFAULT_MAX_GRADE,
    max_vertical_curvature: float = DEFAULT_MAX_VERTICAL_CURVATURE,
    cutoff_m: float | None = None,
    asymmetry: float | None = None,
    iterations: int = 6,
) -> FloatArray:
    """Driveable road heights z (N,) at stations `s` (N,) metres along the road.

    - `z_samples`: (N,) or (N, K) raw DEM heights (K samples across the road); NaN = NoData.
    - `weights`: (N,) data weights, default 1; 0 on bridges/tunnels (straight across), e.g.
      0.3 on cuttings for coarse data. NaN stations get weight 0 automatically.
    - `source_resolution_m`: DEM pixel size; sets the smoothing (`cutoff_wavelength_m`) unless
      `cutoff_m` is given, and the default `asymmetry`.
    - `max_grade` (rise/run) and `max_vertical_curvature` (1/m, = 1/R_v) are hard limits.
    - `asymmetry` p in (0, 1): weight of points above the fit relative to below; default 0.1
      for coarse surface models (>= 20 m), 0.5 (symmetric) for lidar-class DTMs.
    """
    s = np.asarray(s, np.float64)
    y = cross_road_median(z_samples)
    n = s.size
    if s.ndim != 1 or y.shape != (n,):
        raise ValueError(f"s must be (N,) and z_samples (N,) or (N, K); got {s.shape}, {y.shape}")
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64).copy()
    if w.shape != (n,):
        raise ValueError(f"weights must be (N,) = ({n},), got {w.shape}")
    if not np.isfinite(s).all() or (n > 1 and (np.diff(s) <= 0).any()):
        raise ValueError("s must be finite and strictly increasing")
    if not (max_grade > 0 and max_vertical_curvature > 0):
        raise ValueError("max_grade and max_vertical_curvature must be positive")
    if n == 0:
        return np.zeros(0)
    w = np.where(np.isfinite(w) & np.isfinite(y), np.maximum(w, 0.0), 0.0)
    if not (w > 0).any():
        raise ValueError("no usable elevation samples (all weights zero or NaN)")
    if n == 1:
        return y.copy()

    # Unknowns: heights at the weighted stations plus both ends; straight lines in between.
    kept_mask = w > 0
    kept_mask[[0, -1]] = True
    kept = np.flatnonzero(kept_mask)
    sk = s[kept]
    h = np.diff(s)
    station_len = np.zeros(n)
    station_len[:-1] += h / 2
    station_len[1:] += h / 2
    c0 = (w * station_len)[kept]
    yk = np.where(c0 > 0, y[kept], 0.0)
    y_ref = float(np.sum(c0 * yk) / np.sum(c0))
    yk = np.where(c0 > 0, yk - y_ref, 0.0)  # centred: x = 0 is a level road at y_ref

    d1, d2, e = _operators(s, kept)
    cutoff = cutoff_wavelength_m(source_resolution_m) if cutoff_m is None else float(cutoff_m)
    lam = (cutoff / (2.0 * math.pi)) ** 4
    if asymmetry is None:
        asymmetry = 0.1 if source_resolution_m >= DSM_RESOLUTION_M else 0.5
    if not 0.0 < asymmetry < 1.0:
        raise ValueError("asymmetry must be in (0, 1)")
    sigma_floor = 0.05 + 0.01 * source_resolution_m

    # Robust asymmetric Whittaker: a few re-weighted solves.
    active = c0 > 0
    rw = np.ones_like(c0)
    for _ in range(max(1, iterations)):
        zk = whittaker_smooth(yk, c0 * rw, lam, d2, e)
        r = yk - zk
        asym = np.where(r > 0, 2.0 * asymmetry, 2.0 * (1.0 - asymmetry))
        sigma = max(1.4826 * float(np.median(np.abs(r[active]))), sigma_floor)
        u = r / (4.685 * sigma)
        bisq = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        new = np.maximum(asym * bisq, 1e-3)
        done = np.allclose(new, rw, atol=1e-3)
        rw = new
        if done:
            break

    # Hard limits: same objective, rows of A scaled so the box is [-1, 1].
    c = c0 * rw
    pmat = 2.0 * _penalised_matrix(c, lam, d2, e)
    q = -2.0 * c * yk
    scale = float(pmat.diagonal().mean())
    a = sparse.vstack([d1 / max_grade, d2 / max_vertical_curvature], format="csr")
    xk = _box_qp(pmat / scale, q / scale, a)
    return np.interp(s, sk, xk + y_ref)
