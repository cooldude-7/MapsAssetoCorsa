"""Coordinate conventions for Assetto Corsa.

AC world space is Y-up and right-handed with X = east, Y = up, Z = south (so north = -Z).
Getting this wrong mirrors the track and stops left/right timing gates firing
(docs/research/ac_format.md §4.4). In-game spike S4 re-checks it.

A node's local frame (dummy transform rows) is: +Z = forward, +Y = up, +X = Y x Z. For a car
facing along +Z that puts +X on the driver's LEFT, so `left = up x forward`.
"""

from __future__ import annotations

import numpy as np

WORLD_UP = np.array([0.0, 1.0, 0.0])


def enu_to_ac(enu: np.ndarray) -> np.ndarray:
    """Local east/north/up metres (..., 3) -> AC world (..., 3): (E, U, -N)."""
    enu = np.asarray(enu, dtype=np.float64)
    return np.stack([enu[..., 0], enu[..., 2], -enu[..., 1]], axis=-1)


def ac_to_enu(xyz: np.ndarray) -> np.ndarray:
    """AC world (..., 3) -> local east/north/up (..., 3)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.stack([xyz[..., 0], -xyz[..., 2], xyz[..., 1]], axis=-1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def left_of(forward: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """Unit vector pointing to the driver's left for a car heading along `forward`."""
    return _unit(np.cross(np.asarray(up, float), np.asarray(forward, float)))


def frame_transform(
    position: np.ndarray, forward: np.ndarray, up: np.ndarray = WORLD_UP
) -> np.ndarray:
    """4x4 dummy transform placing a node at `position` facing `forward`.

    Rows: X = up x forward (driver's left), Y = up (re-orthogonalised), Z = forward,
    row 3 = translation. This is the DirectX row-vector layout KN5 stores.
    """
    z = _unit(np.asarray(forward, float))
    x = _unit(np.cross(np.asarray(up, float), z))
    y = np.cross(z, x)
    m = np.eye(4)
    m[0, :3] = x
    m[1, :3] = y
    m[2, :3] = z
    m[3, :3] = np.asarray(position, float)
    return m.astype(np.float32)


def translation_transform(position: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[3, :3] = np.asarray(position, float)
    return m.astype(np.float32)
