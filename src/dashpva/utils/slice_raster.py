"""Rasterize a scattered 3D slice slab onto a regular 2D grid.

Single source of truth shared by the Slice 2D display tab and the HDF5 save
path so the displayed image is identical to the saved one. Given the points and
intensities of a thin slab extracted near a plane, this projects them into the
plane's in-plane basis (U, V) and averages intensities into an HxW image, while
also producing per-pixel HKL (qx/qy/qz) grids at the bin centers.

Example:
    >>> import numpy as np
    >>> from dashpva.utils.slice_raster import rasterize_slab
    >>> pts = np.random.rand(1000, 3)
    >>> vals = np.random.rand(1000)
    >>> out = rasterize_slab(pts, vals, normal=[0, 0, 1], origin=[0.5, 0.5, 0.5], shape=(64, 64))
    >>> out["image"].shape, out["qx"].shape, out["orientation"]
    ((64, 64), (64, 64), 'HK')
"""

from typing import Optional, Tuple

import numpy as np

_TOL = 0.95  # dot-product threshold to call a normal axis-aligned


def infer_orientation(normal: np.ndarray) -> Tuple[str, Optional[Tuple[int, int]], Optional[str]]:
    """Infer slice orientation from the plane normal in HKL coordinates.

    Returns (orientation, (u_idx, v_idx) or None, orth_label) where u_idx/v_idx
    index columns of the points array (0:H, 1:K, 2:L). orientation is one of
    'HK', 'KL', 'HL', 'Custom'; orth_label is the axis perpendicular to the plane.
    """
    n = np.array(normal, dtype=float).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if not np.isfinite(n_norm) or n_norm <= 0.0:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n = n / n_norm
    dX = abs(float(n[0]))
    dY = abs(float(n[1]))
    dZ = abs(float(n[2]))
    if dZ >= _TOL:
        return "HK", (0, 1), "L"
    if dX >= _TOL:
        return "KL", (1, 2), "H"
    if dY >= _TOL:
        return "HL", (0, 2), "K"
    return "Custom", None, None


def _plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal in-plane basis (u, v) for a normalized normal n."""
    world_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    ref = world_axes[0]
    for ax in world_axes:
        if abs(float(np.dot(ax, n))) < 0.9:
            ref = ax
            break
    u = np.cross(n, ref)
    u_norm = float(np.linalg.norm(u))
    if not np.isfinite(u_norm) or u_norm <= 0.0:
        ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref)
        u_norm = float(np.linalg.norm(u))
        if not np.isfinite(u_norm) or u_norm <= 0.0:
            u = np.array([1.0, 0.0, 0.0])
            u_norm = 1.0
    u = u / u_norm
    v = np.cross(n, u)
    v_norm = float(np.linalg.norm(v))
    if not np.isfinite(v_norm) or v_norm <= 0.0:
        v = np.array([0.0, 1.0, 0.0])
    return u, v


def rasterize_slab(
    points: np.ndarray,
    intensities: np.ndarray,
    normal,
    origin,
    shape: Optional[Tuple[int, int]] = None,
) -> Optional[dict]:
    """Rasterize slab points into a 2D image plus per-pixel HKL grids.

    Args:
        points: (N, 3) slab points in HKL space.
        intensities: (N,) intensity per point.
        normal: plane normal (3,).
        origin: plane origin (3,).
        shape: desired (H, W); defaults to (512, 512) when absent/invalid.

    Returns:
        dict with keys: image (H,W float32), qx/qy/qz (H,W float32),
        u_axis, v_axis (3,), u_range, v_range (2,), orientation (str),
        orth_label (str|None), orth_value (float|None). None if there is
        nothing to rasterize.
    """
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(intensities, dtype=float).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    if vals.size == 0 or vals.shape[0] != pts.shape[0]:
        return None

    n = np.array(normal, dtype=float).reshape(3)
    n_norm = float(np.linalg.norm(n))
    n = np.array([0.0, 0.0, 1.0]) if (not np.isfinite(n_norm) or n_norm <= 0.0) else n / n_norm
    o = np.array(origin, dtype=float).reshape(3)
    if not np.all(np.isfinite(o)):
        o = np.mean(pts, axis=0)

    # Target raster shape
    H = W = 512
    if shape and isinstance(shape, (tuple, list)) and len(shape) == 2:
        try:
            if int(shape[0]) > 0 and int(shape[1]) > 0:
                H, W = int(shape[0]), int(shape[1])
        except Exception:
            pass

    orientation, uv_idxs, orth_label = infer_orientation(n)

    if uv_idxs is not None:
        # Axis-aligned: use absolute HKL coordinates directly.
        u_idx, v_idx = uv_idxs
        U = pts[:, u_idx]
        V = pts[:, v_idx]
        u_axis = np.zeros(3)
        v_axis = np.zeros(3)
        u_axis[u_idx] = 1.0
        v_axis[v_idx] = 1.0
        orth_value = float(o[{"H": 0, "K": 1, "L": 2}[orth_label]]) if orth_label else None
    else:
        # Custom: project onto in-plane basis relative to origin.
        u_axis, v_axis = _plane_basis(n)
        rel = pts - o[None, :]
        U = rel.dot(u_axis)
        V = rel.dot(v_axis)
        orth_value = float(np.dot(n, o))

    U_min, U_max = float(np.min(U)), float(np.max(U))
    V_min, V_max = float(np.min(V)), float(np.max(V))
    if not np.isfinite(U_min) or not np.isfinite(U_max) or U_max == U_min:
        U_min, U_max = -0.5, 0.5
    if not np.isfinite(V_min) or not np.isfinite(V_max) or V_max == V_min:
        V_min, V_max = -0.5, 0.5

    du = (U_max - U_min) / float(W)
    dv = (V_max - V_min) / float(H)
    if not np.isfinite(du) or du <= 0.0:
        du = 1.0 / float(max(W, 1))
    if not np.isfinite(dv) or dv <= 0.0:
        dv = 1.0 / float(max(H, 1))

    iu = np.clip(np.floor((U - U_min) / du).astype(int), 0, W - 1)
    iv = np.clip(np.floor((V - V_min) / dv).astype(int), 0, H - 1)

    image_sum = np.zeros((H, W), dtype=np.float64)
    image_cnt = np.zeros((H, W), dtype=np.int64)
    np.add.at(image_sum, (iv, iu), vals.astype(np.float64))
    np.add.at(image_cnt, (iv, iu), 1)
    image = np.zeros((H, W), dtype=np.float32)
    nz = image_cnt > 0
    image[nz] = (image_sum[nz] / image_cnt[nz]).astype(np.float32)

    # Per-pixel HKL at bin centers: q = o + U*u_axis + V*v_axis
    Uc = U_min + (np.arange(W, dtype=np.float64) + 0.5) * du
    Vc = V_min + (np.arange(H, dtype=np.float64) + 0.5) * dv
    U_grid = np.broadcast_to(Uc[None, :], (H, W))
    V_grid = np.broadcast_to(Vc[:, None], (H, W))
    if uv_idxs is not None:
        # Axis-aligned grids run along the two in-plane HKL axes; the orthogonal
        # component is fixed at the plane origin value.
        u_idx, v_idx = uv_idxs
        grids = [np.full((H, W), float(o[i]), dtype=np.float32) for i in range(3)]
        grids[u_idx] = U_grid.astype(np.float32)
        grids[v_idx] = V_grid.astype(np.float32)
        qx, qy, qz = grids
    else:
        qx = (o[0] + U_grid * u_axis[0] + V_grid * v_axis[0]).astype(np.float32)
        qy = (o[1] + U_grid * u_axis[1] + V_grid * v_axis[1]).astype(np.float32)
        qz = (o[2] + U_grid * u_axis[2] + V_grid * v_axis[2]).astype(np.float32)

    return {
        "image": image,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "u_axis": u_axis.astype(float),
        "v_axis": v_axis.astype(float),
        "u_range": (U_min, U_max),
        "v_range": (V_min, V_max),
        "orientation": orientation,
        "orth_label": orth_label,
        "orth_value": orth_value,
    }
