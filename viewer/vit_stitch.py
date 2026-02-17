"""
LiveStitch-style patching/stitching for vit:1:CSSI.
Produces five panels: transmission, diffraction, beam position, NN prediction, NN stitched.
Used by PVAReader when subscribing to vit:1:CSSI.
Works with or without PyTorch (numpy fallback when torch unavailable).
"""
from math import pi
import os
import numpy as np

try:
    import torch
    from torch import Tensor
    _USE_TORCH = True
except Exception as e:
    torch = None
    Tensor = None
    _USE_TORCH = False
    print(f"[vit_stitch] PyTorch not available ({e}); using numpy backend. Stitching will still work.")

try:
    import h5py
except ImportError:
    h5py = None

# Stitch pad and stream layout (match LiveStitch4 / LiveStitch6)
STREAM_SHAPE = (512, 256)
DIFF_ROWS, PATCH_ROWS = 256, 256
# Patch crop: 32 -> data[32:-32] (192x192); 66 -> data[66:-66] (124x124) for LiveStitch4
PAD = 32  # pad argument to place_patches_fourier_shift (crop after shift)


# ---------- NumPy backend (used when torch is not available) ----------
def _batch_put_np(image: np.ndarray, patches: np.ndarray, sy: np.ndarray, sx: np.ndarray, op: str) -> np.ndarray:
    h, w = image.shape[-2:]
    ph, pw = patches.shape[-2], patches.shape[-1]
    n = len(sy)
    sy = np.asarray(sy, dtype=np.int64)
    sx = np.asarray(sx, dtype=np.int64)
    # inds[i, r, c] = (sy[i] + r) * w + (sx[i] + c)
    rr = np.arange(ph, dtype=np.int64)[:, None, None]  # (ph, 1, 1)
    cc = np.arange(pw, dtype=np.int64)[None, :, None]  # (1, pw, 1)
    row = rr + sy[None, None, :]   # (ph, 1, n)
    col = cc + sx[None, None, :]   # (1, pw, n)
    inds = (row * w + col).transpose(2, 0, 1).ravel()  # (n, ph, pw) -> ravel
    vals = patches.ravel()
    out = image.ravel().copy()
    # Bounds check: avoid out-of-range indices (can cause black output or crashes)
    valid = (inds >= 0) & (inds < out.size)
    if not np.all(valid):
        inds = inds[valid]
        vals = vals[valid]
    if op == "add":
        np.add.at(out, inds, vals)
    else:
        out[inds] = vals
    return out.reshape(h, w)


def _fourier_shift_np(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    ft = np.fft.fft2(images.astype(np.complex128), norm=None)
    ny, nx = images.shape[-2], images.shape[-1]
    freq_y = np.fft.fftfreq(ny)
    freq_x = np.fft.fftfreq(nx)
    freq_yy, freq_xx = np.meshgrid(freq_y, freq_x, indexing="ij")
    # shifts (n, 2): [dy, dx]; expand to (n, 1, 1) for broadcast with (1, ny, nx)
    dy = np.asarray(shifts[:, 0], dtype=np.float64).reshape(-1, 1, 1)
    dx = np.asarray(shifts[:, 1], dtype=np.float64).reshape(-1, 1, 1)
    phase = -2j * pi * (freq_xx[None, :, :] * dx + freq_yy[None, :, :] * dy)
    mult = np.exp(phase)
    shifted = np.fft.ifft2(ft * mult, norm=None).real
    return shifted.astype(images.dtype)


def _place_patches_fourier_shift_np(
    image: np.ndarray,
    positions: np.ndarray,
    patches: np.ndarray,
    op: str,
    adjoint_mode: bool,
    pad: int,
) -> np.ndarray:
    if patches.ndim == 2:
        patches = np.broadcast_to(patches[None, :, :], (positions.shape[0], patches.shape[0], patches.shape[1])).copy()
    ph, pw = patches.shape[-2], patches.shape[-1]
    patch_padding = pad if adjoint_mode else -pad
    sys_float = positions[:, 0] - (ph - 1.0) / 2.0
    sxs_float = positions[:, 1] - (pw - 1.0) / 2.0
    sys = np.floor(sys_float).astype(np.int64) - patch_padding
    eys = sys + ph + 2 * patch_padding
    sxs = np.floor(sxs_float).astype(np.int64) - patch_padding
    exs = sxs + pw + 2 * patch_padding
    fractional_shifts = np.column_stack([sys_float - sys - patch_padding, sxs_float - sxs - patch_padding])
    pad_lengths = [
        max(int(-sxs.min()), 0),
        max(int(exs.max() - image.shape[1]), 0),
        max(int(-sys.min()), 0),
        max(int(eys.max() - image.shape[0]), 0),
    ]
    image = np.pad(image, ((pad_lengths[2], pad_lengths[3]), (pad_lengths[0], pad_lengths[1])), mode="constant", constant_values=0)
    sys = sys + pad_lengths[2]
    eys = eys + pad_lengths[2]
    sxs = sxs + pad_lengths[0]
    exs = exs + pad_lengths[0]
    if not np.allclose(fractional_shifts, 0, atol=1e-7):
        patches = _fourier_shift_np(patches, fractional_shifts)
    if not adjoint_mode:
        pads = abs(patch_padding)
        patches = patches[:, pads : patches.shape[1] - pads, pads : patches.shape[2] - pads]
    image = _batch_put_np(image, patches, sys, sxs, op=op)
    image = image[
        pad_lengths[2] : image.shape[0] - pad_lengths[3],
        pad_lengths[0] : image.shape[1] - pad_lengths[1],
    ]
    return image


# ---------- PyTorch backend (when available) ----------
if _USE_TORCH:
    def _batch_put(
        image: Tensor,
        patches: Tensor,
        sy: Tensor,
        sx: Tensor,
        op: str,
    ) -> Tensor:
        h, w = image.shape[-2:]
        patch_size = patches.shape[-2:]
        x = torch.arange(patch_size[1], device=sx.device)[None, :]
        y = torch.arange(patch_size[0], device=sy.device)[None, :]
        x = x.expand(len(sx), x.shape[1])
        y = y.expand(len(sy), y.shape[1])
        x = x + sx[:, None]
        y = y + sy[:, None]
        inds = (y * w).unsqueeze(-1) + x.unsqueeze(1)
        image = image.reshape(-1)
        patches_flattened = patches.reshape(-1)
        if op == "add":
            image.scatter_add_(0, inds.view(-1), patches_flattened)
        else:
            image.scatter_(0, inds.view(-1), patches_flattened)
        return image.reshape(h, w)

    def _fourier_shift(images: Tensor, shifts: Tensor) -> Tensor:
        ft = torch.fft.fft2(images.to(torch.complex128), norm=None)
        freq_y, freq_x = torch.meshgrid(
            torch.fft.fftfreq(images.shape[-2], device=images.device),
            torch.fft.fftfreq(images.shape[-1], device=images.device),
            indexing="ij",
        )
        freq_x = freq_x.to(ft.device).unsqueeze(0).expand(images.shape[0], -1, -1)
        freq_y = freq_y.to(ft.device).unsqueeze(0).expand(images.shape[0], -1, -1)
        mult = torch.exp(
            -2j * pi * (freq_x * shifts[:, 1].view(-1, 1, 1) + freq_y * shifts[:, 0].view(-1, 1, 1))
        )
        shifted = torch.fft.ifft2(ft * mult, norm=None).real
        return shifted.to(images.dtype)

    def _place_patches_fourier_shift(
        image: Tensor,
        positions: Tensor,
        patches: Tensor,
        op: str,
        adjoint_mode: bool,
        pad: int,
    ) -> Tensor:
        if patches.dim() == 2:
            patches = patches.unsqueeze(0).expand(positions.shape[0], -1, -1)
        ph, pw = patches.shape[-2], patches.shape[-1]
        patch_padding = pad if adjoint_mode else -pad
        sys_float = positions[:, 0] - (ph - 1.0) / 2.0
        sxs_float = positions[:, 1] - (pw - 1.0) / 2.0
        sys = sys_float.floor().int() - patch_padding
        eys = sys + ph + 2 * patch_padding
        sxs = sxs_float.floor().int() - patch_padding
        exs = sxs + pw + 2 * patch_padding
        fractional_shifts = torch.stack(
            [sys_float - sys - patch_padding, sxs_float - sxs - patch_padding], dim=-1
        )
        pad_lengths = [
            max(-sxs.min().item(), 0),
            max(exs.max().item() - image.shape[1], 0),
            max(-sys.min().item(), 0),
            max(eys.max().item() - image.shape[0], 0),
        ]
        image = torch.nn.functional.pad(image, pad_lengths)
        sys = sys + pad_lengths[2]
        eys = eys + pad_lengths[2]
        sxs = sxs + pad_lengths[0]
        exs = exs + pad_lengths[0]
        if not torch.allclose(fractional_shifts, torch.zeros_like(fractional_shifts), atol=1e-7):
            patches = _fourier_shift(patches, fractional_shifts)
        if not adjoint_mode:
            patches = patches[
                :,
                abs(patch_padding) : patches.shape[-2] - abs(patch_padding),
                abs(patch_padding) : patches.shape[-1] - abs(patch_padding),
            ]
        image = _batch_put(image, patches, sys, sxs, op=op)
        image = image[
            pad_lengths[2] : image.shape[0] - pad_lengths[3],
            pad_lengths[0] : image.shape[1] - pad_lengths[1],
        ]
        return image


def _parse_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# Default CSV path — match LiveStitch7.py reference (no header, col0=y_m, col1=x_m in meters).
DEFAULT_POSITIONS_CSV = "/home/beams/AILEENLUO/ptycho-vit/workspace/positions_10um.csv"


def _find_csv_path(env_path: str = "", explicit_path: str = None) -> str:
    """Return first existing path: explicit, env, then default CSV path."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    if env_path and os.path.isfile(env_path):
        return env_path
    if os.path.isfile(DEFAULT_POSITIONS_CSV):
        return DEFAULT_POSITIONS_CSV
    return env_path or explicit_path or ""


def _find_npz_path(env_path: str = "", explicit_path: str = None) -> str:
    """Return first existing path: explicit, env, then LiveStitch4-style fallbacks."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    if env_path and os.path.isfile(env_path):
        return env_path
    _dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    fallbacks = [
        os.path.join(cwd, "optimized_route.npz"),
        os.path.join(_dir, "..", "optimized_route.npz"),
        os.path.join(_dir, "..", "..", "optimized_route.npz"),
        "/home/beams/USER26ID/optimized_route.npz",
        "/home/beams25/USER26ID/optimized_route.npz",
        os.path.join(os.path.expanduser("~"), "USER26ID", "optimized_route.npz"),
        os.path.join(os.path.expanduser("~"), "optimized_route.npz"),
    ]
    for p in fallbacks:
        p = os.path.normpath(p)
        if os.path.isfile(p):
            return p
    return env_path or explicit_path or ""


class VitStitcher:
    """
    Stateful stitcher for vit:1:CSSI stream.
    Expects stream shape (512, 256): diff = stream[:256,:], data = stream[256:,:].
    Positions: from CSV (default) if VIT_STITCH_POSITIONS_CSV or default path exists, else npz, else HDF5.
    Returns five panels: transmission, diffraction, beam_position, nn_prediction, nn_stitched.
    LiveStitch7: patch_edge_crop=32 (data[32:-32]), id_offset=621356, CSV col0=y_m, col1=x_m.
    """

    def __init__(
        self,
        positions_hdf5_path: str = None,
        positions_npz_path: str = None,
        positions_csv_path: str = None,
        patch_edge_crop: int = None,
        center_crop_display: int = None,
    ):
        self._ready = False
        self._pos_x_pix = None
        self._pos_y_pix = None
        self._object_size = None
        self._pos_origin_coords = None
        self._pred_ph = None
        self._buffer = None
        self._fly_ny = None
        self._fly_nx = None
        self._accu_int = None  # (fly_ny, fly_nx) masked array for transmission panel
        self._curr_traj = None  # 1D flattened (fly_ny+sq_size)*(fly_nx+sq_size) for beam position
        self._sq_size = 5
        self._half_sq_size = 2
        self._patch_edge_crop = (
            patch_edge_crop
            if patch_edge_crop is not None
            else _parse_int_env("VIT_STITCH_PATCH_EDGE_CROP", 32)
        )
        self._center_crop_display = (
            center_crop_display
            if center_crop_display is not None
            else _parse_int_env("VIT_STITCH_CENTER_CROP", 150)
        )
        # Load the offset from environment (start-from-correct-place for spiral). Match LiveStitch7 default.
        self._id_offset = _parse_int_env("VIT_STITCH_ID_OFFSET", 621356)
        # Reset period set from number of positions when CSV/NPZ load (env override still applies). 0 = disabled.
        self._reset_period = _parse_int_env("VIT_STITCH_RESET_PERIOD", 0)

        csv_path = _find_csv_path(
            os.environ.get("VIT_STITCH_POSITIONS_CSV", ""),
            positions_csv_path,
        )
        if csv_path and os.path.isfile(csv_path):
            self._load_positions_csv(csv_path)
        else:
            npz_path = _find_npz_path(
                os.environ.get("VIT_STITCH_NPZ_PATH", ""),
                positions_npz_path,
            )
            if npz_path and os.path.isfile(npz_path):
                self._load_positions_npz(npz_path)
            else:
                path = positions_hdf5_path or os.environ.get("VIT_STITCH_POSITIONS_HDF5", "")
                if path and h5py is not None and os.path.isfile(path):
                    self._load_positions(path)
                else:
                    print(
                        "[VitStitcher] Positions not loaded — prediction and stitching panels will be black. "
                        "Set VIT_STITCH_POSITIONS_CSV to a CSV path (col0=y_m, col1=x_m in meters) or VIT_STITCH_NPZ_PATH."
                    )

    def _compute_canvas_size_from_positions(
        self, pos_x_pix: np.ndarray, pos_y_pix: np.ndarray
    ) -> tuple:
        """Compute (obj_h, obj_w) so that all positions fit with patch and margin.
        Works for both raster (arbitrary min/max) and spiral/circular (centered) scans.
        Origin remains at canvas center; nothing gets cropped."""
        patch_side = 256 - 2 * self._patch_edge_crop  # placed patch size
        margin = int(os.environ.get("VIT_STITCH_OBJECT_MARGIN", "256"))
        # Require: origin (obj/2) + pos in [patch_side/2, obj - patch_side/2] for both dims
        half = patch_side / 2.0
        extent_y = float(np.ceil(2 * max(np.max(pos_y_pix) + half, half - np.min(pos_y_pix))))
        extent_x = float(np.ceil(2 * max(np.max(pos_x_pix) + half, half - np.min(pos_x_pix))))
        obj_h = max(int(extent_y) + margin, margin)
        obj_w = max(int(extent_x) + margin, margin)
        return (obj_h, obj_w)

    def _load_positions_npz(self, path: str) -> None:
        """Load positions from npz. Supports:
        - 'x_pix', 'y_pix': pixel offsets from center (used as-is).
        - VIT_STITCH_NPZ_PIXEL_OFFSETS=1: treat 'x', 'y' as pixel offsets from center.
        - Else: 'x', 'y' in route units with step (step_m = spiral_step * -1e-6) and pixel_size → pixels (LiveStitch4.py convention; e.g. LiveStitch4.py 0.05).
        """
        try:
            data = np.load(path)
            use_pixel_offsets = os.environ.get("VIT_STITCH_NPZ_PIXEL_OFFSETS", "").strip().lower() in ("1", "true", "yes")
            if "x_pix" in data and "y_pix" in data:
                pos_x_pix = np.asarray(data["x_pix"], dtype=np.float64)
                pos_y_pix = np.asarray(data["y_pix"], dtype=np.float64)
                self._pos_x_pix = pos_x_pix
                self._pos_y_pix = pos_y_pix
                mode = "x_pix/y_pix"
            elif use_pixel_offsets:
                pos_x_pix = np.asarray(data["x"], dtype=np.float64)
                pos_y_pix = np.asarray(data["y"], dtype=np.float64)
                self._pos_x_pix = pos_x_pix
                self._pos_y_pix = pos_y_pix
                mode = "pixel offsets (VIT_STITCH_NPZ_PIXEL_OFFSETS=1)"
            else:
                step_raw = float(os.environ.get("VIT_STITCH_STEP", "0.05"))
                # LiveStitch4.py 0.05 → step in meters = 0.05 * -1e-6; accept either spiral_step (0.05) or step_m (-5e-8)
                step = (step_raw * -1e-6) if (0 < step_raw <= 100) else step_raw
                pixel_size = float(os.environ.get("VIT_STITCH_PIXEL_SIZE", "6.89e-9"))
                x_m = np.asarray(data["x"], dtype=np.float64) * step
                y_m = np.asarray(data["y"], dtype=np.float64) * step
                pos_x_pix = -x_m / pixel_size
                pos_y_pix = y_m / pixel_size
                self._pos_x_pix = np.asarray(pos_x_pix, dtype=np.float64)
                self._pos_y_pix = np.asarray(pos_y_pix, dtype=np.float64)
                mode = "meters (step/pixel_size)"

            # Object size: from env if both set, else from position range (raster or spiral; no crop)
            obj_h_env = os.environ.get("VIT_STITCH_OBJECT_H", "").strip()
            obj_w_env = os.environ.get("VIT_STITCH_OBJECT_W", "").strip()
            if obj_h_env.isdigit() and obj_w_env.isdigit():
                obj_h = int(obj_h_env)
                obj_w = int(obj_w_env)
            elif len(self._pos_x_pix) > 0 and len(self._pos_y_pix) > 0:
                obj_h, obj_w = self._compute_canvas_size_from_positions(
                    self._pos_x_pix, self._pos_y_pix
                )
            else:
                obj_h = 1161
                obj_w = 1161
            self._object_size = (obj_h, obj_w)
            origin = np.round(np.array(self._object_size, dtype=np.float64) / 2.0) + 0.5

            if _USE_TORCH:
                self._pos_origin_coords = (torch.tensor(self._object_size, dtype=torch.float32) / 2.0).round() + 0.5
                self._pred_ph = torch.zeros(self._object_size, dtype=torch.float32)
                self._buffer = torch.zeros(self._object_size, dtype=torch.float32)
            else:
                self._pos_origin_coords = origin.astype(np.float64)
                self._pred_ph = np.zeros(self._object_size, dtype=np.float32)
                self._buffer = np.zeros(self._object_size, dtype=np.float32)

            self._ready = True
            npos = len(self._pos_x_pix)
            if self._reset_period == 0 and npos > 0:
                self._reset_period = npos
            self._init_transmission_beam(npos)
            backend = "torch" if _USE_TORCH else "numpy"
            print(f"[VitStitcher] LiveStitch4 positions loaded from {path} — {npos} points, object {self._object_size}, backend={backend}, mode={mode}, reset_period={self._reset_period}")

            # Sanity check: first position after adding origin should be inside image
            if npos > 0:
                py0 = self._pos_y_pix[0] + origin[0]
                px0 = self._pos_x_pix[0] + origin[1]
                if py0 < -50 or py0 > obj_h + 50 or px0 < -50 or px0 > obj_w + 50:
                    print(
                        "[VitStitcher] WARNING: Positions are outside the image (first pos_after_origin=({:.1f},{:.1f}), image {}x{}). "
                        "Set VIT_STITCH_NPZ_PIXEL_OFFSETS=1 if your NPZ stores pixel offsets from center.".format(
                            py0, px0, obj_h, obj_w
                        )
                    )
        except Exception as e:
            print(f"[VitStitcher] Failed to load positions from npz {path}: {e}")

    def _init_transmission_beam(self, npos: int) -> None:
        """Initialize fly_ny, fly_nx, accu_int, curr_traj for transmission and beam position panels (LiveStitch7 style)."""
        fly_nx = _parse_int_env("VIT_STITCH_FLY_NX", 100)
        fly_ny = npos // fly_nx
        if fly_ny * fly_nx != npos:
            fly_ny = int(np.sqrt(npos))
            fly_nx = npos // fly_ny
            if fly_ny * fly_nx != npos:
                fly_nx = npos
                fly_ny = 1
        fly_ny_env = os.environ.get("VIT_STITCH_FLY_NY", "").strip()
        if fly_ny_env.isdigit():
            fly_ny = int(fly_ny_env)
            fly_nx = npos // fly_ny if fly_ny > 0 else fly_nx
        self._fly_nx = fly_nx
        self._fly_ny = fly_ny
        self._accu_int = np.ma.masked_array(
            data=np.zeros((fly_ny, fly_nx), dtype=np.float64),
            mask=np.ones((fly_ny, fly_nx), dtype=bool),
        )
        # curr_traj: 1D flattened (fly_ny+sq_size)*(fly_nx+sq_size) with cursor pattern at top-left (LiveStitch7)
        traj_shape = (self._fly_ny + self._sq_size, self._fly_nx + self._sq_size)
        self._curr_traj = np.zeros(traj_shape, dtype=np.float64)
        self._curr_traj[self._half_sq_size, : self._sq_size] = 1
        self._curr_traj[: self._sq_size, self._half_sq_size] = 1
        self._curr_traj = self._curr_traj.flatten()

    def _load_positions_csv(self, path: str) -> None:
        """Load positions from CSV: no header, col0=y_m, col1=x_m (meters). Converts to pixels via VIT_STITCH_PIXEL_SIZE.
        Object size (prediction/stitch canvas) is computed from position range so all positions fit with patch margin.
        Matches LiveStitch7 column order and sign convention."""
        try:
            data = np.loadtxt(path, delimiter=",", dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            y_m = np.asarray(data[:, 0], dtype=np.float64)
            x_m = np.asarray(data[:, 1], dtype=np.float64)
            pixel_size = float(os.environ.get("VIT_STITCH_PIXEL_SIZE", "6.89e-9"))
            pos_y_pix = y_m / pixel_size
            pos_x_pix = x_m / pixel_size
            self._pos_x_pix = pos_x_pix
            self._pos_y_pix = pos_y_pix
            # Object size: from env if set, else from position range (raster or spiral; no crop)
            obj_h_env = os.environ.get("VIT_STITCH_OBJECT_H", "")
            obj_w_env = os.environ.get("VIT_STITCH_OBJECT_W", "")
            if obj_h_env.isdigit() and obj_w_env.isdigit():
                obj_h = int(obj_h_env)
                obj_w = int(obj_w_env)
            elif len(pos_x_pix) > 0 and len(pos_y_pix) > 0:
                obj_h, obj_w = self._compute_canvas_size_from_positions(pos_x_pix, pos_y_pix)
            else:
                obj_h = 1161
                obj_w = 1161
            self._object_size = (obj_h, obj_w)
            origin = np.round(np.array(self._object_size, dtype=np.float64) / 2.0) + 0.5
            if _USE_TORCH:
                self._pos_origin_coords = (torch.tensor(self._object_size, dtype=torch.float32) / 2.0).round() + 0.5
                self._pred_ph = torch.zeros(self._object_size, dtype=torch.float32)
                self._buffer = torch.zeros(self._object_size, dtype=torch.float32)
            else:
                self._pos_origin_coords = origin.astype(np.float64)
                self._pred_ph = np.zeros(self._object_size, dtype=np.float32)
                self._buffer = np.zeros(self._object_size, dtype=np.float32)
            self._ready = True
            npos = len(self._pos_x_pix)
            if self._reset_period == 0 and npos > 0:
                self._reset_period = npos
            self._init_transmission_beam(npos)
            backend = "torch" if _USE_TORCH else "numpy"
            print(f"[VitStitcher] Positions loaded from CSV {path} — {npos} points, object {self._object_size}, backend={backend}, reset_period={self._reset_period}, fly_ny={self._fly_ny}, fly_nx={self._fly_nx}")
            if npos > 0:
                py0 = self._pos_y_pix[0] + origin[0]
                px0 = self._pos_x_pix[0] + origin[1]
                if py0 < -50 or py0 > obj_h + 50 or px0 < -50 or px0 > obj_w + 50:
                    print(
                        "[VitStitcher] WARNING: Positions are outside the image (first pos_after_origin=({:.1f},{:.1f}), image {}x{}).".format(
                            py0, px0, obj_h, obj_w
                        )
                    )
        except Exception as e:
            print(f"[VitStitcher] Failed to load positions from CSV {path}: {e}")

    def _load_positions(self, path: str) -> None:
        try:
            with h5py.File(path, "r") as h5:
                pos_pix = np.column_stack(
                    [h5["probe_position_y_m"][...], h5["probe_position_x_m"][...]]
                )
                self._pos_x_pix = pos_pix[:, 1]
                self._pos_y_pix = pos_pix[:, 0]
                self._object_size = tuple(h5["object"][0].shape)
            if _USE_TORCH:
                self._pos_origin_coords = (torch.tensor(self._object_size, dtype=torch.float32) / 2.0).round() + 0.5
                self._pred_ph = torch.zeros(self._object_size, dtype=torch.float32)
                self._buffer = torch.zeros(self._object_size, dtype=torch.float32)
            else:
                self._pos_origin_coords = (np.array(self._object_size, dtype=np.float64) / 2.0).round() + 0.5
                self._pred_ph = np.zeros(self._object_size, dtype=np.float32)
                self._buffer = np.zeros(self._object_size, dtype=np.float32)
            self._ready = True
            npos = len(self._pos_x_pix)
            if npos > 0:
                self._init_transmission_beam(npos)
        except Exception as e:
            print(f"[VitStitcher] Failed to load positions from {path}: {e}")

    def reset_accumulator(self) -> None:
        """Zero accumulation buffers (e.g. after detecting missed frames). Safe to call anytime."""
        if not self._ready or self._pred_ph is None or self._buffer is None:
            return
        if _USE_TORCH:
            self._pred_ph.zero_()
            self._buffer.zero_()
        else:
            self._pred_ph.fill(0)
            self._buffer.fill(0)
        if self._accu_int is not None:
            self._accu_int.data[...] = 0
            self._accu_int.mask[...] = True

    def _beam_panel_for_uid(self, uid: int) -> np.ndarray:
        """Build (fly_ny, fly_nx) beam position panel for given uid (LiveStitch7 style roll and crop)."""
        if self._curr_traj is None or self._fly_ny is None or self._fly_nx is None:
            return np.zeros((2, 2), dtype=np.float32)  # placeholder
        fly_ny, fly_nx = self._fly_ny, self._fly_nx
        sq = self._sq_size
        half = self._half_sq_size
        offset = int(uid // fly_nx) * (fly_nx + sq) + half + int(uid % fly_nx)
        rolled = np.roll(self._curr_traj, offset)
        traj_2d = rolled.reshape(fly_ny + sq, fly_nx + sq)
        return traj_2d[half : half + fly_ny, half : half + fly_nx].astype(np.float32)

    def process_frames_batch(self, frames: list) -> tuple:
        """
        Batch processing: runs reference-accurate accumulation (clamp buffer to 1 before every add)
        so newest frame has ~50% weight. Processes batch sequentially; each step uses vectorized
        place_patches for speed. Returns (composite, [transmission, diffraction, beam_position, nn_prediction, nn_stitched]).
        """
        if not frames:
            return None, None

        # 1. Unpack the batch — frames is list of (stream, unique_id)
        last_stream, last_unique_id = frames[-1]

        # 2. If not ready (no positions), fallback to single frame logic on the last frame
        if not self._ready:
            return self.process_frame(last_stream, last_unique_id)

        # 3. Pre-allocate batch arrays
        ec = self._patch_edge_crop
        npos = len(self._pos_x_pix)

        streams = [x[0] for x in frames]
        batch_stream = np.stack(streams, axis=0) if len(streams) > 1 else np.asarray(streams[0])[None, ...]
        if batch_stream.ndim == 2:
            batch_stream = batch_stream[None, ...]

        # Patches: data[ec:-ec, ec:-ec] from bottom half
        data_patches = batch_stream[:, 256 + ec : -ec, ec:-ec]

        unique_ids = np.array([x[1] for x in frames], dtype=np.int64)
        if self._reset_period > 0 and (unique_ids[0] - self._id_offset) % self._reset_period == 0:
            self.reset_accumulator()
        uids = (unique_ids - self._id_offset) % npos if npos else np.zeros_like(unique_ids)

        batch_ys = self._pos_y_pix[uids]
        batch_xs = self._pos_x_pix[uids]

        # 4. Run stitching and transmission updates in the same loop
        if _USE_TORCH:
            patches_t = torch.from_numpy(data_patches.astype(np.float32))
            pos_y_t = torch.from_numpy(batch_ys.astype(np.float32))
            pos_x_t = torch.from_numpy(batch_xs.astype(np.float32))
            pos_tensor = torch.stack([pos_y_t, pos_x_t], dim=1) + self._pos_origin_coords.to(pos_y_t.device)
            ones_t = torch.ones_like(patches_t[0:1])

            for i in range(len(patches_t)):
                curr_pos = pos_tensor[i : i + 1]
                curr_patch = patches_t[i : i + 1]
                self._pred_ph = _place_patches_fourier_shift(
                    self._pred_ph, curr_pos, curr_patch, op="add", adjoint_mode=False, pad=PAD
                )
                self._buffer = torch.clamp(self._buffer, max=1.0)
                self._buffer = _place_patches_fourier_shift(
                    self._buffer, curr_pos, ones_t, op="add", adjoint_mode=False, pad=PAD
                )
                self._pred_ph = self._pred_ph / torch.clamp(self._buffer, min=1.0)
                if self._accu_int is not None:
                    uid_i = int(uids[i])
                    diff_i = np.maximum(batch_stream[i, :256, :], 0.0)
                    row, col = uid_i // self._fly_nx, uid_i % self._fly_nx
                    if 0 <= row < self._fly_ny and 0 <= col < self._fly_nx:
                        self._accu_int.data[row, col] = float(np.sum(diff_i))
                        self._accu_int.mask[row, col] = False

            acc_np = self._pred_ph.cpu().numpy()
        else:
            origin = np.asarray(self._pos_origin_coords, dtype=np.float64)
            positions = (np.column_stack([batch_ys, batch_xs]) + origin).astype(np.float64)
            ones_batch = np.ones_like(data_patches[0:1])

            for i in range(len(data_patches)):
                curr_pos = positions[i : i + 1]
                curr_patch = data_patches[i : i + 1]
                self._pred_ph = _place_patches_fourier_shift_np(
                    self._pred_ph, curr_pos, curr_patch, op="add", adjoint_mode=False, pad=PAD
                )
                self._buffer = np.clip(self._buffer, None, 1.0)
                self._buffer = _place_patches_fourier_shift_np(
                    self._buffer, curr_pos, ones_batch, op="add", adjoint_mode=False, pad=PAD
                )
                self._pred_ph = self._pred_ph / np.clip(self._buffer, 1.0, None)
                if self._accu_int is not None:
                    uid_i = int(uids[i])
                    diff_i = np.maximum(batch_stream[i, :256, :], 0.0)
                    row, col = uid_i // self._fly_nx, uid_i % self._fly_nx
                    if 0 <= row < self._fly_ny and 0 <= col < self._fly_nx:
                        self._accu_int.data[row, col] = float(np.sum(diff_i))
                        self._accu_int.mask[row, col] = False

            acc_np = self._pred_ph.copy()

        # Prediction panel: raw inference patch
        single_np = data_patches[-1].copy()

        # 5. Display crops and build five panels
        diff_panel = np.maximum(last_stream[:256, :], 0.0).astype(np.float32)
        cc = self._center_crop_display
        acc_panel = acc_np
        if cc > 0:
            h, w = acc_np.shape
            if h > 2 * cc and w > 2 * cc:
                acc_panel = acc_np[cc:-cc, cc:-cc].copy()
        single_panel = single_np

        transmission_panel = self._accu_int.data.astype(np.float32).copy() if self._accu_int is not None else np.zeros((self._fly_ny or 2, self._fly_nx or 2), dtype=np.float32)
        last_uid = int(uids[-1]) if len(uids) else 0
        beam_panel = self._beam_panel_for_uid(last_uid)

        panels = [transmission_panel, diff_panel, beam_panel, single_panel, acc_panel]
        composite = transmission_panel  # PVAReader uses vit_panels; composite kept for shape/backward compat
        return composite, panels

    def process_frame(self, stream_512_256: np.ndarray, unique_id: int) -> np.ndarray:
        """
        stream_512_256: (512, 256) float32 from PVA value.
        unique_id: PVA uniqueId, used to index position.
        Returns (composite, panels): panels [transmission, diffraction, beam_position, nn_prediction, nn_stitched].
        If positions not loaded, returns 5 panels with zeros where needed.
        """
        stream = np.asarray(stream_512_256, dtype=np.float32)
        if stream.shape != STREAM_SHAPE:
            stream = np.broadcast_to(
                stream.ravel()[: np.prod(STREAM_SHAPE)].reshape(STREAM_SHAPE),
                STREAM_SHAPE,
            ).copy()
        diff = np.maximum(stream[:DIFF_ROWS, :], 0.0).astype(np.float32)
        data = stream[DIFF_ROWS:, :]

        data_max = float(np.max(data))
        if data_max == 0 and not getattr(self, "_zero_data_warned", False):
            self._zero_data_warned = True
            print("[VitStitcher] WARNING: Input patch data is all zero. Stitching panels will be black (check shutter / vit:1:CSSI).")

        if not self._ready:
            transmission = np.zeros((2, 2), dtype=np.float32)
            beam = np.zeros((2, 2), dtype=np.float32)
            single = np.zeros_like(diff)
            acc = np.zeros_like(diff)
            panels = [transmission, diff.copy(), beam, single, acc]
            return transmission, panels

        npos = len(self._pos_x_pix)
        uid = int(unique_id - self._id_offset) % npos if npos else 0
        if self._reset_period > 0 and (unique_id - self._id_offset) % self._reset_period == 0:
            self.reset_accumulator()

        if self._accu_int is not None:
            row, col = uid // self._fly_nx, uid % self._fly_nx
            if 0 <= row < self._fly_ny and 0 <= col < self._fly_nx:
                self._accu_int.data[row, col] = float(np.sum(diff))
                self._accu_int.mask[row, col] = False

        pos_y_pix = float(self._pos_y_pix[uid])
        pos_x_pix = float(self._pos_x_pix[uid])
        ec = self._patch_edge_crop
        patch = data[ec:-ec, ec:-ec].astype(np.float32)

        if _USE_TORCH:
            pos_tensor = torch.tensor([[pos_y_pix, pos_x_pix]], dtype=torch.float32) + self._pos_origin_coords
            patch_t = torch.from_numpy(patch[np.newaxis, ...].astype(np.float32))
            ones_t = torch.ones_like(patch_t)
            self._pred_ph = _place_patches_fourier_shift(self._pred_ph, pos_tensor, patch_t, op="add", adjoint_mode=False, pad=PAD)
            self._buffer = torch.clamp(self._buffer, max=1.0)
            self._buffer = _place_patches_fourier_shift(self._buffer, pos_tensor, ones_t, op="add", adjoint_mode=False, pad=PAD)
            self._pred_ph = self._pred_ph / torch.clamp(self._buffer, min=1.0)
            acc_np = self._pred_ph.numpy()
        else:
            origin = np.asarray(self._pos_origin_coords, dtype=np.float64)
            positions = (np.array([[pos_y_pix, pos_x_pix]], dtype=np.float64) + origin).astype(np.float64)
            patch_batch = patch[np.newaxis, :, :]
            ones_batch = np.ones_like(patch_batch)
            self._pred_ph = _place_patches_fourier_shift_np(self._pred_ph, positions, patch_batch, op="add", adjoint_mode=False, pad=PAD)
            self._buffer = np.clip(self._buffer, None, 1.0)
            self._buffer = _place_patches_fourier_shift_np(self._buffer, positions, ones_batch, op="add", adjoint_mode=False, pad=PAD)
            self._pred_ph = self._pred_ph / np.clip(self._buffer, 1.0, None)
            acc_np = self._pred_ph.copy()

        single_np = patch.copy()
        cc = self._center_crop_display
        acc_panel = acc_np
        if cc > 0:
            h, w = acc_np.shape
            if h > 2 * cc and w > 2 * cc:
                acc_panel = acc_np[cc:-cc, cc:-cc].copy()
        diff_panel = diff.copy()
        single_panel = single_np

        transmission_panel = self._accu_int.data.astype(np.float32).copy() if self._accu_int is not None else np.zeros((self._fly_ny or 2, self._fly_nx or 2), dtype=np.float32)
        beam_panel = self._beam_panel_for_uid(uid)

        panels = [transmission_panel, diff_panel, beam_panel, single_panel, acc_panel]
        return transmission_panel, panels


# Module-level singleton for PVAReader (one channel = one stitcher)
_stitcher = None


def get_stitcher(
    positions_hdf5_path: str = None,
    positions_npz_path: str = None,
    positions_csv_path: str = None,
    patch_edge_crop: int = None,
    center_crop_display: int = None,
) -> VitStitcher:
    global _stitcher
    if _stitcher is None:
        _stitcher = VitStitcher(
            positions_hdf5_path=positions_hdf5_path,
            positions_npz_path=positions_npz_path,
            positions_csv_path=positions_csv_path,
            patch_edge_crop=patch_edge_crop,
            center_crop_display=center_crop_display,
        )
    return _stitcher
