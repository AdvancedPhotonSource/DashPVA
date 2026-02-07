"""
LiveStitch-style patching/stitching for vit:1:input_phase.
Produces three panels: raw diff, single patch placed, accumulated stitched image.
Used by PVAReader when subscribing to vit:1:input_phase.
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


# Default CSV positions path (format: no header, col0=y_m, col1=x_m in meters)
DEFAULT_POSITIONS_CSV = "/home/beams/AILEENLUO/ptycho-vit/workspace/positions78.csv"


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
    Stateful stitcher for vit:1:input_phase stream.
    Expects stream shape (512, 256): diff = stream[:256,:], data = stream[256:,:].
    Positions: from CSV (default) if VIT_STITCH_POSITIONS_CSV or default path exists, else npz, else HDF5.
    - LiveStitch4: patch_edge_crop=66 (data[66:-66]), center_crop_display=150 for panels 2 & 3.
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
        self._patch_edge_crop = (
            patch_edge_crop
            if patch_edge_crop is not None
            else _parse_int_env("VIT_STITCH_PATCH_EDGE_CROP", 32)
        )
        self._center_crop_display = (
            center_crop_display
            if center_crop_display is not None
            else _parse_int_env("VIT_STITCH_CENTER_CROP", 0)
        )
        # Load the offset from environment (start-from-correct-place for spiral)
        self._id_offset = _parse_int_env("VIT_STITCH_ID_OFFSET", 507270)
        # Clear stitched canvas when (unique_id - offset) % reset_period == 0 (start of cycle). 0 = disabled.
        self._reset_period = _parse_int_env("VIT_STITCH_RESET_PERIOD", 10100)

        csv_path = _find_csv_path(
            os.environ.get("VIT_STITCH_POSITIONS_CSV", ""),
            positions_csv_path,
        )
        if csv_path and os.path.isfile(csv_path):
            if patch_edge_crop is None and "VIT_STITCH_PATCH_EDGE_CROP" not in os.environ:
                self._patch_edge_crop = 66
            if center_crop_display is None and "VIT_STITCH_CENTER_CROP" not in os.environ:
                self._center_crop_display = 150
            self._load_positions_csv(csv_path)
        else:
            npz_path = _find_npz_path(
                os.environ.get("VIT_STITCH_NPZ_PATH", ""),
                positions_npz_path,
            )
            if npz_path and os.path.isfile(npz_path):
                if patch_edge_crop is None and "VIT_STITCH_PATCH_EDGE_CROP" not in os.environ:
                    self._patch_edge_crop = 66
                if center_crop_display is None and "VIT_STITCH_CENTER_CROP" not in os.environ:
                    self._center_crop_display = 150
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

    def _load_positions_npz(self, path: str) -> None:
        """Load positions from npz. Supports:
        - 'x_pix', 'y_pix': pixel offsets from center (used as-is).
        - VIT_STITCH_NPZ_PIXEL_OFFSETS=1: treat 'x', 'y' as pixel offsets from center.
        - Else: 'x', 'y' in route units with step (step_m = spiral_step * -1e-6) and pixel_size → pixels (LiveStitch4.py convention; e.g. LiveStitch4.py 0.05).
        """
        try:
            data = np.load(path)
            obj_h = int(os.environ.get("VIT_STITCH_OBJECT_H", "1161"))
            obj_w = int(os.environ.get("VIT_STITCH_OBJECT_W", "1161"))
            self._object_size = (obj_h, obj_w)
            origin = np.round(np.array(self._object_size, dtype=np.float64) / 2.0) + 0.5

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
            backend = "torch" if _USE_TORCH else "numpy"
            print(f"[VitStitcher] LiveStitch4 positions loaded from {path} — {npos} points, object {self._object_size}, backend={backend}, mode={mode}")

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

    def _load_positions_csv(self, path: str) -> None:
        """Load positions from CSV: no header, col0=x_m, col1=y_m (meters). Converts to pixels via VIT_STITCH_PIXEL_SIZE.
        Object size (prediction/stitch canvas) is computed from position range so all positions fit with patch margin."""
        try:
            data = np.loadtxt(path, delimiter=",", dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            x_m = np.asarray(data[:, 0], dtype=np.float64)
            y_m = np.asarray(data[:, 1], dtype=np.float64)
            pixel_size = float(os.environ.get("VIT_STITCH_PIXEL_SIZE", "6.89e-9"))
            pos_x_pix = -x_m / pixel_size
            pos_y_pix = y_m / pixel_size
            self._pos_x_pix = pos_x_pix
            self._pos_y_pix = pos_y_pix
            # Object size: from env if set, else from position range (origin at center; add margin for patch)
            obj_h_env = os.environ.get("VIT_STITCH_OBJECT_H", "")
            obj_w_env = os.environ.get("VIT_STITCH_OBJECT_W", "")
            if obj_h_env.isdigit() and obj_w_env.isdigit():
                obj_h = int(obj_h_env)
                obj_w = int(obj_w_env)
            elif len(pos_x_pix) > 0 and len(pos_y_pix) > 0:
                patch_margin = int(os.environ.get("VIT_STITCH_OBJECT_MARGIN", "256"))
                extent_x = 2 * float(np.ceil(np.max(np.abs(pos_x_pix))))
                extent_y = 2 * float(np.ceil(np.max(np.abs(pos_y_pix))))
                obj_w = max(int(extent_x) + patch_margin, patch_margin)
                obj_h = max(int(extent_y) + patch_margin, patch_margin)
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
            backend = "torch" if _USE_TORCH else "numpy"
            print(f"[VitStitcher] Positions loaded from CSV {path} — {npos} points, object {self._object_size}, backend={backend}")
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

    def process_frames_batch(self, frames: list) -> tuple:
        """
        Vectorized batch processing. Stacks 'data' from all frames to update accumulation
        in one pass, but only processes 'diff'/'single' for the final frame in the batch.
        """
        if not frames:
            return None, None

        # 1. Unpack the batch — frames is list of (stream, unique_id)
        # We need the LAST stream for the 'diff' panel display
        last_stream, last_unique_id = frames[-1]

        # 2. If not ready (no positions), fallback to single frame logic on the last frame
        if not self._ready:
            return self.process_frame(last_stream, last_unique_id)

        # 3. Pre-allocate batch arrays
        ec = self._patch_edge_crop
        npos = len(self._pos_x_pix)

        # Extract streams and stack: (N, 512, 256)
        streams = [x[0] for x in frames]
        batch_stream = np.stack(streams, axis=0) if len(streams) > 1 else np.asarray(streams[0])[None, ...]
        if batch_stream.ndim == 2:
            batch_stream = batch_stream[None, ...]

        # Data is the bottom half [:, 256:, :]; patches: crop [:, 256+ec:-ec, ec:-ec]
        data_patches = batch_stream[:, 256 + ec : -ec, ec:-ec]

        unique_ids = np.array([x[1] for x in frames], dtype=np.int64)
        # At scan position 0 (start of cycle), clear stitched canvas before adding this batch
        if self._reset_period > 0 and (unique_ids[0] - self._id_offset) % self._reset_period == 0:
            self.reset_accumulator()
        uids = (unique_ids - self._id_offset) % npos if npos else np.zeros_like(unique_ids)

        batch_ys = self._pos_y_pix[uids]
        batch_xs = self._pos_x_pix[uids]

        # 4. Run vectorized stitching (same logic as original: add patches to pred_ph, add ones to buffer, then divide)
        if _USE_TORCH:
            patches_t = torch.from_numpy(data_patches.astype(np.float32))
            pos_y_t = torch.from_numpy(batch_ys.astype(np.float32))
            pos_x_t = torch.from_numpy(batch_xs.astype(np.float32))
            pos_tensor = torch.stack([pos_y_t, pos_x_t], dim=1) + self._pos_origin_coords.to(pos_y_t.device)
            ones_t = torch.ones_like(patches_t)

            # Original order: add patches to pred_ph, clamp buffer then add ones, then divide
            self._pred_ph = _place_patches_fourier_shift(
                self._pred_ph, pos_tensor, patches_t, op="add", adjoint_mode=False, pad=PAD
            )
            self._buffer = torch.clamp(self._buffer, max=1.0)
            self._buffer = _place_patches_fourier_shift(
                self._buffer, pos_tensor, ones_t, op="add", adjoint_mode=False, pad=PAD
            )
            self._pred_ph = self._pred_ph / torch.clamp(self._buffer, min=1.0)

            # Single panel: last frame only
            last_pos_t = pos_tensor[-1:, :]
            last_patch_t = patches_t[-1:, ...]
            tmp = torch.zeros(self._object_size, dtype=torch.float32, device=patches_t.device)
            single_t = _place_patches_fourier_shift(
                tmp, last_pos_t, last_patch_t, op="add", adjoint_mode=False, pad=PAD
            )
            single_np = single_t.cpu().numpy()
            acc_np = self._pred_ph.cpu().numpy()
        else:
            origin = np.asarray(self._pos_origin_coords, dtype=np.float64)
            positions = (np.column_stack([batch_ys, batch_xs]) + origin).astype(np.float64)
            ones_batch = np.ones_like(data_patches)

            # Original order: add patches to pred_ph, clamp buffer then add ones, then divide
            self._pred_ph = _place_patches_fourier_shift_np(
                self._pred_ph, positions, data_patches, op="add", adjoint_mode=False, pad=PAD
            )
            self._buffer = np.clip(self._buffer, None, 1.0)
            self._buffer = _place_patches_fourier_shift_np(
                self._buffer, positions, ones_batch, op="add", adjoint_mode=False, pad=PAD
            )
            self._pred_ph = self._pred_ph / np.clip(self._buffer, 1.0, None)
            acc_np = self._pred_ph.copy()

            tmp = np.zeros(self._object_size, dtype=np.float32)
            single_np = _place_patches_fourier_shift_np(
                tmp, positions[-1:], data_patches[-1:], op="add", adjoint_mode=False, pad=PAD
            )

        # 5. Prepare final composite for display
        diff = np.maximum(last_stream[:256, :], 0.0).astype(np.float32)
        cc = self._center_crop_display
        if cc > 0 and single_np.shape[0] > 2 * cc and single_np.shape[1] > 2 * cc:
            single_np = single_np[cc:-cc, cc:-cc].copy()
            acc_np = acc_np[cc:-cc, cc:-cc].copy()

        diff_panel = diff
        single_panel = single_np
        acc_panel = acc_np

        h_diff, w_diff = diff.shape
        h_obj, w_obj = single_np.shape[0], single_np.shape[1]
        h_common = max(h_diff, h_obj)
        if h_diff < h_common:
            d_comp = np.zeros((h_common, w_diff), dtype=np.float32)
            d_comp[:h_diff, :] = diff
        else:
            d_comp = diff
        if h_obj < h_common:
            s_comp = np.zeros((h_common, w_obj), dtype=np.float32)
            a_comp = np.zeros((h_common, w_obj), dtype=np.float32)
            s_comp[:h_obj, :] = single_np
            a_comp[:h_obj, :] = acc_np
        else:
            s_comp, a_comp = single_np, acc_np
        composite = np.hstack([d_comp, s_comp, a_comp])
        # Raw patch for last frame (no ghostly canvas); order so View 2 = stitched, View 3 = raw patch
        raw_patch = data_patches[-1].copy()
        return composite.astype(np.float32), [diff_panel, raw_patch, acc_panel]

    def process_frame(self, stream_512_256: np.ndarray, unique_id: int) -> np.ndarray:
        """
        stream_512_256: (512, 256) float32 from PVA value.
        unique_id: PVA uniqueId, used to index position.
        Returns (composite, panels): composite (H, 3*W), panels [diff, single, accumulated]
        for separate image views with per-panel color bars.
        If positions not loaded, returns composite and [diff, zeros, zeros].
        """
        stream = np.asarray(stream_512_256, dtype=np.float32)
        if stream.shape != STREAM_SHAPE:
            stream = np.broadcast_to(
                stream.ravel()[: np.prod(STREAM_SHAPE)].reshape(STREAM_SHAPE),
                STREAM_SHAPE,
            ).copy()
        # Diffraction: clamp to >= 0 so log scale and percentiles are well-defined
        diff = np.maximum(stream[:DIFF_ROWS, :], 0.0).astype(np.float32)   # (256, 256)
        data = stream[DIFF_ROWS:, :]   # (256, 256)

        # Warn once if detector patch data is all zeros (shutter closed or wrong channel)
        data_max = float(np.max(data))
        if data_max == 0 and not getattr(self, "_zero_data_warned", False):
            self._zero_data_warned = True
            print("[VitStitcher] WARNING: Input patch data is all zero. Stitching panels will be black (check shutter / vit:1:input_phase).")

        if not self._ready:
            # No positions: show diff in first panel, zeros in others
            single = np.zeros_like(diff)
            acc = np.zeros_like(diff)
            composite = np.hstack([diff, single, acc])
            return composite.astype(np.float32), [diff, single, acc]

        npos = len(self._pos_x_pix)
        # Subtract offset, then modulo (treat offset as spiral index 0)
        uid = int(unique_id - self._id_offset) % npos if npos else 0
        # At scan position 0 (start of cycle), clear stitched canvas for fresh accumulation
        if self._reset_period > 0 and (unique_id - self._id_offset) % self._reset_period == 0:
            self.reset_accumulator()

        pos_y_pix = float(self._pos_y_pix[uid])
        pos_x_pix = float(self._pos_x_pix[uid])
        ec = self._patch_edge_crop  # 32 -> (192,192); 66 -> (124,124) LiveStitch4
        patch = data[ec:-ec, ec:-ec].astype(np.float32)

        if _USE_TORCH:
            pos_tensor = torch.tensor([[pos_y_pix, pos_x_pix]], dtype=torch.float32) + self._pos_origin_coords
            patch_t = torch.from_numpy(patch[np.newaxis, ...].astype(np.float32))
            ones_t = torch.ones_like(patch_t)
            tmp = torch.zeros(self._object_size, dtype=torch.float32)
            tmp = _place_patches_fourier_shift(tmp, pos_tensor, patch_t, op="add", adjoint_mode=False, pad=PAD)
            self._pred_ph = _place_patches_fourier_shift(self._pred_ph, pos_tensor, patch_t, op="add", adjoint_mode=False, pad=PAD)
            self._buffer = torch.clamp(self._buffer, max=1.0)
            self._buffer = _place_patches_fourier_shift(self._buffer, pos_tensor, ones_t, op="add", adjoint_mode=False, pad=PAD)
            self._pred_ph = self._pred_ph / torch.clamp(self._buffer, min=1.0)
            single_np = tmp.numpy()
            acc_np = self._pred_ph.numpy()
        else:
            origin = np.asarray(self._pos_origin_coords, dtype=np.float64)
            positions = (np.array([[pos_y_pix, pos_x_pix]], dtype=np.float64) + origin).astype(np.float64)
            patch_batch = patch[np.newaxis, :, :]
            ones_batch = np.ones_like(patch_batch)
            tmp = np.zeros(self._object_size, dtype=np.float32)
            tmp = _place_patches_fourier_shift_np(tmp, positions, patch_batch, op="add", adjoint_mode=False, pad=PAD)
            self._pred_ph = _place_patches_fourier_shift_np(self._pred_ph, positions, patch_batch, op="add", adjoint_mode=False, pad=PAD)
            self._buffer = np.clip(self._buffer, None, 1.0)
            self._buffer = _place_patches_fourier_shift_np(self._buffer, positions, ones_batch, op="add", adjoint_mode=False, pad=PAD)
            self._pred_ph = self._pred_ph / np.clip(self._buffer, 1.0, None)
            single_np = tmp
            acc_np = self._pred_ph.copy()
        cc = self._center_crop_display
        if cc > 0 and single_np.shape[0] > 2 * cc and single_np.shape[1] > 2 * cc:
            single_np = single_np[cc:-cc, cc:-cc].copy()
            acc_np = acc_np[cc:-cc, cc:-cc].copy()
        # Panels for viewer: original diff (256x256), single and acc at their natural size (e.g. 861x861)
        diff_panel = diff.copy()   # (256, 256) — do not pad for panel 1
        single_panel = single_np.copy()
        acc_panel = acc_np.copy()
        # Build composite (padded) only for optional hstack display; viewer uses panels list
        h_diff, w_diff = diff.shape[0], diff.shape[1]
        h_obj, w_obj = single_np.shape[0], single_np.shape[1]
        h_common = max(h_diff, h_obj)
        if h_diff < h_common:
            diff_for_composite = np.zeros((h_common, w_diff), dtype=np.float32)
            diff_for_composite[:h_diff, :] = diff
        else:
            diff_for_composite = diff
        if h_obj < h_common:
            single_for_composite = np.zeros((h_common, w_obj), dtype=np.float32)
            acc_for_composite = np.zeros((h_common, w_obj), dtype=np.float32)
            single_for_composite[:h_obj, :] = single_np
            acc_for_composite[:h_obj, :] = acc_np
        else:
            single_for_composite = single_np
            acc_for_composite = acc_np
        composite = np.hstack([diff_for_composite, single_for_composite, acc_for_composite])
        # Return raw patch (no ghostly canvas); order so View 2 = stitched, View 3 = raw 64×64 patch
        return composite.astype(np.float32), [diff_panel, patch.copy(), acc_panel]


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
