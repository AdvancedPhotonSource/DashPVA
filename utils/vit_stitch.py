"""
LiveStitch-style patching/stitching for vit:1:input_phase.
Produces three panels: raw diff, single patch placed, accumulated stitched image.
Used by PVAReader when subscribing to vit:1:input_phase.
"""
from math import pi
import os
import numpy as np
import torch
from torch import Tensor

try:
    import h5py
except ImportError:
    h5py = None

# Stitch pad and stream layout (match LiveStitch6)
STREAM_SHAPE = (512, 256)
DIFF_ROWS, PATCH_ROWS = 256, 256
PAD = 32  # crop data to data[32:-32, 32:-32] before placing


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


class VitStitcher:
    """
    Stateful stitcher for vit:1:input_phase stream.
    Expects stream shape (512, 256): diff = stream[:256,:], data = stream[256:,:].
    Positions loaded from HDF5 (path from env VIT_STITCH_POSITIONS_HDF5 or fallback path).
    """

    def __init__(self, positions_hdf5_path: str = None):
        self._ready = False
        self._pos_x_pix = None
        self._pos_y_pix = None
        self._object_size = None
        self._pos_origin_coords = None
        self._pred_ph = None
        self._buffer = None
        path = positions_hdf5_path or os.environ.get("VIT_STITCH_POSITIONS_HDF5", "")
        if path and h5py is not None and os.path.isfile(path):
            self._load_positions(path)
        else:
            if not path:
                path = "(none set)"
            print(f"[VitStitcher] Positions HDF5 not loaded: {path} — will output raw diff only.")

    def _load_positions(self, path: str) -> None:
        try:
            with h5py.File(path, "r") as h5:
                pos_pix = np.column_stack(
                    [h5["probe_position_y_m"][...], h5["probe_position_x_m"][...]]
                )
                self._pos_x_pix = pos_pix[:, 1]
                self._pos_y_pix = pos_pix[:, 0]
                self._object_size = tuple(h5["object"][0].shape)
            self._pos_origin_coords = torch.tensor(self._object_size, dtype=torch.float32) / 2.0
            self._pos_origin_coords = self._pos_origin_coords.round() + 0.5
            self._pred_ph = torch.zeros(self._object_size, dtype=torch.float32)
            self._buffer = torch.zeros(self._object_size, dtype=torch.float32)
            self._ready = True
        except Exception as e:
            print(f"[VitStitcher] Failed to load positions from {path}: {e}")

    def process_frame(self, stream_512_256: np.ndarray, unique_id: int) -> np.ndarray:
        """
        stream_512_256: (512, 256) float32 from PVA value.
        unique_id: PVA uniqueId, used to index position.
        Returns (H, 3*W) composite: [ raw diff | single patch placed | accumulated ].
        If positions not loaded, returns (512, 256*3) with diff repeated/zeros in other panels.
        """
        stream = np.asarray(stream_512_256, dtype=np.float32)
        if stream.shape != STREAM_SHAPE:
            stream = np.broadcast_to(
                stream.ravel()[: np.prod(STREAM_SHAPE)].reshape(STREAM_SHAPE),
                STREAM_SHAPE,
            ).copy()
        diff = stream[:DIFF_ROWS, :]   # (256, 256)
        data = stream[DIFF_ROWS:, :]   # (256, 256)

        if not self._ready:
            # No positions: show diff in all three panels (or diff | zeros | zeros)
            single = np.zeros_like(diff)
            acc = np.zeros_like(diff)
            composite = np.hstack([diff, single, acc])
            return composite.astype(np.float32)

        npos = len(self._pos_x_pix)
        uid = int(unique_id) % npos if npos else 0
        if uid == 0:
            self._pred_ph.zero_()
            self._buffer.zero_()

        pos_y_pix = float(self._pos_y_pix[uid])
        pos_x_pix = float(self._pos_x_pix[uid])
        pos_tensor = torch.tensor([[pos_y_pix, pos_x_pix]], dtype=torch.float32)
        pos_tensor = pos_tensor + self._pos_origin_coords

        patch = data[PAD:-PAD, PAD:-PAD]  # (192, 192)
        patch_t = torch.from_numpy(patch[np.newaxis, ...].astype(np.float32))
        ones_t = torch.ones_like(patch_t)

        tmp = torch.zeros(self._object_size, dtype=torch.float32)
        tmp = _place_patches_fourier_shift(
            tmp, pos_tensor, patch_t, op="add", adjoint_mode=False, pad=PAD
        )
        self._pred_ph = _place_patches_fourier_shift(
            self._pred_ph, pos_tensor, patch_t, op="add", adjoint_mode=False, pad=PAD
        )
        self._buffer = torch.clamp(self._buffer, max=1.0)
        self._buffer = _place_patches_fourier_shift(
            self._buffer, pos_tensor, ones_t, op="add", adjoint_mode=False, pad=PAD
        )
        self._pred_ph = self._pred_ph / torch.clamp(self._buffer, min=1.0)

        single_np = tmp.numpy()
        acc_np = self._pred_ph.numpy()
        # Align heights for hstack: diff (256,256) vs stitched (Hy,Wy)
        h_diff, w_diff = diff.shape[0], diff.shape[1]
        h_obj, w_obj = single_np.shape[0], single_np.shape[1]
        h_common = max(h_diff, h_obj)
        if h_diff < h_common:
            diff_pad = np.zeros((h_common, w_diff), dtype=np.float32)
            diff_pad[:h_diff, :] = diff
            diff = diff_pad
        elif h_obj < h_common:
            single_pad = np.zeros((h_common, w_obj), dtype=np.float32)
            acc_pad = np.zeros((h_common, w_obj), dtype=np.float32)
            single_pad[:h_obj, :] = single_np
            acc_pad[:h_obj, :] = acc_np
            single_np, acc_np = single_pad, acc_pad
        composite = np.hstack([diff, single_np, acc_np])
        return composite.astype(np.float32)


# Module-level singleton for PVAReader (one channel = one stitcher)
_stitcher = None


def get_stitcher(positions_hdf5_path: str = None) -> VitStitcher:
    global _stitcher
    if _stitcher is None:
        _stitcher = VitStitcher(positions_hdf5_path)
    return _stitcher
