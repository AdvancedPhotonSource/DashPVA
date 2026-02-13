"""
ROI operations utilities

Provides helpers to extract ROI stacks from the current viewer state, align/crop
two stacks for comparison, and compute per-frame means for series plots.
"""

from typing import Tuple, Dict, Any
import numpy as np


def _extract_roi_subarray(frame: np.ndarray, roi, image_item) -> np.ndarray:
    """
    Transform-aware ROI extraction for a single 2D frame using pyqtgraph ROI helpers.
    Falls back to pixel slicing via roi.pos()/roi.size() when needed.
    Returns a 2D numpy array (may be empty if out-of-bounds).
    """
    sub = None
    # Try transform-aware extraction first
    try:
        if image_item is not None and hasattr(roi, "getArrayRegion"):
            sub = roi.getArrayRegion(frame, image_item)
            if sub is not None and hasattr(sub, "ndim") and sub.ndim > 2:
                sub = np.squeeze(sub)
    except Exception:
        sub = None

    # Fallback to pixel slicing via ROI position/size
    if sub is None or int(getattr(sub, "size", 0)) == 0:
        try:
            pos = roi.pos(); size = roi.size()
            x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
            w = max(1, int(size.x())); h = max(1, int(size.y()))
            height, width = frame.shape
            x1 = min(width, x0 + w); y1 = min(height, y0 + h)
            if x0 < x1 and y0 < y1:
                sub = frame[y0:y1, x0:x1]
        except Exception:
            sub = None

    # Ensure we return a 2D float32 array when possible
    if sub is not None and int(getattr(sub, "size", 0)) > 0:
        return np.asarray(sub, dtype=np.float32)
    return None


def extract_roi_stack(main_window, roi) -> np.ndarray:
    """
    Extract an ROI stack from the viewer's current 2D data.

    - If current_2d_data is 3D: returns shape (T, H, W)
    - If single 2D frame: returns shape (H, W)

    Uses transform-aware extraction (getArrayRegion with imageItem); falls back to
    pixel slicing via roi.pos()/roi.size(). Pads nothing; if any frame yields empty
    area, falls back to zeros of the current ROI box size. Ensures consistent H/W
    across frames by trimming to the smallest extracted size.
    """
    data = getattr(main_window, "current_2d_data", None)
    image_item = getattr(main_window.image_view, "imageItem", None) if hasattr(main_window, "image_view") else None

    if data is None:
        # Attempt to grab current frame from viewer as a fallback
        try:
            if hasattr(main_window, "roi_manager"):
                frame = main_window.roi_manager.get_current_frame_data()
                if frame is not None:
                    data = frame
        except Exception:
            pass

    if data is None:
        return None

    if isinstance(data, np.ndarray) and data.ndim == 3:
        T = int(data.shape[0])
        samples = []
        # Determine a default ROI box size for zero-fallbacks
        try:
            size = roi.size(); w_box = max(1, int(size.x())); h_box = max(1, int(size.y()))
        except Exception:
            w_box = max(1, int(data.shape[2] if data.shape[2] > 0 else 1))
            h_box = max(1, int(data.shape[1] if data.shape[1] > 0 else 1))
        for i in range(T):
            frame = np.asarray(data[i], dtype=np.float32)
            sub = _extract_roi_subarray(frame, roi, image_item)
            if sub is None or int(getattr(sub, "size", 0)) == 0:
                samples.append(np.zeros((h_box, w_box), dtype=np.float32))
            else:
                samples.append(sub)
        # Trim to smallest H/W across frames to ensure stackability
        min_h = min(int(s.shape[0]) for s in samples)
        min_w = min(int(s.shape[1]) for s in samples)
        stack = np.stack([s[:min_h, :min_w] for s in samples], axis=0)
        return stack
    else:
        frame = np.asarray(data, dtype=np.float32)
        sub = _extract_roi_subarray(frame, roi, image_item)
        if sub is None or int(getattr(sub, "size", 0)) == 0:
            # Return zeros of the current ROI box size when empty
            try:
                size = roi.size(); w_box = max(1, int(size.x())); h_box = max(1, int(size.y()))
                return np.zeros((h_box, w_box), dtype=np.float32)
            except Exception:
                return np.zeros_like(frame, dtype=np.float32)
        return sub


def align_stacks(A: np.ndarray, B: np.ndarray, *, strict: bool = True, auto_intersect: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Align two ROI stacks A and B for comparison.

    - strict=True: require identical shapes (including frame counts); if not, return error info.
    - auto_intersect=True: crop both to min(H, W) and truncate to min(T) when not strict.
    - Returns (A_aligned, B_aligned, info) where info contains notes/status.
    """
    info: Dict[str, Any] = {"ok": True, "notes": []}

    if A is None or B is None:
        return None, None, {"ok": False, "notes": ["Missing ROI stack(s)"]}

    # Normalize dimensionality: treat 2D as (1, H, W) for alignment logic
    def _normalize(X):
        if X.ndim == 2:
            return X[np.newaxis, ...]
        return X

    A_n = _normalize(np.asarray(A))
    B_n = _normalize(np.asarray(B))

    # Shapes
    T_A, H_A, W_A = A_n.shape
    T_B, H_B, W_B = B_n.shape

    if strict:
        if (T_A, H_A, W_A) != (T_B, H_B, W_B):
            info = {"ok": False, "notes": [f"Strict match failed: A shape {A_n.shape} vs B shape {B_n.shape}"]}
            return None, None, info
        return A_n, B_n, {"ok": True, "notes": ["Strict match: shapes identical"]}

    # Non-strict: optionally auto-intersect
    T = min(T_A, T_B)
    H = min(H_A, H_B)
    W = min(W_A, W_B)
    if auto_intersect:
        A_al = A_n[:T, :H, :W]
        B_al = B_n[:T, :H, :W]
        info["notes"].append(f"Auto-intersect applied: T={T}, H={H}, W={W}")
        if T_A != T_B:
            info["notes"].append(f"Frames truncated to min(T): {T}")
        if (H_A, W_A) != (H, W):
            info["notes"].append("Cropped to overlapping region")
        return A_al, B_al, info

    # Non-strict and no auto-intersect: require equality of shapes, else error
    if (T_A, H_A, W_A) != (T_B, H_B, W_B):
        return None, None, {"ok": False, "notes": ["Non-strict without auto-intersect still requires equal shapes"]}
    return A_n, B_n, {"ok": True, "notes": ["Non-strict: shapes equal"]}


def per_frame_mean(stack: np.ndarray) -> np.ndarray:
    """
    Compute mean per frame for a stack. Returns a 1D array of length T.
    For a single 2D image, returns an array of length 1.
    """
    if stack is None:
        return None
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return np.array([float(np.mean(arr))], dtype=np.float32)
    if arr.ndim == 3:
        # mean over H and W for each frame
        return np.mean(arr, axis=(1, 2)).astype(np.float32)
    # Unexpected dims: flatten
    return np.array([float(np.mean(arr))], dtype=np.float32)
