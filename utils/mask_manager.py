import os
import json
import logging
import numpy as np
import pathlib

logger = logging.getLogger(__name__)


class MaskManager:
    """
    Manages detector pixel masks for DashPVA.

    Handles loading (.edf, .npy, .tif/.tiff, .json), combining (OR), saving, and applying masks.
    Also provides temporal-variance-based dead pixel detection.

    Convention: True = masked pixel, False = good pixel (matches pyFAI).
    """

    DEFAULT_MASK_FILENAME = 'active_mask.npy'

    def __init__(self, masks_dir=None):
        self.project_root = pathlib.Path(__file__).resolve().parents[1]
        self.masks_dir = masks_dir or str(self.project_root / 'masks')
        os.makedirs(self.masks_dir, exist_ok=True)

        self.mask = None
        self.mask_path = None
        self.mask_sources = []
        self._shape_mismatch_warned = False
        self.shape_mismatch_info = None  # Set to (mask_shape, image_shape) on first mismatch

        # Auto-load existing active mask
        default_path = os.path.join(self.masks_dir, self.DEFAULT_MASK_FILENAME)
        if os.path.exists(default_path):
            try:
                self.mask = np.load(default_path).astype(bool)
                self.mask_path = default_path
                self.mask_sources.append(default_path)
                logger.info(f"Loaded existing active mask: {default_path} "
                            f"({self.num_masked_pixels} masked pixels)")
            except Exception as e:
                logger.error(f"Failed to load active mask: {e}")

    def load_mask(self, filepath, detector_shape=None):
        """
        Load a mask from file. Supports .edf, .npy, .tif/.tiff, .json.

        For .json: expects EPICS NDPluginBadPixel format with [X,Y] = [col,row]
        in raw detector coordinates. Requires detector_shape=(rows, cols).

        Convention: nonzero = masked (True), zero = good (False).
        This matches pyFAI where 1=masked, 0=good.

        Returns boolean array.
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.json':
            return self._load_json_mask(filepath, detector_shape)
        elif ext == '.npy':
            data = np.load(filepath)
        elif ext in ('.tif', '.tiff'):
            from PIL import Image
            data = np.array(Image.open(filepath))
        elif ext == '.edf':
            import fabio
            data = fabio.open(filepath).data
        else:
            import fabio
            data = fabio.open(filepath).data

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        mask = data.astype(bool)
        logger.info(f"Loaded mask: {filepath}, shape={mask.shape}, "
                    f"masked={np.sum(mask)}/{mask.size}")
        return mask

    def _load_json_mask(self, filepath, detector_shape=None):
        """
        Load mask from EPICS NDPluginBadPixel JSON format.

        JSON [X,Y] = [col, row] in raw detector coordinates.
        Our mask is mask[row, col], so Pixel [X,Y] → mask[Y, X] = True.
        """
        if detector_shape is None:
            raise ValueError(
                "detector_shape=(rows, cols) required for JSON mask loading")

        with open(filepath, 'r') as f:
            data = json.load(f)

        bad_pixels = data.get('Bad pixels', [])
        mask = np.zeros(detector_shape, dtype=bool)
        skipped = 0

        for entry in bad_pixels:
            pixel = entry.get('Pixel')
            if pixel is None or len(pixel) < 2:
                skipped += 1
                continue
            x, y = int(pixel[0]), int(pixel[1])
            if 0 <= y < detector_shape[0] and 0 <= x < detector_shape[1]:
                mask[y, x] = True
            else:
                skipped += 1

        logger.info(f"Loaded JSON mask: {filepath}, shape={detector_shape}, "
                    f"masked={np.sum(mask)}, skipped={skipped}")
        return mask

    def export_json_mask(self, filepath, set_value=0):
        """
        Export current mask as EPICS NDPluginBadPixel JSON format.

        Mask is in detector-native orientation: mask[row, col].
        JSON uses [X, Y] = [col, row] in raw detector coordinates.
        All bad pixels use "Set" mode with the given replacement value.
        """
        if self.mask is None:
            logger.warning("No mask to export")
            return None

        bad_pixels = []
        rows, cols = np.where(self.mask)
        for row, col in zip(rows, cols):
            bad_pixels.append({"Pixel": [int(col), int(row)], "Set": set_value})

        data = {"Bad pixels": bad_pixels}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported JSON mask: {filepath}, "
                    f"{len(bad_pixels)} bad pixels")
        return filepath

    def _resize_mask(self, mask, target_shape):
        """Resize mask to target shape using nearest-neighbor interpolation."""
        from skimage.transform import resize
        logger.debug(f"Resizing mask {mask.shape} → {target_shape}")
        resized = resize(mask.astype(np.uint8), target_shape,
                         order=0, preserve_range=True, anti_aliasing=False)
        return resized.astype(bool)

    def combine_masks(self, new_mask, replace=False):
        """
        Combine a new mask with the existing mask via logical OR.
        If replace=True or no existing mask, sets mask directly.
        Handles shape mismatch with resize + warning.
        """
        self._shape_mismatch_warned = False
        if replace or self.mask is None:
            self.mask = new_mask.astype(bool)
        else:
            if new_mask.shape != self.mask.shape:
                new_mask = self._resize_mask(new_mask, self.mask.shape)
            self.mask = self.mask | new_mask.astype(bool)

    def save_active_mask(self, path=None):
        """Save the current mask as .npy. Returns the saved path."""
        if self.mask is None:
            logger.warning("No mask to save")
            return None
        save_path = path or os.path.join(self.masks_dir, self.DEFAULT_MASK_FILENAME)
        np.save(save_path, self.mask)
        self.mask_path = save_path
        logger.info(f"Saved active mask: {save_path} ({self.num_masked_pixels} masked pixels)")
        return save_path

    def detect_dead_pixels(self, frames, variance_threshold=1.0):
        """
        Detect stuck/dead pixels using temporal variance (illuminated mode).

        Pixels with variance below threshold across N frames are flagged.
        These are pixels that don't respond to changing illumination.

        Args:
            frames: list of 2D numpy arrays (N frames)
            variance_threshold: pixels with var < this are dead

        Returns:
            boolean mask (True = dead pixel)
        """
        if len(frames) < 3:
            logger.warning("Need at least 3 frames for dead pixel detection")
            return None

        stack = np.stack(frames, axis=0).astype(np.float64)
        pixel_variance = np.var(stack, axis=0)
        dead_mask = pixel_variance < variance_threshold
        num_dead = np.sum(dead_mask)
        logger.info(f"Dead pixel detection (illuminated): {num_dead} pixels with "
                    f"variance < {variance_threshold} (from {len(frames)} frames)")
        return dead_mask

    def detect_hot_pixels(self, frames, sigma=5.0):
        """
        Detect hot/stuck pixels from dark frames (no X-ray beam).

        Uses median stacking to reject cosmic rays — a cosmic ray only
        hits a pixel in 1-2 out of N frames, so the median ignores it.
        Then flags pixels where the median is significantly above zero
        using MAD (Median Absolute Deviation) robust statistics.

        A pixel is flagged as hot if:
            median_value > global_median + sigma * MAD

        This avoids flagging normal read-noise pixels while catching
        genuinely stuck/hot pixels.

        Args:
            frames: list of 2D numpy arrays (N dark frames)
            sigma: number of MAD above global median to flag (default: 5)

        Returns:
            boolean mask (True = hot pixel)
        """
        if len(frames) < 3:
            logger.warning("Need at least 3 frames for hot pixel detection")
            return None

        stack = np.stack(frames, axis=0).astype(np.float64)

        # Median across frames — rejects cosmic rays
        pixel_median = np.median(stack, axis=0)

        # Robust threshold using MAD (Median Absolute Deviation)
        global_median = np.median(pixel_median)
        mad = np.median(np.abs(pixel_median - global_median))
        # Scale MAD to equivalent std dev (for normal distribution, std ≈ 1.4826 * MAD)
        mad_std = 1.4826 * mad

        if mad_std < 1e-10:
            # All pixels have the same median — use simple > 0 threshold
            threshold = 0
            hot_mask = pixel_median > threshold
        else:
            threshold = global_median + sigma * mad_std
            hot_mask = pixel_median > threshold

        num_hot = int(np.sum(hot_mask))
        logger.info(f"Hot pixel detection (dark): {num_hot} pixels above threshold "
                    f"{threshold:.1f} (median={global_median:.1f}, MAD_std={mad_std:.1f}, "
                    f"sigma={sigma}, from {len(frames)} frames)")
        return hot_mask

    def apply_to_image(self, image):
        """
        Apply mask to image for display. Returns a copy with masked pixels set to 0.
        Sets shape_mismatch_info on first mismatch for caller to show a dialog.
        """
        if self.mask is None:
            return image

        mask = self.mask
        if mask.shape != image.shape:
            if not self._shape_mismatch_warned:
                self._shape_mismatch_warned = True
                self.shape_mismatch_info = (mask.shape, image.shape)
            mask = self._resize_mask(mask, image.shape)

        result = image.copy()
        result[mask] = 0
        return result

    def clear_mask(self):
        """Clear the active mask."""
        self.mask = None
        self.mask_sources = []
        # Remove saved file
        default_path = os.path.join(self.masks_dir, self.DEFAULT_MASK_FILENAME)
        if os.path.exists(default_path):
            os.remove(default_path)
            logger.info(f"Removed active mask file: {default_path}")
        self.mask_path = None

    @property
    def num_masked_pixels(self):
        if self.mask is None:
            return 0
        return int(np.sum(self.mask))

    @property
    def mask_fraction(self):
        if self.mask is None:
            return 0.0
        return float(np.sum(self.mask)) / self.mask.size
