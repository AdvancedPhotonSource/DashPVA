import os
import logging
import numpy as np
import pathlib

logger = logging.getLogger(__name__)


class MaskManager:
    """
    Manages detector pixel masks for DashPVA.

    Handles loading (.edf, .npy, .tif/.tiff), combining (OR), saving, and applying masks.
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

    def load_mask(self, filepath):
        """
        Load a mask from file. Supports .edf, .npy, .tif/.tiff.

        Any 2D array with two unique values is treated as a binary mask.
        Convention: nonzero = masked (True), zero = good (False).
        This matches pyFAI where 1=masked, 0=good.

        Returns boolean array.
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.npy':
            data = np.load(filepath)
        elif ext in ('.tif', '.tiff'):
            from PIL import Image
            data = np.array(Image.open(filepath))
        elif ext == '.edf':
            import fabio
            data = fabio.open(filepath).data
        else:
            # Try fabio as fallback for other detector formats
            import fabio
            data = fabio.open(filepath).data

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        mask = data.astype(bool)
        logger.info(f"Loaded mask: {filepath}, shape={mask.shape}, "
                    f"masked={np.sum(mask)}/{mask.size}")
        return mask

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
        Detect stuck/dead pixels using temporal variance.

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
        logger.info(f"Dead pixel detection: {num_dead} pixels with variance < {variance_threshold} "
                    f"(from {len(frames)} frames)")
        return dead_mask

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
