"""
Deterministic per-frame feature extraction.

Takes a raw image array and the blob detection bounding boxes from
HpcBlobTrackingProcessor and returns a JSON-serializable feature dict
with per-blob shape/intensity metrics and global frame statistics.
"""

import numpy as np


def _compute_circularity(crop: np.ndarray) -> float:
    """Circularity of the dominant contour in a thresholded crop (0–1)."""
    try:
        import cv2
        mn, mx = float(crop.min()), float(crop.max())
        if mx == mn:
            return 0.0
        u8 = ((crop - mn) / (mx - mn) * 255).astype(np.uint8)
        _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            return 0.0
        return round(float(4 * np.pi * area / (perimeter ** 2)), 4)
    except Exception:
        return 0.0


class FrameFeatureExtractor:
    """
    Extracts deterministic numerical features from a detector frame.

    Usage
    -----
    extractor = FrameFeatureExtractor()
    features = extractor.extract(image, blob_detections)
    # features is a JSON-serializable dict
    """

    def extract(self, image: np.ndarray, blob_detections: np.ndarray) -> dict:
        """
        Parameters
        ----------
        image : np.ndarray  shape (H, W) or (H, W, C)
        blob_detections : np.ndarray  shape (N, 5)  [x1, y1, x2, y2, score]

        Returns
        -------
        dict:
            n_blobs : int
            blobs   : list of per-blob dicts
            frame   : global frame stat dict
        """
        img = self._to_float2d(image)
        h, w = img.shape

        frame_feats = self._global_features(img, h, w)
        blob_list = self._per_blob_features(img, blob_detections, h, w)
        radial_peaks = self._radial_profile_peaks(img, h, w)
        bg_texture = self._background_texture(img, h, w)

        return {
            'n_blobs': len(blob_list),
            'blobs': blob_list,
            'frame': {
                **frame_feats,
                'background_texture': bg_texture,
                'radial_profile_peaks': radial_peaks,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_float2d(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        if img.ndim == 3:
            img = img.mean(axis=2)
        return img

    def _global_features(self, img: np.ndarray, h: int, w: int) -> dict:
        total = float(np.sum(img))
        background = float(np.median(img))
        peak_val = float(np.max(img))
        snr = round(peak_val / background, 3) if background > 0 else 0.0

        peak_flat = int(np.argmax(img))
        peak_y, peak_x = divmod(peak_flat, w)

        if total > 0:
            ys = np.arange(h, dtype=np.float64)
            xs = np.arange(w, dtype=np.float64)
            com_y = float(np.dot(ys, img.sum(axis=1)) / total)
            com_x = float(np.dot(xs, img.sum(axis=0)) / total)
        else:
            com_y, com_x = h / 2.0, w / 2.0

        threshold = max(1.0, background * 2.0)
        active_fraction = round(float(np.sum(img > threshold) / img.size), 6)

        return {
            'total_intensity': round(total, 2),
            'background': round(background, 4),
            'snr': snr,
            'peak_x': int(peak_x),
            'peak_y': int(peak_y),
            'com_x': round(com_x, 2),
            'com_y': round(com_y, 2),
            'active_fraction': active_fraction,
        }

    def _radial_profile_peaks(self, img: np.ndarray, h: int, w: int,
                               n_bins: int = 128) -> list:
        """
        Compute radial mean intensity profile from the image center and return
        the prominent peaks as [{r_px, intensity}].  A peak in I(r) indicates
        a powder ring or circular artifact that blob detection would miss.
        """
        try:
            cy, cx = h / 2.0, w / 2.0
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            r_max = min(cx, cy)
            bins = np.linspace(0, r_max, n_bins + 1)
            profile = np.zeros(n_bins)
            for i in range(n_bins):
                mask = (r >= bins[i]) & (r < bins[i + 1])
                if mask.any():
                    profile[i] = float(img[mask].mean())
            # Find local maxima using simple neighbour comparison
            bg = float(np.median(profile))
            threshold = bg * 1.5
            peaks = []
            for i in range(1, n_bins - 1):
                if profile[i] > threshold and profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                    r_px = int(round((bins[i] + bins[i+1]) / 2))
                    peaks.append({'r_px': r_px, 'intensity': round(float(profile[i]), 2)})
            # Keep top 5 peaks by intensity
            peaks.sort(key=lambda p: p['intensity'], reverse=True)
            return peaks[:5]
        except Exception:
            return []

    def _background_texture(self, img: np.ndarray, h: int, w: int,
                             border_fraction: float = 0.2) -> float:
        """
        Std deviation of pixels in the outer border of the image (away from
        the centre where blobs/diffraction live).  Captures diffuse scatter,
        ice rings, or other background structure that blob detection ignores.
        """
        try:
            bh = max(1, int(h * border_fraction))
            bw = max(1, int(w * border_fraction))
            top    = img[:bh, :]
            bottom = img[h - bh:, :]
            left   = img[bh:h - bh, :bw]
            right  = img[bh:h - bh, w - bw:]
            border = np.concatenate([top.ravel(), bottom.ravel(),
                                     left.ravel(), right.ravel()])
            return round(float(np.std(border)), 4)
        except Exception:
            return 0.0

    def _per_blob_features(self, img: np.ndarray, detections: np.ndarray,
                           h: int, w: int) -> list:
        blobs = []
        if detections is None or len(detections) == 0:
            return blobs

        for bbox in detections:
            x1, y1, x2, y2 = bbox[:4]
            x1i = max(0, int(x1))
            y1i = max(0, int(y1))
            x2i = min(w, int(x2))
            y2i = min(h, int(y2))
            bw = x2i - x1i
            bh = y2i - y1i
            if bw <= 0 or bh <= 0:
                continue

            cx = round((x1 + x2) / 2.0, 2)
            cy = round((y1 + y2) / 2.0, 2)
            area = float(bw * bh)
            aspect_ratio = round(bw / bh, 3) if bh > 0 else 1.0

            crop = img[y1i:y2i, x1i:x2i]
            mean_i = round(float(np.mean(crop)), 4)
            max_i  = round(float(np.max(crop)), 4)
            std_i  = round(float(np.std(crop)), 4)
            circularity = _compute_circularity(crop)

            blobs.append({
                'cx': cx, 'cy': cy,
                'w': bw, 'h': bh,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'mean_intensity': mean_i,
                'max_intensity': max_i,
                'std_intensity': std_i,
            })

        return blobs
