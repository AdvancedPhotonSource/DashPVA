import cv2
import numpy as np


class BlobDetector:
    """Wraps cv2.SimpleBlobDetector to produce SORT-compatible (N,5) detection arrays."""

    def __init__(self, **params):
        self._params = cv2.SimpleBlobDetector_Params()
        self._detector = None
        if params:
            self.update_params(**params)

    def update_params(self, **params):
        """Update SimpleBlobDetector parameters by name. Rebuilds detector on next detect()."""
        for k, v in params.items():
            if hasattr(self._params, k):
                setattr(self._params, k, type(getattr(self._params, k))(v))
        self._detector = None

    def _build(self):
        # Guard against invalid area params — cv2 requires 0 < minArea <= maxArea.
        # Clamp rather than crash so a bad JSON file doesn't kill the consumer.
        if self._params.filterByArea:
            self._params.minArea = max(1.0, self._params.minArea)
            self._params.maxArea = max(self._params.minArea + 1.0, self._params.maxArea)
        if self._params.filterByCircularity:
            self._params.minCircularity = max(0.0, min(self._params.minCircularity, 1.0))
            self._params.maxCircularity = max(self._params.minCircularity, self._params.maxCircularity)
        if self._params.filterByConvexity:
            self._params.minConvexity = max(0.0, min(self._params.minConvexity, 1.0))
            self._params.maxConvexity = max(self._params.minConvexity, self._params.maxConvexity)
        if self._params.filterByInertia:
            self._params.minInertiaRatio = max(0.0, min(self._params.minInertiaRatio, 1.0))
            self._params.maxInertiaRatio = max(self._params.minInertiaRatio, self._params.maxInertiaRatio)
        self._detector = cv2.SimpleBlobDetector_create(self._params)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect blobs in *image* and return shape (N, 5): [x1, y1, x2, y2, score=1.0].

        Compatible with Sort.update(). Returns np.empty((0, 5)) when no blobs found.
        Image is normalised to uint8 before detection so this works on raw uint16,
        float32, or log-scaled float64 arrays without caller preprocessing.
        Coordinates are in (col, row) / (x, y) convention matching pyqtgraph.
        """
        if self._detector is None:
            self._build()

        # Normalise to uint8 so SimpleBlobDetector thresholds are meaningful.
        # Use the dtype's full range rather than the per-frame min/max: global
        # normalisation is consistent across frames.  Per-frame normalisation
        # stretches contrast differently every frame (Poisson amplitude noise
        # changes min/max each frame), making threshold values inconsistent and
        # causing detection counts to jump wildly between 0 and hundreds.
        if image.dtype == np.uint8:
            img8 = image                                  # already in range, no copy
        elif np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            img8 = (image.astype(np.float32) / info.max * 255).astype(np.uint8)
        else:
            # Float image: clip to [0,1] convention, then scale
            img = np.clip(image.astype(np.float32), 0.0, None)
            hi = img.max()
            img8 = (img / hi * 255).astype(np.uint8) if hi > 0 else np.zeros_like(image, dtype=np.uint8)

        keypoints = self._detector.detect(img8)
        if not keypoints:
            return np.empty((0, 5), dtype=np.float64)

        boxes = []
        for kp in keypoints:
            r = kp.size / 2.0
            cx, cy = kp.pt  # (x=col, y=row)
            boxes.append([cx - r, cy - r, cx + r, cy + r, 1.0])
        return np.array(boxes, dtype=np.float64)

    # ------------------------------------------------------------------
    # Convenience accessors used by the dock to read current param values
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        p = self._params
        return {
            'minThreshold':    p.minThreshold,
            'maxThreshold':    p.maxThreshold,
            'filterByArea':    p.filterByArea,
            'minArea':         p.minArea,
            'maxArea':         p.maxArea,
            'filterByCircularity': p.filterByCircularity,
            'minCircularity':  p.minCircularity,
            'maxCircularity':  p.maxCircularity,
            'filterByConvexity': p.filterByConvexity,
            'minConvexity':    p.minConvexity,
            'maxConvexity':    p.maxConvexity,
            'filterByInertia': p.filterByInertia,
            'minInertiaRatio': p.minInertiaRatio,
            'maxInertiaRatio': p.maxInertiaRatio,
        }
