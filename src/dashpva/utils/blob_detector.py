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

        img = image.astype(np.float64)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img8 = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            img8 = np.zeros_like(image, dtype=np.uint8)

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
