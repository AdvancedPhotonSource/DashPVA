"""Background buffering helpers for streaming large frame sequences.

Provides :class:`FramePrefetcher`, a QThread worker that streams per-frame 3D
clouds for folder playback so the UI never blocks computing reciprocal-space
maps (video-style buffering with a bounded sliding window).
"""

import numpy as np
from PyQt5.QtCore import QMutex, QObject, QWaitCondition, pyqtSignal, pyqtSlot


class FramePrefetcher(QObject):
    """Background loader that streams per-frame 3D clouds for folder playback.

    Runs on its own ``QThread`` and computes frame clouds (reciprocal-space maps)
    off the UI thread. The UI calls :meth:`request` with the indices it wants (a
    sliding window, target-first); the worker loads them and emits
    ``frame_loaded(index, points, intensities)`` for each. This keeps memory and
    latency bounded for folders with thousands of frames.

    Example:
        >>> pf = FramePrefetcher(files)                          # doctest: +SKIP
        >>> pf.frame_loaded.connect(on_loaded)                   # doctest: +SKIP
        >>> pf.moveToThread(t); t.started.connect(pf.run); t.start()  # doctest: +SKIP
        >>> pf.request([5, 6, 7])   # load frame 5 first, then 6, 7  # doctest: +SKIP
    """

    frame_loaded = pyqtSignal(int, object, object)  # index, points, intensities
    finished = pyqtSignal()

    def __init__(self, files, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self._pending = []            # indices to load, in priority order
        self._stop = False
        self._mutex = QMutex()
        self._cond = QWaitCondition()

    def request(self, indices):
        """Replace the pending queue with `indices` (priority order) and wake."""
        self._mutex.lock()
        try:
            self._pending = [int(i) for i in indices]
            self._cond.wakeOne()
        finally:
            self._mutex.unlock()

    def stop(self):
        """Ask the run loop to exit and wake it."""
        self._mutex.lock()
        try:
            self._stop = True
            self._cond.wakeOne()
        finally:
            self._mutex.unlock()

    @pyqtSlot()
    def run(self):
        from dashpva.utils.rsm_converter import RSMConverter
        conv = RSMConverter()
        while True:
            self._mutex.lock()
            while not self._pending and not self._stop:
                self._cond.wait(self._mutex)
            if self._stop:
                self._mutex.unlock()
                break
            index = self._pending.pop(0)
            self._mutex.unlock()

            if index < 0 or index >= len(self.files):
                continue
            try:
                points, intensities, _num, _shape = conv.load_h5_to_3d(self.files[index])
                pts = np.asarray(points, dtype=float)
                ints = np.asarray(intensities, dtype=float).reshape(-1)
            except Exception:
                continue
            self._mutex.lock()
            stopping = self._stop
            self._mutex.unlock()
            if stopping:
                break
            self.frame_loaded.emit(index, pts, ints)
        self.finished.emit()
