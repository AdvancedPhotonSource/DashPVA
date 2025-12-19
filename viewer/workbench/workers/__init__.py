from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class Render3D(QObject):
    """Worker that prepares 3D render data off the UI thread and signals the main thread to render."""
    finished = pyqtSignal()
    render_ready = pyqtSignal(object, object, int, tuple)

    def __init__(self, *, points, intensities, num_images, shape, parent=None):
        super().__init__(parent)
        self.points = points
        self.intensities = intensities
        self.num_images = int(num_images) if num_images is not None else 0
        # Normalize shape to a tuple
        try:
            self.shape = tuple(shape) if shape is not None else (0, 0)
        except Exception:
            self.shape = (0, 0)

    @pyqtSlot()
    def run(self):
        """Long-running task placeholder.
        In a real scenario, heavy processing would occur here. For now,
        simply emit the data to the main thread to render, then finish.
        """
        try:
            self.render_ready.emit(self.points, self.intensities, self.num_images, self.shape)
        finally:
            self.finished.emit()
