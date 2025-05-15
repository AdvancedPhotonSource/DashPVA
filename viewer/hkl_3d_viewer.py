import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import pyvistaqt as pvqt
from pyvistaqt import QtInteractor
import pyvista as pv
from pva_reader import PVAReader

class HKL3DViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HKL PyVista Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.all_points = None
        self.all_intensity = None

        self.pva_reader = PVAReader(config_filepath='pv_configs/metadata_pvs.toml')
        self.pva_reader.start_channel_monitor()

        self.initUI()
        self.initTimer()

    def initUI(self):
        self.frame = QWidget()
        self.layout = QVBoxLayout()

        # PyVista rendering widget
        pv.set_plot_theme('dark')

        self.plotter = QtInteractor(self.frame)
        self.layout.addWidget(self.plotter.interactor)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        # Add axes and initial empty point cloud
        self.point_cloud = None
        self.plotter.show_bounds(xtitle='H Axis', ytitle='K Axis', ztitle='L Axis')
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')

    def initTimer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # update every 1000 ms

    def update_plot(self):
        try:
            # Skip if nothing cached
            if not len(self.pva_reader.cache_images) > 0:
                return
            
            # Collect all cached data
            all_points = []
            all_intensity = []

            for img, qx, qy, qz in zip(
                self.pva_reader.cache_images,
                self.pva_reader.cache_qx,
                self.pva_reader.cache_qy,
                self.pva_reader.cache_qz
            ):
                flat_points = np.column_stack((
                    qx.flatten(), qy.flatten(), qz.flatten()
                ))
                flat_intensity = img.flatten()

                all_points.append(flat_points)
                all_intensity.append(flat_intensity)

            all_points = np.vstack(all_points)
            all_intensity = np.concatenate(all_intensity)

            # Normalize intensity
            norm_intensity = (all_intensity - all_intensity.min()) / (all_intensity.ptp())
            scalars = (norm_intensity * 255).astype(np.uint8)

            cloud = pv.PolyData(all_points)
            cloud["intensity"] = scalars

            # First-time setup
            if self.point_cloud is None:
                self.point_cloud = self.plotter.add_points(
                    cloud,
                    scalars="intensity",
                    cmap="jet",
                    render_points_as_spheres=False,
                    point_size=10,
                    opacity=0.8,
                )
            else:
                self.plotter.update_coordinates(all_points, render=False)
                self.plotter.update_scalars(scalars, render=False)
                self.plotter.render()

        except Exception as e:
            print(f"[Viewer] Failed to update 3D plot: {e}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HKL3DViewer()
    viewer.show()
    sys.exit(app.exec_())
