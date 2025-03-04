import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow
import sys
import threading
#TODO: add axis, legend, and labels

def visualize_rsm_with_colors(qx, qy, qz, intensity=None):
    # intensity = np.array(intensity)
    points = np.column_stack((qx.ravel(), qy.ravel(), qz.ravel()))
    
    # Normalize qz for color mapping
    intensity_min, intensity_max = np.min(intensity), np.max(intensity)
    norm_intensity = (intensity.ravel() - intensity_min) / (intensity_max - intensity_min)

    # Apply a colormap
    cmap = plt.get_cmap("jet")
    colors = cmap(norm_intensity)[:, :3]  # Extract RGB

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Show the Open3D plot
    o3d.visualization.draw_geometries([pcd])

def open_3d_plot(qx, qy, qz, intensity):
    # Run Open3D in a separate thread
    threading.Thread(target=visualize_rsm_with_colors, args=(qx, qy, qz, intensity), daemon=True).start()

class HKL3DViewer(QMainWindow):
    def __init__(self, qx, qy, qz, intensity):
        super().__init__()
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.intensity = intensity
        self.initUI()

    def initUI(self):
        self.setWindowTitle("HKL Data Viewer")
        self.setGeometry(100, 100, 300, 200)

        # Button to open Open3D visualization
        btn = QPushButton("Show 3D Plot", self)
        btn.setGeometry(80, 80, 140, 40)
        btn.clicked.connect(self.show_3d_plot)

    def show_3d_plot(self):
        open_3d_plot(self.qx, self.qy, self.qz, self.intensity)

# if __name__ == "__main__":
#     # Example dummy data for testing
#     qx, qy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
#     qz = np.sin(4 * np.pi * qx) * np.cos(4 * np.pi * qy)  # Example pattern

#     app = QApplication(sys.argv)
#     window = HKL3DViewer(qx, qy, qz)
#     window.show()
#     sys.exit(app.exec_())
