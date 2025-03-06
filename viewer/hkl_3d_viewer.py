import sys
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# from pyqtgraph import colormap
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow
#TODO: add axis, legend, and labels

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

    def show_3d_plot(self) -> None:
        # open_3d_plot(self.qx, self.qy, self.qz, self.intensity)
        points = np.column_stack((self.qx, self.qy, self.qz))
        
        # Normalize qz for color mapping
        intensity_min, intensity_max = np.min(self.intensity), np.max(self.intensity)
        norm_intensity = (self.intensity - intensity_min) / (intensity_max - intensity_min)

        # Apply a colormap
        cmap = plt.get_cmap("jet")
        colors = cmap(norm_intensity)[:, :3]  # Extract RGB


        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)


        # Show the Open3D plot
        try:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
            o3d.visualization.draw_geometries([pcd, axes])

            # o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            print(f'Failed to perform visualization:{e}')
            sys.exit(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Q-space data using Open3D.")
    parser.add_argument("--qx-file", type=str, required=True, help="Path to NumPy file containing qx array.")
    parser.add_argument("--qy-file", type=str, required=True, help="Path to NumPy file containing qy array.")
    parser.add_argument("--qz-file", type=str, required=True, help="Path to NumPy file containing qz array.")
    parser.add_argument("--intensity-file", type=str, required=True, help="Path to NumPy file containing Intensity array.")

    args = parser.parse_args()

    try:
        qx = np.load(args.qx_file)
        qy = np.load(args.qy_file)
        qz = np.load(args.qz_file)
        intensity = np.load(args.intensity_file)
    except Exception as e:
        print(f"Failed to Load Numpy File: {e}")

    app = QApplication(sys.argv)
    window = HKL3DViewer(qx, qy,qz,intensity)
    window.show()
    sys.exit(app.exec_())
