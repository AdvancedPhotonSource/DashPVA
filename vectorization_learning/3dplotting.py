import numpy as np
import open3d as o3d

# Generate sample data
X, Y, Z = np.mgrid[0:1:100j, 0:1:100j, 0:1:100j]
values = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.cos(np.pi * Z)

# Prepare points and colors
points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
colors = values.flatten()
colors_normalized = (colors - colors.min()) / (colors.max() - colors.min())
colors_rgb = np.zeros((colors_normalized.size, 3))
colors_rgb[:, 0] = colors_normalized  # Map values to red channel

# Create point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors_rgb)

# Visualize
o3d.visualization.draw_geometries([point_cloud])