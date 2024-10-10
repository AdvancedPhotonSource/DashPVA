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

# Code for NVIDIA GPUs below
if not o3d.cuda.is_available():
    print("CUDA is not available. ")
    exit()

# Generate sample data
X, Y, Z = np.mgrid[0:1:100j, 0:1:100j, 0:1:100j]
values = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.cos(np.pi * Z)

# Prepare points and colors
points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
colors = values.flatten()
colors_normalized = (colors - colors.min()) / (colors.max() - colors.min())
colors_rgb = np.zeros((colors_normalized.size, 3))
colors_rgb[:, 0] = colors_normalized  # Map values to red channel

# Convert to Open3D Tensor PointCloud on GPU
pcd_tensor = o3d.t.geometry.PointCloud()
pcd_tensor.point.positions = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CUDA:0"))
pcd_tensor.point.colors = o3d.core.Tensor(colors_rgb, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device("CUDA:0"))

voxel_size = 0.02
pcd_down = pcd_tensor.voxel_down_sample(voxel_size)

pcd_down_legacy = pcd_down.to_legacy()

o3d.visualization.draw_geometries([pcd_down_legacy])
