import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def create_point_cloud(x, y, z, intensity):
    pts = np.column_stack((x, y, z))
    
    intensity_min = np.min(intensity)
    intensity_max = np.max(intensity)
    norm_intensity = (intensity - intensity_min) / (intensity_max - intensity_min + 1e-8)
    
    colormap = matplotlib.colormaps["viridis"]
    colors = np.array([colormap(val)[:3] for val in norm_intensity])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_axes_with_labels(length=10, origin=[0,0,0]):
    """ Create coordinate axes with labels and tick marks in both directions """
    lines = []
    colors = []

    # Define axis end points in both directions
    points = [
        origin,      # Origin
        [origin[0]+length, origin[1], origin[2]], # +X-axis
        [origin[0]-length, origin[1], origin[2]], # -X-axis
        [origin[0], origin[1]+length, origin[2]], # +Y-axis
        [origin[0], origin[1]-length, origin[2]], # -Y-axis
        [origin[0], origin[1], origin[2]+length], # +Z-axis
        [origin[0], origin[1], origin[2]-length]  # -Z-axis
    ]
    
    # Define axis lines (positive and negative directions)
    lines.append([0, 1])  # X-axis (positive)
    lines.append([0, 2])  # X-axis (negative)
    lines.append([0, 3])  # Y-axis (positive)
    lines.append([0, 4])  # Y-axis (negative)
    lines.append([0, 5])  # Z-axis (positive)
    lines.append([0, 6])  # Z-axis (negative)

    # Assign colors (red = X, green = Y, blue = Z)
    colors.append([1, 0, 0])  # Red for +X
    colors.append([1, 0, 0])  # Red for -X
    colors.append([0, 1, 0])  # Green for +Y
    colors.append([0, 1, 0])  # Green for -Y
    colors.append([0, 0, 1])  # Blue for +Z
    colors.append([0, 0, 1])  # Blue for -Z

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def main():
    # Example data: replace these with your actual data.
    N = 1000
    x = np.random.rand(N) * 100
    y = np.random.rand(N) * 100
    z = np.random.rand(N) * 100
    intensity = np.random.rand(N)
    
    pcd = create_point_cloud(x, y, z, intensity)
    
    pts_np = np.asarray(pcd.points)
    print("Point cloud range: min =", np.min(pts_np, axis=0), 
          "max =", np.max(pts_np, axis=0))
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=pcd.get_center())
    
    axes = create_axes_with_labels(length=100, origin=pcd.get_center())  # Create axes extending in both directions

    o3d.visualization.draw_geometries([pcd, coord_frame, axes])

if __name__ == "__main__":
    main()
