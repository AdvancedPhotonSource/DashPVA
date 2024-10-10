import numpy as np
import open3d as o3d

# Create a point cloud from 4D array (x, y, z, color)
def create_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    
    # Use only the first 3 columns for x, y, z coordinates
    red_points =points[:, :3]
    red_points = red_points.astype(np.float64)
    point_cloud.points = o3d.utility.Vector3dVector(red_points)  # x, y, z

    # The fourth column is treated as color; normalize to grayscale (0, 1) for RGB
    colors = points[:, 3]
    rcolor = abs(1-colors)
    gcolor = colors*0
    bcolor = colors*1
    normalized_colors = np.stack([rcolor, gcolor, bcolor], axis=1) # grayscale to RGB
    normalized_colors = normalized_colors.astype(np.float64)
    point_cloud.colors = o3d.utility.Vector3dVector(normalized_colors)
    
    return point_cloud

# Filter the point cloud based on the color range
def filter_point_cloud_by_color(points, min_color, max_color):
    # Filter based on the color (4th column)
    mask = (points[:, 3] >= min_color) & (points[:, 3] <= max_color)
    
    # Return only the filtered points (x, y, z, color)
    return points[mask]

# Example interactive visualizer with color filtering
def visualize_point_cloud_with_color_filter(points):
    # Create the initial point cloud
    original_points = points.copy()  # Keep a copy of the original points
    point_cloud = create_point_cloud(points)
    
    # Create a visualizer window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Add point cloud to the visualizer
    vis.add_geometry(point_cloud)
    
    # Add a coordinate frame with x, y, z axes
   # axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
   #vis.add_geometry(axis_frame)


    # Function to update the point cloud based on color filtering
    def update_color_filter(vis, min_color, max_color):
        filtered_points = filter_point_cloud_by_color(original_points, min_color, max_color)
        if filtered_points.size > 0:  # Ensure that some points remain after filtering
            filtered_points_xyz=filtered_points[:, :3]
            filtered_points_xyz=filtered_points_xyz.astype(np.float64)
            point_cloud.points = o3d.utility.Vector3dVector(filtered_points_xyz)  # x, y, z
            colors = filtered_points[:, 3]
            rcolor = abs(1-colors)
            gcolor = colors*0
            bcolor = colors*1 
            normalized_colors = np.stack([rcolor, gcolor, bcolor], axis=1)
            normalized_colors = normalized_colors.astype(np.float64)
            point_cloud.colors = o3d.utility.Vector3dVector(normalized_colors)
            vis.update_geometry(point_cloud)
            #vis.add_geometry(axis_frame)  # Re-add the axis frame
            vis.poll_events()
            vis.update_renderer()

    # Key events to change color filtering
    color_range = [0.0, 1.0]  # Initial range
    def increase_min_color(vis):
        color_range[0] = min(color_range[0] + 0.01, color_range[1])
        update_color_filter(vis, color_range[0], color_range[1])

    def decrease_min_color(vis):
        color_range[0] = max(color_range[0] - 0.01, 0.0)
        update_color_filter(vis, color_range[0], color_range[1])

    def increase_max_color(vis):
        color_range[1] = min(color_range[1] + 0.01, 1.0)
        update_color_filter(vis, color_range[0], color_range[1])

    def decrease_max_color(vis):
        color_range[1] = max(color_range[1] - 0.01, color_range[0])
        update_color_filter(vis, color_range[0], color_range[1])

    # Bind keys to color filter functions
    vis.register_key_callback(ord(','), lambda vis: decrease_min_color(vis))
    vis.register_key_callback(ord('.'), lambda vis: increase_min_color(vis))
    vis.register_key_callback(ord('['), lambda vis: decrease_max_color(vis))
    vis.register_key_callback(ord(']'), lambda vis: increase_max_color(vis))

    # Start the visualizer loop
    vis.run()
    vis.destroy_window()

# Generate random point cloud data (x, y, z, color) for demonstration
X, Y, Z = np.mgrid[0:1:100j, 0:1:100j, 0:1:100j]
values =    np.sin(np.pi*X) * np.cos(np.pi*Z) * np.cos(np.pi*Z)* np.sin(np.pi*Y)
x=X.flatten()
y=Y.flatten()
z=Z.flatten()
value=values.flatten()
x=np.transpose(x) 
points =  np.stack([x, y, z, value], axis=1)



#num_points = 10000000
#points = np.random.rand(num_points, 4)  # Random points with color in the range [0, 1]
#points[:,0]=points[:,0]*20
#points = np.array([[0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2],[0.3,0.3,0.3,0.3],[0.4,0.4,0.4,0.4],[0.5,0.5,0.5,0.5],[0.6,0.6,0.6,0.6],[0.7,0.7,0.7,0.7],[0.8,0.8,0.8,0.8],[0.9,0.9,0.9,0.9]])


# Run the visualization with interactive color filtering
visualize_point_cloud_with_color_filter(points)
