import pyvista as pv
import numpy as np

# Set Plot Theme
pv.set_plot_theme('dark')

# Load data
x = np.load('qx.npy')
y = np.load('qy.npy')
z = np.load('qz.npy')
intensity = np.load('intensity.npy')

# Normalize intensity to [0, 1] for colormap
intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

# Generate points
points = np.column_stack((x, y, z))
cloud = pv.PolyData(points)

# Apply colors
cloud['intensity'] = intensity  # Store intensity for scalar mapping

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(
    cloud,
    scalars='intensity',  # Use intensity for the scalar bar
    cmap='viridis',
    show_scalar_bar=True,  # Enable scalar bar
    opacity=intensity_norm,  # Set transparency based on intensity
    # render_points_as_spheres=True, 
    # point_size=10

)
plotter.show_bounds()
plotter.show_axes()
plotter.show()
