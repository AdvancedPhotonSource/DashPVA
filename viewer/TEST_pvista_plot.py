import pyvista as pv
import numpy as np

# Create a PyVista plotter
plotter = pv.Plotter()

x = np.load('qx.npy')
y = np.load('qy.npy')
z = np.load('qz.npy')
intensity = np.load('intensity.npy')

# Generate random 3D points
points = np.column_stack((x, y, z))

cloud = pv.PolyData(points)
cloud['point_color'] = intensity  # just use z coordinate

pv.plot(cloud, scalars='point_color', cmap='jet', show_bounds=True, cpos='yz')
