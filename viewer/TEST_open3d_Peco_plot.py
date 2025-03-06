import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from matplotlib import cm

# =============================================================================
# Global Parameters (adjust as needed)
# =============================================================================
slice_width = 100       # width of each 2D slice (x-axis)
slice_height = 100      # height of each 2D slice (y-axis)
voxel_size = 0.5        # voxel downsampling size (controls point density)
min_intensity = 0.0     # minimum intensity value for color mapping
max_intensity = 1.0     # maximum intensity value for color mapping

# =============================================================================
# Function Definitions
# =============================================================================
def generate_slice(width: int, height: int, z_index: int) -> np.ndarray:
    """
    Generate a synthetic 2D slice representing intensity values.

    This function creates a sine-cosine pattern that shifts with the slice index,
    simulating a varying intensity pattern over time.

    Args:
        width (int): Number of pixels along the x-axis.
        height (int): Number of pixels along the y-axis.
        z_index (int): The current slice index; higher values shift the pattern.

    Returns:
        np.ndarray: 2D array (height x width) of intensity values normalized to [0,1].
    """
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    # Create a pattern that varies with the slice index
    intensity = (np.sin(np.pi * X + z_index / 10.0) * np.cos(np.pi * Y + z_index / 10.0) + 1) / 2.0
    return intensity

def map_intensity_to_color(intensity: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Map intensity values to RGB colors using the 'viridis' colormap.

    This function first normalizes the intensity values between min_val and max_val.
    Then, it converts the normalized intensity to an RGB triplet using matplotlib's
    viridis colormap (discarding the alpha channel).

    Args:
        intensity (np.ndarray): 2D array of intensity values.
        min_val (float): Minimum intensity value for normalization.
        max_val (float): Maximum intensity value for normalization.

    Returns:
        np.ndarray: 3D array of RGB colors with shape (height, width, 3).
    """
    norm_intensity = np.clip((intensity - min_val) / (max_val - min_val), 0, 1)
    colormap = cm.get_cmap("viridis")
    colors = colormap(norm_intensity)[:, :, :3]  # discard alpha channel
    return colors

def volume_to_point_cloud(slices: list) -> o3d.geometry.PointCloud:
    """
    Convert a list of 2D slices into a 3D point cloud.

    The slices are stacked along the z-axis to form a volume.
    Each pixel in a slice becomes a 3D point with coordinates (x, y, z) and a color
    determined by its intensity value (mapped via a colormap).

    Args:
        slices (list): List of 2D numpy arrays representing intensity slices.

    Returns:
        o3d.geometry.PointCloud: A point cloud constructed from the 3D volume.
    """
    # Stack slices along a new axis to create a volume: shape (n_slices, height, width)
    volume = np.stack(slices, axis=0)
    n_slices, height, width = volume.shape
    pts = []
    colors = []
    
    # Iterate through every voxel in the volume
    for z in range(n_slices):
        slice_intensity = volume[z]
        slice_colors = map_intensity_to_color(slice_intensity, min_intensity, max_intensity)
        for y in range(height):
            for x in range(width):
                pts.append([x, y, z])
                colors.append(slice_colors[y, x])
    pts = np.array(pts)
    colors = np.array(colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# =============================================================================
# LivePlot Class for Managing Live Updates
# =============================================================================
class LivePlot:
    """
    Manages the live updating of a 3D point cloud generated from sequential 2D slices.

    Attributes:
        scene_widget (gui.SceneWidget): The 3D scene widget where geometry is rendered.
        volume_slices (list): List of generated 2D slices.
        z_index (int): Counter to track the current slice index.
        pcd (o3d.geometry.PointCloud): The point cloud representing the volume.
        material (rendering.MaterialRecord): Material record used for rendering the point cloud.
    """
    def __init__(self, scene_widget: gui.SceneWidget):
        self.scene_widget = scene_widget
        self.volume_slices = []
        self.z_index = 0
        self.pcd = o3d.geometry.PointCloud()
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        # Add an initial (empty) geometry to the scene with a unique name.
        self.scene_widget.scene.add_geometry("LivePCD", self.pcd, self.material)
    
    def update(self):
        """
        Update the 3D point cloud by generating a new 2D slice, appending it to the volume,
        converting the accumulated volume into a point cloud, and updating the geometry in the scene.
        """
        # Generate a new slice
        new_slice = generate_slice(slice_width, slice_height, self.z_index)
        self.volume_slices.append(new_slice)
        self.z_index += 1
        
        # Convert the current volume into a point cloud and downsample it
        new_pcd = volume_to_point_cloud(self.volume_slices)
        new_pcd = new_pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Update the stored point cloud
        self.pcd.points = new_pcd.points
        self.pcd.colors = new_pcd.colors
        
        # Inform the scene to update the geometry
        self.scene_widget.scene.modify_geometry("LivePCD", self.pcd)

# =============================================================================
# Callback to Update Point Size
# =============================================================================
def on_point_size_change(slider_value: float, scene_widget: gui.SceneWidget):
    """
    Callback function triggered when the point size slider value changes.

    This function accesses the render options of the scene and updates the point size,
    which effectively adjusts the "pixel" size of the rendered points.

    Args:
        slider_value (float): The new point size value from the slider.
        scene_widget (gui.SceneWidget): The scene widget whose render options are updated.
    """
    opt = scene_widget.scene.renderer.get_render_option()
    opt.point_size = slider_value

# =============================================================================
# Layout Callback
# =============================================================================
def layout(window: gui.Window, scene_widget: gui.SceneWidget, panel: gui.Widget):
    """
    Layout callback to position the 3D scene widget and the control panel within the window.

    Args:
        window (gui.Window): The main application window.
        scene_widget (gui.SceneWidget): The widget displaying the 3D scene.
        panel (gui.Widget): The control panel widget containing GUI controls.
    """
    r = window.content_rect
    # Allocate ~75% width to the 3D scene and ~25% to the control panel.
    scene_widget.frame = gui.Rect(r.x, r.y, int(r.width * 0.75), r.height)
    panel.frame = gui.Rect(r.x + int(r.width * 0.75) + 10, r.y, int(r.width * 0.25) - 10, r.height)

# =============================================================================
# Main Application
# =============================================================================
def main():
    """
    Set up and run the Open3D GUI application that:
      - Continuously generates 2D slices and updates a 3D point cloud.
      - Uses GPU-accelerated rendering via OpenGL (leveraging NVIDIA GPUs if available).
      - Provides a slider to adjust the point size in the rendered scene.
      - Overlays text labels (as GUI widgets) and a coordinate frame to indicate axes.
    
    The live update occurs every 2 seconds using a timer callback.
    """
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Live 3D Plot with Controls", 1024, 768)
    
    # Create a SceneWidget for 3D rendering.
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    
    # Set an initial camera view.
    center = np.array([slice_width / 2, slice_height / 2, 5])  # approximate center of volume
    eye = center + np.array([100, 100, 100])
    up = np.array([0, 0, 1])
    scene_widget.scene.camera.look_at(center, eye, up)
    
    # Add a coordinate frame (red: X, green: Y, blue: Z) for axis orientation.
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
    scene_widget.scene.add_geometry("CoordinateFrame", coord_frame, rendering.MaterialRecord())
    
    # Create a control panel to hold GUI elements.
    panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
    
    # Create and add a slider for adjusting point size.
    slider = gui.Slider(gui.Slider.INT)
    slider.set_limits(1, 20)  # point size between 1 and 20 pixels
    slider.int_value = 5     # initial point size
    slider.set_on_value_changed(lambda value: on_point_size_change(value, scene_widget))
    panel.add_child(gui.Label("Point Size"))
    panel.add_child(slider)
    
    # Add text labels for axes (2D overlay labels).
    label_x = gui.Label("X-Axis (Red)")
    label_y = gui.Label("Y-Axis (Green)")
    label_z = gui.Label("Z-Axis (Blue)")
    panel.add_child(label_x)
    panel.add_child(label_y)
    panel.add_child(label_z)
    
    # Set up the layout for the scene widget and panel.
    window.set_on_layout(lambda w: layout(w, scene_widget, panel))
    
    # Create an instance of LivePlot to manage the live-updating point cloud.
    live_plot = LivePlot(scene_widget)
    
    # Define a timer callback that updates the live plot every 2 seconds.
    def timer_callback():
        live_plot.update()
        return gui.TimerCallback.REPEAT
    
    # Create the timer callback (period in milliseconds)
    gui.Application.instance.create_timer_callback(timer_callback, 2000)
    
    # Add the scene widget and control panel to the window.
    window.add_child(scene_widget)
    window.add_child(panel)
    
    # Run the GUI application. The rendering is GPU-accelerated via OpenGL.
    gui.Application.instance.run()

if __name__ == "__main__":
    main()
