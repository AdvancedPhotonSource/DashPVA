import sys
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

class ImageWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ImageWindow, self).__init__()
        self.setWindowTitle('Image Viewer with Dynamic Scatter Plot')
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget and set it as the central widget
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create a horizontal layout for the central widget
        self.layout = QtWidgets.QHBoxLayout(self.central_widget)

        # Create a GraphicsLayoutWidget
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_layout)

        # Add a plot to the graphics layout in the first column
        self.plot = self.graphics_layout.addPlot(row=0, col=0)
        self.plot.setAspectLocked()
        self.plot.enableAutoRange()

        # Create an ImageItem (required for ColorBarItem)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Create a ScatterPlotItem
        self.scatter_plot = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter_plot)

        # Create a ColorBarItem
        cmap = pg.colormap.get('viridis')
        self.colorbar = pg.ColorBarItem(values=(0, 1), colorMap=cmap, label='Intensity')
        self.colorbar.setImageItem(self.image_item)  # Link to the ImageItem

        # Add the ColorBarItem to the graphics layout in the second column
        self.graphics_layout.addItem(self.colorbar, row=0, col=1)

        # Initialize the data lists
        self.x_data = []
        self.y_data = []
        self.colors_data = []

        # Initialize the update counter and timer
        self.update_counter = 0
        self.max_updates = 10  # Stop after 10 updates
        self.timer_interval = 1000  # Timer interval in milliseconds (1000 ms = 1 s)

        # Set up the timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.timer_interval)

        # Load and display the image (optional)
        self.load_image()

    def load_image(self):
        # For demonstration, create a random image
        image = np.random.rand(512, 512)
        # self.image_item.setImage(image.T)
        # If you don't want to display an image, you can skip this step

    def update_plot(self):
        # Increment the update counter
        self.update_counter += 1

        # Generate new random data
        num_points = np.random.randint(5, 15)  # Random number of points between 5 and 15
        x_new = np.random.uniform(-200, 700, num_points)  # Allow some points to be out of initial range
        y_new = np.random.uniform(-200, 700, num_points)
        colors_new = np.random.uniform(0, 10, num_points)

        # Append new data to existing data
        self.x_data.extend(x_new)
        self.y_data.extend(y_new)
        self.colors_data.extend(colors_new)

        # Normalize colors for colormap
        colors_array = np.array(self.colors_data)
        norm_colors = (colors_array - colors_array.min()) / (colors_array.max() - colors_array.min())

        # Get colormap
        cmap = pg.colormap.get('viridis')

        # Map normalized colors to brushes
        brushes = [pg.mkBrush(cmap.map(c, mode='qcolor')) for c in norm_colors]

        # Update scatter plot data
        spots = [{'pos': (self.x_data[i], self.y_data[i]), 'brush': brushes[i], 'size': 10} for i in range(len(self.x_data))]
        self.scatter_plot.setData(spots)

        # Update colorbar levels
        self.colorbar.setLevels((colors_array.min(), colors_array.max()))
        self.colorbar.setColorMap(cmap)
        # Update ImageItem levels (required for ColorBarItem)
        self.image_item.setLevels([colors_array.min(), colors_array.max()])

        # Auto-range the plot to include all data
        self.plot.enableAutoRange()

        print(f"Plot updated {self.update_counter} time(s).")

        # Check if the maximum number of updates has been reached
        if self.update_counter >= self.max_updates:
            self.timer.stop()
            print("Plot updates stopped after 10 seconds.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    sys.exit(app.exec_())
