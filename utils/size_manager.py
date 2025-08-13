from PyQt5.QtCore import QObject, QEvent
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QApplication, QScrollArea

class SizeManger(QObject):
    """
    Manages how QWidgets, QFonts, and other QObjects automatically 
    scale, when resized
    
    # Usage:
        app = QApplication()
        size_manager = SizeManger(app)
        That's it now you have automatic scaling
    """
    
    def __init__(self, app: QApplication) -> None:
        """
        Initialize the size manager with a QApplication.

        Args:
            app (QApplication): The application to manage for window resize events
        """
        super().__init__()
        
        # Application reference
        self.app = app
        
        # Storage for original values of the widgets
        self.original_font_size = {}
        self.base_window_widths = {}
        
        # Install event monitoring
        self.app.installEventFilter(self)
        
        
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
        Monitors window objects resize events to trigger scaling.

        Args:
            obj (QObject): The object being monitored for events
            event (QEvent): The event that occurred (checking for resize)
        """
        if event.type() == QEvent.Resize and isinstance(obj, QWidget) and obj.isWindow():
            self.scale_widgets(obj)
        return super().eventFilter(obj, event)
    
    
    def scale_widgets(self, window: QWidget) -> None:
        """
        Calculates scale factor and applies scaling to window widgets.

        Args:
            window (QWidget): The window that was resized
        """
        
        # Check that the curr_width and window is within the dict
        curr_width = window.width()
        if window not in self.base_window_widths:
            self.base_window_widths[window] = curr_width
        
        # Create a scale
        base_width = self.base_window_widths[window]
        scale = curr_width / base_width
        self._apply_scaling(window=window, scale=scale)
                        
    
    def _apply_scaling(self, window: QWidget, scale: float) -> None:
        """
        Applies font and geometry scaling to all widgets in the window.

        Args:
            window (QWidget): The window containing widgets to scale
            scale (float): The scaling factor to apply (1.0 = original size)
        """
        
        # Ensures that scaling is 50% < scale < 200% of original size
        # scale = 0.2 then scale = 0.5
        # scale = 3.0 then scale = 2.0
        scale = max(0.5, min(2.0, scale))
        
        # Lazily inits of the storage dictionaries
        if not hasattr(self, 'original_geometries'):
            self.original_geometries = {}
        if not hasattr(self, 'original_sizes'):
            self.original_sizes = {}
        if not hasattr(self, 'original_margins'):
            self.original_margins = {}
        
        # Scales a layout if it's a layout
        if window.layout():
            self._scale_layout(window.layout(), scale)
        
        # Iterates through all the widgets within window
        for widget in window.findChildren(QWidget):
            
            # -- ONLY FOR: scaling for QScrollArea -- #
            if isinstance(widget, QScrollArea):
                
                # Set the values of the widgets original size
                if widget not in self.original_sizes:
                    self.original_sizes[widget] = {
                        'min': widget.minimumSize(),
                        'max': widget.maximumSize(),
                        'size_hint': widget.sizeHint()
                    }
                
                # Scaling the minimum HxW of the widget 
                orig_sizes = self.original_sizes[widget]
                if orig_sizes['min'].width() > 0:
                    widget.setMinimumWidth(int(orig_sizes['min'].width() * scale))
                if orig_sizes['min'].height() > 0:
                    widget.setMinimumHeight(int(orig_sizes['min'].height() * scale))
                
                # Stores the contents of the QScrollArea to 
                # the original size dict.
                if widget.widget():
                    content_widget = widget.widget()
                    if content_widget not in self.original_sizes:
                        self.original_sizes[content_widget] = {
                            'min': content_widget.minimumSize(),
                            'size_hint': content_widget.sizeHint()
                        }
            # ----------------------------------------- #
            
            # Stores a widget font size within a dictionary
            if widget not in self.original_font_size:
                self.original_font_size[widget] = widget.font().pointSizeF()
            
            # Scales and sets the widgets font
            font = widget.font()
            font.setPointSizeF(self.original_font_size[widget] * scale)
            widget.setFont(font)
            
            # Set the values of the widgets original size
            if widget not in self.original_sizes:
                self.original_sizes[widget] = {
                    'min': widget.minimumSize(),
                    'max': widget.maximumSize(),
                    'size_hint': widget.sizeHint()
                }
            
            # Set the scaled Min and Max HxW 
            if widget.parent() and widget.parent().layout():
                orig_sizes = self.original_sizes[widget]
                if orig_sizes['min'].width() > 0:
                    widget.setMinimumWidth(int(orig_sizes['min'].width() * scale))
                if orig_sizes['min'].height() > 0:
                    widget.setMinimumHeight(int(orig_sizes['min'].height() * scale))
                if orig_sizes['max'].width() < 16777215:
                    widget.setMaximumWidth(int(orig_sizes['max'].width() * scale))
                if orig_sizes['max'].height() < 16777215:
                    widget.setMaximumHeight(int(orig_sizes['max'].height() * scale))
            
            # Store the widget in geom dict if parent  is not a layout
            else:
                if widget not in self.original_geometries:
                    self.original_geometries[widget] = widget.geometry()
                
                # Scale the geomtry
                orig_geom = self.original_geometries[widget]
                widget.setGeometry(
                    int(orig_geom.x() * scale),
                    int(orig_geom.y() * scale),
                    int(orig_geom.width() * scale),
                    int(orig_geom.height() * scale)
                )
            
            # Scale layouts
            if widget.layout():
                self._scale_layout(widget.layout(), scale)
    
    
    def _scale_layout(self, layout, scale: float) -> None:
        """
        Scale spacing and margins of a layout proportionally.

        Args:
            layout: The layout object to scale (QLayout subclass)
            scale (float): The scaling factor to apply to spacing and margins
        """
        # Store the original margins and spacing
        if layout not in self.original_margins:
            self.original_margins[layout] = {
                'spacing': layout.spacing(),
                'margins': layout.getContentsMargins()
            }
        
        # Scale and set the spacing
        orig = self.original_margins[layout]
        layout.setSpacing(int(orig['spacing'] * scale))
        
        # Scale and set the margins
        margins = orig['margins']
        layout.setContentsMargins(
            int(margins[0] * scale),
            int(margins[1] * scale),
            int(margins[2] * scale),
            int(margins[3] * scale)
        )
