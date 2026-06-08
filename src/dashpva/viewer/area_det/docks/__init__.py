"""Area Detector 2D viewer docks."""

from dashpva.viewer.area_det.docks.analysis_dock import AnalysisDock
from dashpva.viewer.area_det.docks.blob_tracking_dock import BlobTrackingDock
from dashpva.viewer.area_det.docks.image_dock import ImageDock
from dashpva.viewer.area_det.docks.mask_dock import MaskDock
from dashpva.viewer.area_det.docks.mouse_pos_dock import MousePosDock
from dashpva.viewer.area_det.docks.roi_dock import RoiDock
from dashpva.viewer.area_det.docks.stats_dock import StatsDock

__all__ = [
    'StatsDock', 'MousePosDock', 'ImageDock',
    'RoiDock', 'AnalysisDock', 'MaskDock', 'BlobTrackingDock',
]
