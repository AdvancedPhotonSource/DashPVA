"""Area Detector 2D viewer docks."""

from dashpva.viewer.area_det.docks.analysis_dock import AnalysisDock
from dashpva.viewer.area_det.docks.beam_fit_dock import BeamFitDock
from dashpva.viewer.area_det.docks.image_dock import ImageDock
from dashpva.viewer.area_det.docks.line_cut_dock import LineCutDock
from dashpva.viewer.area_det.docks.mask_dock import MaskDock
from dashpva.viewer.area_det.docks.mouse_pos_dock import MousePosDock
from dashpva.viewer.area_det.docks.roi_2d_plot_dock import AreaDetRoi2DPlotDock
from dashpva.viewer.area_det.docks.roi_dock import RoiDock
from dashpva.viewer.area_det.docks.roi_plot_dock import AreaDetRoiPlotDock
from dashpva.viewer.area_det.docks.software_roi_dock import SoftwareRoiDock
from dashpva.viewer.area_det.docks.stats_dock import StatsDock
from dashpva.viewer.area_det.docks.waterfall_dock import WaterfallDock

__all__ = [
    'StatsDock', 'MousePosDock', 'ImageDock',
    'RoiDock', 'AnalysisDock', 'MaskDock', 'WaterfallDock', 'BeamFitDock',
    'SoftwareRoiDock', 'AreaDetRoiPlotDock', 'AreaDetRoi2DPlotDock', 'LineCutDock',
]
