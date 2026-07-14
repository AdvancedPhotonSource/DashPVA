from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget

from dashpva.viewer.core.docks.base_dock import BaseDock


class Playback3DDock(BaseDock):
    """Dockable panel that hosts the 3D folder-playback controls.

    The playback group box (``gb_playback``) is defined in the 3D workspace's
    ``.ui`` and simply *reparented* into this dock, so every playback-engine
    reference on ``Workspace3D`` (``self.slider_frame``, ``self.sb_fps``,
    ``self.cb_playback_signal``, the signal plot, …) keeps working unchanged —
    this only relocates the panel out of the tab's control column.

    Example:
        >>> dock = Playback3DDock(main_window=mw, content=mw.tab_3d.gb_playback)  # doctest: +SKIP
        >>> dock.show(); dock.raise_()                                           # doctest: +SKIP
    """

    def __init__(self, main_window=None, content=None, title: str = "3D Playback",
                 segment_name: str = "3d", dock_area=Qt.RightDockWidgetArea,
                 show: bool = False):
        super().__init__(title=title, main_window=main_window,
                         segment_name=segment_name, dock_area=dock_area, show=show)
        self.setWidget(content if content is not None else QWidget(self))
