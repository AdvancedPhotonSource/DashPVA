from .base_tab import BaseTab

class Tab2D(BaseTab):
    def __init__(self, parent=None, main_window=None, title="2D View1"):
        super().__init__(ui_file='gui/workbench/workspace/workspace_2d.ui', parent=parent, main_window=main_window, title=title)