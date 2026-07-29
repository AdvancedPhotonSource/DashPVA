# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from .base_tab import BaseTab


class Tab1D(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "1D")
