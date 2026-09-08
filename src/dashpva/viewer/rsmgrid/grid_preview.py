# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""Isometric wireframe preview of the grid box for the RSM Volume Builder.

Shows the two things the numbers alone do not: the *shape* of the requested HKL
box (a thin slab and a cube both read as "3 ranges" in spin boxes) and how
finely it is subdivided.

Grid lines are capped at RSM_GRID_PREVIEW_MAX_DIVISIONS per axis. Past that they
merge into a solid block and stop conveying anything, so above the cap the
drawing is explicitly labelled as thinned rather than pretending a 500-voxel
axis is being shown.
"""
import math

from PyQt5.QtCore import QLineF, QPointF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget

import dashpva.settings as app_settings
from dashpva.gui import theme_colors

# Isometric basis: +H right-and-down, +K left-and-down, +L up. The camera sits
# at (+H, +K, +L), so the visible faces are h=1, k=1 and l=1 and they meet at
# the box's front vertex.
_COS30 = math.cos(math.radians(30.0))
# Floor on a normalized axis length, so a very flat slab still draws as a solid
# rather than collapsing to a line.
_MIN_ASPECT = 0.08
_CAPTION_HEIGHT = 15
_MARGIN = 10


class GridBoxPreview(QWidget):
    """Isometric sketch of the grid box and its subdivisions.

    Usage::

        preview = GridBoxPreview()
        preview.set_grid((200, 200, 200))                 # auto range: a cube
        preview.set_grid((400, 400, 100),                 # explicit HKL box
                         bounds=(0.9, 1.1, 0.9, 1.1, 0.5, 2.5))
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shape = (2, 2, 2)
        self._bounds = None
        self.setMinimumSize(QSize(150, 120))

    def set_grid(self, shape, bounds=None) -> None:
        """Set the voxel counts and, when the user typed one, the HKL box.

        Args:
            shape: (nx, ny, nz) voxel counts.
            bounds: (hmin, hmax, kmin, kmax, lmin, lmax), or None for auto
                range -- the extent is unknown until the files are scanned, so
                the box is drawn as a cube and labelled as such.
        """
        self._shape = tuple(max(2, int(value)) for value in shape)
        self._bounds = tuple(float(value) for value in bounds) if bounds else None
        self.update()

    # -- geometry ----------------------------------------------------------

    def _axis_lengths(self):
        """Normalized (a, b, c) so the box proportions match the HKL widths."""
        if self._bounds is None:
            return 1.0, 1.0, 1.0
        hmin, hmax, kmin, kmax, lmin, lmax = self._bounds
        widths = [hmax - hmin, kmax - kmin, lmax - lmin]
        longest = max(widths)
        if not math.isfinite(longest) or longest <= 0:
            return 1.0, 1.0, 1.0
        return tuple(
            max(_MIN_ASPECT, width / longest) if width > 0 else _MIN_ASPECT
            for width in widths
        )

    def _projector(self):
        """Map (h, k, l) in [0, 1]^3 to widget coordinates, fitted to the box."""
        a, b, c = self._axis_lengths()

        def raw(h, k, l):  # noqa: E741 -- l is the HKL axis, not a digit
            return ((h * a - k * b) * _COS30, (h * a + k * b) * 0.5 - l * c)

        corners = [raw(h, k, l)
                   for h in (0, 1) for k in (0, 1) for l in (0, 1)]  # noqa: E741
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)

        area = self.rect().adjusted(
            _MARGIN, _MARGIN, -_MARGIN, -(_MARGIN + _CAPTION_HEIGHT))
        scale = min(area.width() / max(span_x, 1e-6),
                    area.height() / max(span_y, 1e-6))
        offset_x = area.center().x() - (min(xs) + max(xs)) / 2.0 * scale
        offset_y = area.center().y() - (min(ys) + max(ys)) / 2.0 * scale

        def project(h, k, l):  # noqa: E741
            x, y = raw(h, k, l)
            return QPointF(x * scale + offset_x, y * scale + offset_y)

        return project

    def _divisions(self):
        """Cells to draw per axis, and whether that thins the real grid."""
        cap = app_settings.RSM_GRID_PREVIEW_MAX_DIVISIONS
        return tuple(min(n, cap) for n in self._shape), any(
            n > cap for n in self._shape)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        project = self._projector()
        divisions, thinned = self._divisions()

        self._fill_faces(painter, project)
        self._draw_grid_lines(painter, project, divisions)
        self._draw_edges(painter, project)
        self._draw_axis_labels(painter, project)
        self._draw_caption(painter, thinned)
        painter.end()

    def _fill_faces(self, painter, project):
        """Shade the three visible faces so the box reads as a solid."""
        faces = (
            [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],   # top,   l = 1
            [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],   # right, h = 1
            [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],   # left,  k = 1
        )
        shades = (18, 34, 26)
        painter.setPen(Qt.NoPen)
        for face, alpha in zip(faces, shades):
            colour = QColor(theme_colors.INFO)
            colour.setAlpha(alpha)
            painter.setBrush(colour)
            painter.drawPolygon(QPolygonF([project(*corner) for corner in face]))
        painter.setBrush(Qt.NoBrush)

    def _draw_grid_lines(self, painter, project, divisions):
        """Interior subdivisions on each visible face."""
        nx, ny, nz = divisions
        painter.setPen(QPen(QColor(theme_colors.TEXT_MUTED), 0.6))
        lines = []
        for i in range(1, nx):
            t = i / nx
            lines.append((project(t, 0, 1), project(t, 1, 1)))   # top
            lines.append((project(t, 1, 0), project(t, 1, 1)))   # left
        for i in range(1, ny):
            t = i / ny
            lines.append((project(0, t, 1), project(1, t, 1)))   # top
            lines.append((project(1, t, 0), project(1, t, 1)))   # right
        for i in range(1, nz):
            t = i / nz
            lines.append((project(1, 0, t), project(1, 1, t)))   # right
            lines.append((project(0, 1, t), project(1, 1, t)))   # left
        for start, end in lines:
            painter.drawLine(QLineF(start, end))

    def _draw_edges(self, painter, project):
        """The six silhouette edges plus the three meeting at the front vertex."""
        painter.setPen(QPen(QColor(theme_colors.TEXT_SECONDARY), 1.4))
        edges = (
            ((0, 0, 1), (1, 0, 1)), ((1, 0, 1), (1, 0, 0)), ((1, 0, 0), (1, 1, 0)),
            ((1, 1, 0), (0, 1, 0)), ((0, 1, 0), (0, 1, 1)), ((0, 1, 1), (0, 0, 1)),
            ((1, 1, 1), (1, 0, 1)), ((1, 1, 1), (0, 1, 1)), ((1, 1, 1), (1, 1, 0)),
        )
        for start, end in edges:
            painter.drawLine(QLineF(project(*start), project(*end)))

    def _draw_axis_labels(self, painter, project):
        """H, K and L against the silhouette edge running along each."""
        font = QFont(painter.font())
        font.setPointSize(max(7, font.pointSize() - 1))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(theme_colors.INFO))
        placements = (
            ("H", (0, 0, 1), (1, 0, 1), 4, -10),
            ("K", (0, 0, 1), (0, 1, 1), -12, -10),
            ("L", (1, 0, 0), (1, 0, 1), 5, 0),
        )
        for label, start, end, dx, dy in placements:
            first, second = project(*start), project(*end)
            midpoint = (first + second) / 2.0
            painter.drawText(QPointF(midpoint.x() + dx, midpoint.y() + dy), label)

    def _draw_caption(self, painter, thinned):
        font = QFont(painter.font())
        font.setPointSize(max(7, font.pointSize() - 1))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(
            theme_colors.WARNING if thinned else theme_colors.TEXT_SECONDARY))
        if self._bounds is None:
            caption = "auto range"
        else:
            hmin, hmax, kmin, kmax, lmin, lmax = self._bounds
            nx, ny, nz = self._shape
            caption = "Δ {:.4g} × {:.4g} × {:.4g} rlu".format(
                (hmax - hmin) / (nx - 1),
                (kmax - kmin) / (ny - 1),
                (lmax - lmin) / (nz - 1),
            )
        if thinned:
            caption += " · lines thinned"
        painter.drawText(
            self.rect().adjusted(_MARGIN, 0, -_MARGIN, -2),
            Qt.AlignBottom | Qt.AlignHCenter,
            caption,
        )
