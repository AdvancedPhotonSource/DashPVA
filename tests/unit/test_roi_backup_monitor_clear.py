# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""_clear_roi_backup_monitor clears exactly the PVs recorded in
self._active_roi_pvs by start_roi_backup_monitor, rather than
re-deriving names from self.rois/self.pva_prefix.
"""

from unittest.mock import patch

import pytest


def _reader_with_active_pvs(pvs):
    pytest.importorskip("pvaccess")
    pytest.importorskip("PyQt5")
    from PyQt5.QtCore import QObject

    from dashpva.utils.pva_reader import PVAReader

    reader = PVAReader.__new__(PVAReader)
    QObject.__init__(reader)
    reader.pva_prefix = "13SIM1"
    reader._active_roi_pvs = list(pvs)
    return reader


class TestClearRoiBackupMonitor:

    def test_clears_only_active_pvs(self):
        pvs = [
            "13SIM1:ROI1:MinX",
            "13SIM1:ROI1:MinY",
            "13SIM1:ROI1:SizeX",
            "13SIM1:ROI1:SizeY",
        ]
        reader = _reader_with_active_pvs(pvs)
        with patch("dashpva.utils.pva_reader.camonitor_clear") as mock_clear:
            reader._clear_roi_backup_monitor()
        cleared = {call.args[0] for call in mock_clear.call_args_list}
        assert cleared == set(pvs)

    def test_clears_and_resets_active_pvs(self):
        reader = _reader_with_active_pvs(["13SIM1:ROI2:MinX"])
        with patch("dashpva.utils.pva_reader.camonitor_clear"):
            reader._clear_roi_backup_monitor()
        assert reader._active_roi_pvs == []

    def test_no_active_pvs_clears_nothing(self):
        reader = _reader_with_active_pvs([])
        with patch("dashpva.utils.pva_reader.camonitor_clear") as mock_clear:
            reader._clear_roi_backup_monitor()
        mock_clear.assert_not_called()

    def test_exception_in_one_clear_does_not_abort_others(self):
        reader = _reader_with_active_pvs([
            "13SIM1:ROI2:MinX",
            "13SIM1:ROI2:MinY",
            "13SIM1:ROI2:SizeX",
            "13SIM1:ROI2:SizeY",
        ])
        with patch(
            "dashpva.utils.pva_reader.camonitor_clear",
            side_effect=Exception("boom"),
        ) as mock_clear:
            reader._clear_roi_backup_monitor()
        assert mock_clear.call_count == 4
