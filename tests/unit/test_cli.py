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

"""Tests for dashpva.cli — Click CLI commands."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from dashpva.cli import cli


@pytest.fixture()
def runner():
    return CliRunner()


class TestCLI:

    def test_help_output(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DashPVA" in result.output

    def test_help_points_to_subcommand_help(self, runner):
        # The main help must direct users to `DashPVA COMMAND --help` (the working
        # path) rather than the old, broken `-s`/`-d`/... help flags.
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "sim --help" in result.output

    def test_old_global_help_flags_removed(self, runner):
        # The confusing broken flags (e.g. -s) are gone; use `DashPVA sim --help`.
        result = runner.invoke(cli, ["-s"])
        assert result.exit_code != 0

    def test_run_help(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "launcher" in result.output.lower() or "DashPVA" in result.output

    def test_run_invokes_subprocess(self, runner):
        with patch("dashpva.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = runner.invoke(cli, ["run"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "dashpva.viewer.launcher.launcher" in cmd[-1]

    def test_detector_invokes_subprocess(self, runner):
        with patch("dashpva.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = runner.invoke(cli, ["detector"])
            assert result.exit_code == 0
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "dashpva.viewer.area_det.area_det_viewer" in cmd[-1]

    def test_monitor_scan(self, runner):
        with patch("dashpva.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = runner.invoke(cli, ["monitor", "scan"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_monitor_invalid_name(self, runner):
        result = runner.invoke(cli, ["monitor", "invalid_view"])
        assert result.exit_code != 0
