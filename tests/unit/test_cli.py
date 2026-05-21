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

    def test_run_help(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "launcher" in result.output.lower() or "DashPVA" in result.output

    def test_run_invokes_subprocess(self, runner):
        with patch("dashpva.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = runner.invoke(cli, ["run"])
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "dashpva.viewer.launcher.launcher" in cmd[-1]

    def test_detector_invokes_subprocess(self, runner):
        with patch("dashpva.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = runner.invoke(cli, ["detector"])
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "dashpva.viewer.area_det.area_det_viewer" in cmd[-1]

    def test_monitor_scan(self, runner):
        with patch("dashpva.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = runner.invoke(cli, ["monitor", "scan"])
            mock_run.assert_called_once()

    def test_monitor_invalid_name(self, runner):
        result = runner.invoke(cli, ["monitor", "invalid_view"])
        assert result.exit_code != 0
