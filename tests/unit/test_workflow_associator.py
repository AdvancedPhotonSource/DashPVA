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

"""Metadata-associator Start validation and staleness warning (see PR #139
restoration plan, Track B).

``Workflow.__init__`` builds the whole dialog (``uic.loadUi``, DB probing),
which none of the methods under test need -- so, like
``test_workflow_config_tree.py``'s ``_Tree`` harness, these bind only the
associator-related methods onto a lightweight stand-in with fake form widgets.
"""

import logging
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt5.QtWidgets")

import dashpva.settings as app_settings  # noqa: E402
import dashpva.workflow.workflow as workflow_module  # noqa: E402
from dashpva.workflow.workflow import Workflow  # noqa: E402


class _FakeWidget:
    """Stands in for whichever of text()/currentText()/value() is called."""

    def __init__(self, value=""):
        self._value = value

    def text(self):
        return self._value

    def currentText(self):
        return self._value

    def value(self):
        return self._value


class _AssociatorHarness:
    _build_metadata_channel_list = Workflow._build_metadata_channel_list
    _build_metadata_channels = Workflow._build_metadata_channels
    _resolved_profile_identity = Workflow._resolved_profile_identity
    run_associator_consumers = Workflow.run_associator_consumers
    stop_associator_consumers = Workflow.stop_associator_consumers
    _sync_associator_metadata = Workflow._sync_associator_metadata
    _on_associator_staleness_timer = Workflow._on_associator_staleness_timer

    def __init__(self):
        self.processes = {}
        self.workers = {}
        self.logger = logging.getLogger("test_workflow_associator")
        self.lineEditInputChannelAssociator = _FakeWidget("pva://input")
        self.lineEditControlChannelAssociator = _FakeWidget("pva://control")
        self.lineEditStatusChannelAssociator = _FakeWidget("pva://status")
        self.lineEditOutputChannelAssociator = _FakeWidget("pva://output")
        self.comboBoxProcessorFileAssociator = _FakeWidget("processor.py")
        self.lineEditProcessorClassAssociator = _FakeWidget("MyProcessor")
        self.spinBoxReportPeriodAssociator = _FakeWidget(5)
        self.spinBoxServerQueueSizeAssociator = _FakeWidget(10)
        self.spinBoxNConsumersAssociator = _FakeWidget(1)
        self.spinBoxDistributorUpdatesAssociator = _FakeWidget(1)
        self.buttonRunAssociatorConsumers = MagicMock()
        self.buttonStopAssociatorConsumers = MagicMock()
        self.labelStatusAssociatorConsumers = MagicMock()
        self.textEditAssociatorConsumersOutput = MagicMock()
        self._save_meta_assoc_last = MagicMock()
        self._format_associator_output = MagicMock()


@pytest.fixture(autouse=True)
def _empty_app_settings(monkeypatch):
    """Every test starts from a known-empty metadata/channel configuration."""
    monkeypatch.setattr(app_settings, "SCAN_FLAG_PV", None, raising=False)
    monkeypatch.setattr(app_settings, "FILE_PATH_PV", None, raising=False)
    monkeypatch.setattr(app_settings, "FILE_NAME_PV", None, raising=False)
    monkeypatch.setattr(app_settings, "METADATA_CA", {}, raising=False)
    monkeypatch.setattr(app_settings, "METADATA_PVA", {}, raising=False)
    monkeypatch.setattr(app_settings, "HKL", {}, raising=False)
    monkeypatch.setattr(app_settings, "LOCATOR", None, raising=False)


@pytest.fixture
def harness():
    return _AssociatorHarness()


class TestBuildMetadataChannelsDedup:
    def test_dedupes_across_scan_metadata_ca_and_hkl_sources(self, harness, monkeypatch):
        # Same PV surfaces from SCAN_FLAG_PV, METADATA_CA, *and* the HKL-derived
        # channels -- must appear exactly once in the built string.
        monkeypatch.setattr(app_settings, "SCAN_FLAG_PV", "6idb1:scan:flag")
        monkeypatch.setattr(app_settings, "METADATA_CA", {"extra": "6idb1:scan:flag"})
        monkeypatch.setattr(
            workflow_module, "semantic_hkl_channels",
            lambda hkl: ("6idb1:scan:flag", "6idb1:Mu:Position"),
        )

        result = harness._build_metadata_channels()
        channels = [c for c in result.split(",") if c]

        assert channels.count("ca://6idb1:scan:flag") == 1
        assert "ca://6idb1:Mu:Position" in channels

    def test_metadata_ca_values_pass_through_unchanged(self, harness, monkeypatch):
        original = {"beam_current": "6idb1:beam:current"}
        monkeypatch.setattr(app_settings, "METADATA_CA", dict(original))

        harness._build_metadata_channels()

        assert app_settings.METADATA_CA == original

    def test_channel_list_survives_a_comma_inside_a_channel_value(self, harness, monkeypatch):
        # Regression: run_associator_consumers used to recover the bare
        # channel-name set by re-splitting the joined ','-delimited string on
        # ',' then '://'. A free-text METADATA_CA value containing a literal
        # comma (a plausible paste/typo in the Settings tree) split into two
        # fragments, the second of which had no '://' -- raising IndexError
        # before the associator could ever start. _build_metadata_channel_list
        # must derive the bare-name set directly, never by re-parsing the
        # joined string.
        monkeypatch.setattr(app_settings, "METADATA_CA", {"oops": "6idb1:a,6idb1:b"})

        channel_list, bare_names = harness._build_metadata_channel_list()

        assert "6idb1:a,6idb1:b" in bare_names
        assert "ca://6idb1:a,6idb1:b" in channel_list


class TestRunAssociatorConsumersValidation:
    def test_refuses_and_lists_missing_channels(self, harness, monkeypatch):
        monkeypatch.setattr(
            workflow_module, "required_rsm_channels",
            lambda hkl: frozenset({"6idb1:Mu:Position", "6idb1:spec:UB_matrix:Value"}),
        )
        mock_msgbox = MagicMock()
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", mock_msgbox)

        harness.run_associator_consumers()

        assert "associator_consumers" not in harness.processes
        critical_calls = mock_msgbox.critical.call_args_list
        assert len(critical_calls) == 1
        message = critical_calls[0].args[2]
        assert "6idb1:Mu:Position" in message
        assert "6idb1:spec:UB_matrix:Value" in message

    def test_calls_reload_before_building_channels(self, harness, monkeypatch):
        # Reuse the missing-channels setup so the method returns before
        # subprocess.Popen -- reload() is called unconditionally either way.
        monkeypatch.setattr(
            workflow_module, "required_rsm_channels",
            lambda hkl: frozenset({"6idb1:Mu:Position"}),
        )
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", MagicMock())
        mock_reload = MagicMock()
        monkeypatch.setattr(app_settings, "reload", mock_reload)

        harness.run_associator_consumers()

        mock_reload.assert_called_once()


class TestRunAssociatorConsumersSuccess:
    """The success path was previously untested -- only the two refusal
    branches were. A regression in the required-vs-built set-difference logic
    (an inverted operator, an off-by-one in the scheme-strip, HKL-shape drift
    between required_rsm_channels and semantic_hkl_channels) would otherwise
    silently block every valid, correctly-configured profile at every
    beamline with no test failing."""

    def _fake_process(self):
        process = MagicMock()
        process.pid = 4242
        process.stdout.readline.return_value = ""
        process.poll.return_value = 0
        return process

    def test_starts_and_records_baseline_when_required_channels_are_covered(
        self, harness, monkeypatch
    ):
        monkeypatch.setattr(
            workflow_module, "semantic_hkl_channels", lambda hkl: ("6idb1:Mu:Position",),
        )
        monkeypatch.setattr(
            workflow_module, "required_rsm_channels",
            lambda hkl: frozenset({"6idb1:Mu:Position"}),
        )
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", MagicMock())
        fake_process = self._fake_process()
        mock_popen = MagicMock(return_value=fake_process)
        monkeypatch.setattr(workflow_module.subprocess, "Popen", mock_popen)

        harness.run_associator_consumers()

        mock_popen.assert_called_once()
        assert harness.processes["associator_consumers"] is fake_process
        assert "associator_consumers" in harness.workers
        assert "ca://6idb1:Mu:Position" in harness._associator_metadata_channels
        assert harness._associator_profile_identity == harness._resolved_profile_identity()

    def test_does_not_start_when_a_required_channel_is_missing(self, harness, monkeypatch):
        monkeypatch.setattr(
            workflow_module, "semantic_hkl_channels", lambda hkl: ("6idb1:Mu:Position",),
        )
        monkeypatch.setattr(
            workflow_module, "required_rsm_channels",
            lambda hkl: frozenset({"6idb1:Mu:Position", "6idb1:spec:UB_matrix:Value"}),
        )
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", MagicMock())
        mock_popen = MagicMock()
        monkeypatch.setattr(workflow_module.subprocess, "Popen", mock_popen)

        harness.run_associator_consumers()

        mock_popen.assert_not_called()
        assert "associator_consumers" not in harness.processes


class TestAssociatorStalenessWarning:
    def _mark_running_and_drifted(self, harness, monkeypatch):
        harness.processes["associator_consumers"] = object()
        harness._associator_metadata_channels = "ca://old:channel"
        harness._associator_profile_identity = None
        monkeypatch.setattr(
            workflow_module, "semantic_hkl_channels", lambda hkl: ("new:channel",),
        )

    def test_does_not_stop_or_restart_running_associator_on_drift(self, harness, monkeypatch):
        self._mark_running_and_drifted(harness, monkeypatch)
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", MagicMock())
        harness.stop_associator_consumers = MagicMock()
        harness.run_associator_consumers = MagicMock()

        harness._sync_associator_metadata()

        harness.stop_associator_consumers.assert_not_called()
        harness.run_associator_consumers.assert_not_called()
        assert "associator_consumers" in harness.processes

    def test_warning_fires_once_per_associator_run_not_once_per_drift_tick(self, harness, monkeypatch):
        self._mark_running_and_drifted(harness, monkeypatch)
        mock_msgbox = MagicMock()
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", mock_msgbox)

        harness._sync_associator_metadata()
        harness._sync_associator_metadata()  # still drifted, user hasn't acted

        assert mock_msgbox.call_count == 1

    def test_a_later_different_drift_still_warns_after_an_earlier_one(self, harness, monkeypatch):
        # Regression: a one-shot-per-run flag would permanently suppress every
        # notice after the first, including a genuinely new, unrelated drift
        # later in the same run (e.g. a second profile switch, or the first
        # drift being a transient blip that then changes again). The baseline
        # must advance to what was just observed, not stay pinned to run start.
        self._mark_running_and_drifted(harness, monkeypatch)
        mock_msgbox = MagicMock()
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", mock_msgbox)

        harness._sync_associator_metadata()
        assert mock_msgbox.call_count == 1

        monkeypatch.setattr(
            workflow_module, "semantic_hkl_channels", lambda hkl: ("yet:another:channel",),
        )
        harness._sync_associator_metadata()

        assert mock_msgbox.call_count == 2

    def test_identity_only_drift_warns_even_when_channels_are_unchanged(self, harness, monkeypatch):
        # Isolates the identity_changed half of the guard: a profile switched
        # elsewhere with no channel-set change must still be caught.
        harness.processes["associator_consumers"] = object()
        current_channels = harness._build_metadata_channels()
        harness._associator_metadata_channels = current_channels
        harness._associator_profile_identity = "profile-a"
        monkeypatch.setattr(harness, "_resolved_profile_identity", lambda: "profile-b")
        mock_msgbox = MagicMock()
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", mock_msgbox)

        harness._sync_associator_metadata()

        mock_msgbox.assert_called_once()
        assert harness._associator_metadata_channels == current_channels

    def test_unchanged_configuration_shows_no_warning(self, harness, monkeypatch):
        harness.processes["associator_consumers"] = object()
        harness._associator_metadata_channels = harness._build_metadata_channels()
        harness._associator_profile_identity = harness._resolved_profile_identity()
        mock_msgbox = MagicMock()
        monkeypatch.setattr(workflow_module.QtWidgets, "QMessageBox", mock_msgbox)

        harness._sync_associator_metadata()

        mock_msgbox.assert_not_called()

    def test_staleness_timer_reloads_and_rechecks_only_while_running(self, harness, monkeypatch):
        mock_reload = MagicMock()
        monkeypatch.setattr(app_settings, "reload", mock_reload)
        harness._sync_associator_metadata = MagicMock()

        harness._on_associator_staleness_timer()  # not running -- must no-op
        mock_reload.assert_not_called()
        harness._sync_associator_metadata.assert_not_called()

        harness.processes["associator_consumers"] = object()
        harness._on_associator_staleness_timer()
        mock_reload.assert_called_once()
        harness._sync_associator_metadata.assert_called_once()


class TestStopAssociatorConsumers:
    def test_stopping_closes_a_lingering_stale_notice_box(self, harness):
        harness.processes["associator_consumers"] = MagicMock()
        harness.workers["associator_consumers"] = (MagicMock(), MagicMock())
        box = MagicMock()
        harness._associator_stale_notice_box = box

        harness.stop_associator_consumers()

        box.close.assert_called_once()
        assert harness._associator_stale_notice_box is None

    def test_stopping_without_a_notice_box_does_not_raise(self, harness):
        harness.processes["associator_consumers"] = MagicMock()
        harness.workers["associator_consumers"] = (MagicMock(), MagicMock())

        harness.stop_associator_consumers()  # must not raise
