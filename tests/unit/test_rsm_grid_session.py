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

"""Live grid session: state rules, config locking, and confined saves."""

import threading

import numpy as np
import pytest

from dashpva.utils.rsm_grid_session import (
    GridSessionState,
    LiveGridSession,
    SessionError,
    geometry_fingerprint,
)
from dashpva.utils.rsm_gridder import RSMMergeError
from dashpva.utils.rsm_live_grid import GridBoundsSpec

BOUNDS = GridBoundsSpec(0.0, 1.0, 0.0, 2.0, 0.0, 3.0, nx=5, ny=6, nz=7)
FINGERPRINT = geometry_fingerprint("sample", ["z-"], 10000.0)


def _frame(value=1.0, n=8):
    t = np.linspace(0.05, 0.95, n)
    return t * 1.0, t * 2.0, t * 3.0, np.full(n, value)


def _session(tmp_path):
    return LiveGridSession(output_dir=str(tmp_path / "out"))


class TestStateMachine:
    def test_starts_idle(self, tmp_path):
        assert _session(tmp_path).state is GridSessionState.IDLE

    def test_start_stop_clear_cycle(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        assert session.state is GridSessionState.RUNNING
        session.stop()
        assert session.state is GridSessionState.STOPPED
        session.clear()
        assert session.state is GridSessionState.IDLE
        assert session.accumulator is None

    def test_stop_from_idle_is_refused(self, tmp_path):
        with pytest.raises(SessionError, match="Nothing is running"):
            _session(tmp_path).stop()

    def test_start_while_running_is_refused(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        with pytest.raises(SessionError, match="already running"):
            session.start(BOUNDS, fingerprint=FINGERPRINT)

    def test_frames_outside_running_are_ignored_not_errors(self, tmp_path):
        """Frames keep arriving after Stop; that is normal, not a fault."""
        session = _session(tmp_path)
        assert session.add_frame(*_frame()) == 0
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        assert session.add_frame(*_frame()) > 0
        session.stop()
        binned_after_stop = session.add_frame(*_frame())
        assert binned_after_stop == 0


class TestConfigLocking:
    def test_first_frame_latches_geometry_and_a_change_stops_fail_closed(
        self, tmp_path
    ):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint="")
        session.validate_geometry("geometry-a")
        assert session.get_state()["geometry_fingerprint"] == "geometry-a"
        with pytest.raises(SessionError, match="geometry changed"):
            session.validate_geometry("geometry-b")
        assert session.state is GridSessionState.STOPPED
        assert session.incomplete

    def test_resume_with_identical_config_continues_the_run(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(*_frame())
        before = session.accumulator.coverage.copy()
        session.stop()

        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(*_frame())

        # Resumed into the same accumulator rather than starting a new one.
        np.testing.assert_array_equal(session.accumulator.coverage, before * 2)

    @pytest.mark.parametrize("change", ["fingerprint", "bounds", "monitor", "mask"])
    def test_resume_after_a_config_change_is_refused(self, tmp_path, change):
        """Mixing incompatible geometry would make the volume uninterpretable."""
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT, monitor_name="I0",
                      mask_signature="sig-a")
        session.add_frame(*_frame(), monitor=1.0)
        session.stop()

        kwargs = dict(fingerprint=FINGERPRINT, monitor_name="I0", mask_signature="sig-a")
        bounds = BOUNDS
        if change == "fingerprint":
            kwargs["fingerprint"] = geometry_fingerprint("different")
        elif change == "bounds":
            bounds = GridBoundsSpec(0.0, 1.0, 0.0, 2.0, 0.0, 3.0, nx=9, ny=6, nz=7)
        elif change == "monitor":
            kwargs["monitor_name"] = "I1"
        else:
            kwargs["mask_signature"] = "sig-b"

        with pytest.raises(SessionError, match="Clear the grid"):
            session.start(bounds, **kwargs)

    def test_clear_then_start_accepts_a_new_configuration(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.stop()
        session.clear()
        wider = GridBoundsSpec(0.0, 2.0, 0.0, 2.0, 0.0, 3.0, nx=9, ny=6, nz=7)
        session.start(wider, fingerprint=geometry_fingerprint("new"))
        assert session.state is GridSessionState.RUNNING


class TestIncompleteness:
    def test_running_frame_accounting_is_disjoint_and_resets_on_clear(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.note_frame_seen()
        session.add_frame(*_frame())
        session.note_frame_seen()
        session.note_binding_rejection()
        session.note_frame_seen()
        session.note_processing_rejection()
        state = session.get_state()
        assert state["frames_seen_running"] == 3
        assert state["frames_accepted"] == 1
        assert state["frames_rejected_binding"] == 1
        assert state["frames_rejected_processing"] == 1
        assert state["accounting_consistent"]

        session.clear()
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        assert session.get_state()["frames_seen_running"] == 0

    def test_out_of_range_points_mark_the_preview_incomplete(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        assert not session.incomplete
        session.add_frame(np.array([99.0]), np.array([1.0]),
                          np.array([1.0]), np.array([1.0]))
        assert session.incomplete

    def test_a_frame_id_gap_marks_the_preview_incomplete(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(*_frame(), followed_gap=True)
        assert session.incomplete

    def test_a_bad_frame_is_absorbed_and_flagged_not_raised(self, tmp_path):
        """One malformed frame must not tear down a long accumulation."""
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        assert session.add_frame(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(5)) == 0
        assert session.incomplete
        assert "disagree in length" in session.get_state()["last_error"]
        # ...and the session is still usable.
        assert session.add_frame(*_frame()) > 0

    def test_rejected_upstream_frames_are_recorded(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.note_rejected_frame()
        assert session.incomplete
        assert session.get_state()["frames_rejected"] == 1


class TestSaving:
    def _stopped(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(*_frame(value=4.0))
        session.stop()
        return session

    def test_save_from_stopped_writes_volume_and_coverage(self, tmp_path):
        import h5py

        session = self._stopped(tmp_path)
        result = session.save("live_volume.h5")
        assert result["job_id"].startswith("save-")
        with h5py.File(result["saved_path"], "r") as handle:
            assert "entry/data/data" in handle
            assert "entry/data/coverage" in handle
            meta = handle["entry/data/metadata"]
            assert meta["aggregation"][()].decode() == "unweighted_mean"
        # Saving must leave the session usable, not consumed.
        assert session.state is GridSessionState.STOPPED

    def test_save_while_running_is_refused(self, tmp_path):
        """A mid-flight write would capture a torn accumulator."""
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(*_frame())
        with pytest.raises(SessionError, match="only allowed from 'stopped'"):
            session.save("x.h5")

    def test_save_with_nothing_accumulated_is_refused(self, tmp_path):
        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.stop()
        session.clear()
        with pytest.raises(SessionError):
            session.save("x.h5")

    @pytest.mark.parametrize("attempt", [
        "../escape.h5",
        "/etc/passwd",
        "../../elsewhere/vol.h5",
    ])
    def test_paths_are_confined_to_the_output_directory(self, tmp_path, attempt):
        """The consumer is reachable over the network; no write-anywhere."""
        import os

        session = self._stopped(tmp_path)
        result = session.save(attempt)
        saved = os.path.realpath(result["saved_path"])
        assert saved.startswith(os.path.realpath(session.output_dir))

    def test_overwriting_an_existing_file_is_refused(self, tmp_path):
        session = self._stopped(tmp_path)
        session.save("once.h5")
        with pytest.raises(SessionError, match="Refusing to overwrite"):
            session.save("once.h5")

    def test_extension_is_added_when_missing(self, tmp_path):
        session = self._stopped(tmp_path)
        assert session.save("noext")["saved_path"].endswith(".h5")

    def test_incompleteness_is_recorded_in_the_saved_metadata(self, tmp_path):
        import h5py

        session = _session(tmp_path)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(np.array([99.0]), np.array([1.0]),
                          np.array([1.0]), np.array([1.0]))
        session.stop()
        path = session.save("partial.h5")["saved_path"]
        with h5py.File(path, "r") as handle:
            assert bool(handle["entry/data/metadata/live_incomplete"][()])

    def test_a_failed_save_returns_to_stopped_rather_than_sticking_in_saving(self, tmp_path):
        def _boom(*args, **kwargs):
            raise OSError("disk full")

        session = LiveGridSession(output_dir=str(tmp_path / "out"), save_fn=_boom)
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(*_frame())
        session.stop()
        with pytest.raises(OSError):
            session.save("fail.h5")
        assert session.state is GridSessionState.STOPPED
        assert "disk full" in session.get_state()["last_error"]

    def test_save_releases_the_lock_but_blocks_other_transitions(self, tmp_path):
        started = threading.Event()
        release = threading.Event()

        def _slow_save(*args, **kwargs):
            started.set()
            assert release.wait(2)

        session = LiveGridSession(
            output_dir=str(tmp_path / "out"), save_fn=_slow_save
        )
        session.start(BOUNDS, fingerprint=FINGERPRINT)
        session.add_frame(*_frame())
        session.stop()
        thread = threading.Thread(target=session.save, args=("slow.h5",))
        thread.start()
        assert started.wait(1)
        assert session.state is GridSessionState.SAVING
        with pytest.raises(SessionError, match="save is in progress"):
            session.clear()
        with pytest.raises(SessionError, match="save is in progress"):
            session.start(BOUNDS, fingerprint=FINGERPRINT)
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert session.state is GridSessionState.STOPPED


def test_memory_guard_runs_before_grid_allocation(tmp_path):
    def _reject(_estimate):
        raise RSMMergeError("grid exceeds configured memory policy")

    session = LiveGridSession(
        output_dir=str(tmp_path / "out"), memory_guard=_reject
    )
    with pytest.raises(SessionError, match="configured memory policy"):
        session.start(BOUNDS, fingerprint=FINGERPRINT)
    assert session.accumulator is None
    assert session.state is GridSessionState.IDLE


class TestFingerprint:
    def test_same_inputs_give_the_same_fingerprint(self):
        assert geometry_fingerprint("a", [1, 2]) == geometry_fingerprint("a", [1, 2])

    def test_different_inputs_differ(self):
        assert geometry_fingerprint("a", [1, 2]) != geometry_fingerprint("a", [1, 3])
