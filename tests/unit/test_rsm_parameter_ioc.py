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

"""Pure/simulator-safe contracts for the profile-driven RSM IOC."""

from __future__ import annotations

import copy

import pytest

from dashpva.consumers.ioc_rsm_parameter import (
    _has_pending_rsm_change,
    _reorder_loaded_axis_rows,
    _restore_axis_row,
    _review_pending_change,
    all_pv_names,
    build_ioc_database,
    static_ioc_values,
)
from dashpva.utils.config.source import ConfigSaveResult, ConfigSaveStatus
from dashpva.utils.rsm_parameter_config import (
    RSMParameterEditSession,
    SnapshotActivationError,
    adoption_diff,
    apply_and_activate,
    default_parameter_mapping,
    profile_from_raw,
    requires_adoption_confirmation,
    update_raw_profile,
    validate_parameter_profile,
)


def _raw(parameters=None):
    return {
        "IOC_PREFIX": "sim",
        "IOC_RSM_PARAMETER": parameters or default_parameter_mapping(),
        "HKL": {"CUSTOM": {"KEEP": "yes"}},
    }


def _axis(role: str, index: int) -> dict[str, str]:
    return {
        "LABEL": f"{role} label {index}",
        "RECORD_NAME": f"{role}Axis{index}",
        "SOURCE_PV": str(index),
        "DIRECTION": "z-" if role == "Sample" else "x+",
        "ANGLE_UNITS": "deg",
    }


@pytest.mark.parametrize(
    ("sample_count", "detector_count", "orientation"),
    [(0, 0, "x+"), (3, 0, "sam"), (0, 8, "det"), (10, 10, "det")],
)
def test_ioc_database_supports_arbitrary_circle_counts(
    sample_count, detector_count, orientation
):
    parameters = default_parameter_mapping()
    parameters["SAMPLE_AXES"] = [
        _axis("Sample", index) for index in range(1, sample_count + 1)
    ]
    parameters["DETECTOR_AXES"] = [
        _axis("Detector", index) for index in range(1, detector_count + 1)
    ]
    parameters["SAMPLE_ORIENTATION"] = orientation
    profile = validate_parameter_profile("sim", parameters)

    database = build_ioc_database(profile)
    records = all_pv_names(profile)

    assert database.count(":Position\")") == sample_count + detector_count
    axis_record_count = sum(
        name.startswith("sim:SampleAxis") or name.startswith("sim:DetectorAxis")
        for name, _ in records
    )
    assert axis_record_count == 4 * (sample_count + detector_count)
    assert sum(name.endswith(":Position") for name, _ in records) == (
        sample_count + detector_count
    )
    if sample_count == 10:
        assert 'record(ai, "sim:SampleAxis10:Position")' in database
        assert static_ioc_values(profile)["sim:SampleAxis10:AxisNumber"] == 10.0


def test_label_can_change_without_changing_stable_record_identity():
    parameters = default_parameter_mapping()
    parameters["SAMPLE_AXES"] = [_axis("Sample", 1)]
    parameters["DETECTOR_AXES"] = [_axis("Detector", 1)]
    first = validate_parameter_profile("sim", parameters)
    parameters["SAMPLE_AXES"][0]["LABEL"] = "human readable rename"
    renamed = validate_parameter_profile("sim", parameters)

    assert first.sample_axes[0].record_name == renamed.sample_axes[0].record_name
    values = static_ioc_values(renamed)
    assert values["sim:SampleAxis1:SpecMotorName"] == "human readable rename"


def test_optional_spec_motor_name_is_distinct_from_display_label():
    parameters = default_parameter_mapping()
    parameters["SAMPLE_AXES"] = [_axis("Sample", 1)]
    parameters["DETECTOR_AXES"] = [_axis("Detector", 1)]
    parameters["SAMPLE_AXES"][0]["LABEL"] = "friendly display label"
    parameters["SAMPLE_AXES"][0]["SPEC_MOTOR_NAME"] = "th"

    profile = validate_parameter_profile("sim", parameters)
    values = static_ioc_values(profile)

    assert profile.sample_axes[0].label == "friendly display label"
    assert profile.sample_axes[0].spec_motor_name == "th"
    assert values["sim:SampleAxis1:SpecMotorName"] == "th"


def test_removed_axis_review_restore_keeps_loaded_identity_and_order():
    rows = [
        (0, {"LABEL": "Mu"}),
        (-1, {"LABEL": "New"}),
        (2, {"LABEL": "Chi"}),
    ]

    restored = _restore_axis_row(rows, 1, {"LABEL": "Eta"})

    assert [origin for origin, _axis in restored] == [0, -1, 1, 2]
    assert restored[2][1] == {"LABEL": "Eta"}
    assert rows == [
        (0, {"LABEL": "Mu"}),
        (-1, {"LABEL": "New"}),
        (2, {"LABEL": "Chi"}),
    ]


def test_review_reorder_keeps_new_axis_in_its_existing_slot():
    rows = [
        (1, {"LABEL": "Eta"}),
        (-1, {"LABEL": "New"}),
        (0, {"LABEL": "Mu"}),
    ]

    reordered = _reorder_loaded_axis_rows(rows, [0, 1])

    assert [origin for origin, _axis in reordered] == [0, -1, 1]
    assert reordered[1][1] == {"LABEL": "New"}


def test_invalid_rsm_form_stays_dirty_and_visible_as_a_review_row():
    def invalid_pending():
        raise ValueError("ENERGY_SOURCE_PV is required")

    pending, rows = _review_pending_change(invalid_pending)

    assert pending is None
    assert _has_pending_rsm_change(invalid_pending, {})
    assert rows == [
        (
            "change",
            "RSM_CONFIGURATION_INVALID",
            "valid loaded configuration",
            "ENERGY_SOURCE_PV is required",
        )
    ]


def test_record_name_is_unique_across_roles():
    parameters = default_parameter_mapping()
    parameters["DETECTOR_AXES"][0]["RECORD_NAME"] = parameters["SAMPLE_AXES"][0][
        "RECORD_NAME"
    ]

    with pytest.raises(ValueError, match="unique across"):
        validate_parameter_profile("sim", parameters)


@pytest.mark.parametrize("source", ["nan", "inf", "-inf", "+INF"])
def test_nonfinite_static_sources_are_rejected(source):
    parameters = default_parameter_mapping()
    parameters["SAMPLE_AXES"][0]["SOURCE_PV"] = source

    with pytest.raises(ValueError, match="finite"):
        validate_parameter_profile("sim", parameters)


def test_explicit_orientation_parallel_to_primary_beam_is_rejected():
    parameters = default_parameter_mapping()
    parameters["SAMPLE_ORIENTATION"] = "y+"

    with pytest.raises(ValueError, match="parallel"):
        validate_parameter_profile("sim", parameters)


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_parameter_schema_version_is_explicit_and_strict(version):
    parameters = default_parameter_mapping()
    parameters["SCHEMA_VERSION"] = version

    with pytest.raises(ValueError, match="SCHEMA_VERSION"):
        validate_parameter_profile("sim", parameters)


def test_update_preserves_unknown_raw_hkl_and_profile_extensions():
    raw = _raw()
    raw["VENDOR"] = {"keep": [1, 2, 3]}
    replacement = update_raw_profile(raw, "new-prefix", default_parameter_mapping())

    assert replacement["IOC_PREFIX"] == "new-prefix:"
    assert replacement["HKL"] == raw["HKL"]
    assert replacement["VENDOR"] == raw["VENDOR"]
    assert raw["IOC_PREFIX"] == "sim"


class _FakeSource:
    def __init__(self, raw):
        self.raw = copy.deepcopy(raw)
        self.revision = "revision-1"
        self.force_conflict = False

    def load_snapshot(self):
        return copy.deepcopy(self.raw), self.revision

    def replace_if_revision(self, full_config, revision):
        if self.force_conflict or revision != self.revision:
            return ConfigSaveResult(ConfigSaveStatus.CONFLICT)
        self.raw = copy.deepcopy(full_config)
        self.revision = "revision-2"
        return ConfigSaveResult(ConfigSaveStatus.SAVED, revision=self.revision)


def test_staged_session_applies_only_after_successful_cas():
    source = _FakeSource(_raw())
    session = RSMParameterEditSession(source)
    parameters = default_parameter_mapping()
    parameters["SAMPLE_AXES"][0]["LABEL"] = "saved label"

    result, saved = session.apply("sim", parameters)

    assert result.saved
    assert saved["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]["LABEL"] == "saved label"
    assert session.revision == "revision-2"


def test_staged_session_conflict_never_applies_candidate():
    source = _FakeSource(_raw())
    session = RSMParameterEditSession(source)
    before = copy.deepcopy(source.raw)
    source.force_conflict = True
    parameters = default_parameter_mapping()
    parameters["SAMPLE_AXES"][0]["LABEL"] = "must not save"

    result, saved = session.apply("sim", parameters)

    assert result.status is ConfigSaveStatus.CONFLICT
    assert saved is None
    assert source.raw == before
    assert session.raw == before


def test_activation_is_skipped_on_cas_conflict():
    source = _FakeSource(_raw())
    source.force_conflict = True
    session = RSMParameterEditSession(source)
    activated = []

    result, saved = apply_and_activate(
        session,
        "sim",
        default_parameter_mapping(),
        activated.append,
    )

    assert result.status is ConfigSaveStatus.CONFLICT
    assert saved is None
    assert activated == []


def test_activator_receives_exact_cas_saved_snapshot():
    source = _FakeSource(_raw())
    session = RSMParameterEditSession(source)
    activated = []

    result, saved = apply_and_activate(
        session,
        "beamline",
        default_parameter_mapping(),
        activated.append,
    )

    assert result.saved
    assert activated == [saved]
    assert activated[0] == source.raw


def test_activation_failure_retains_exact_saved_snapshot_for_retry():
    source = _FakeSource(_raw())
    session = RSMParameterEditSession(source)

    def fail(_snapshot):
        raise RuntimeError("simulator unavailable")

    with pytest.raises(SnapshotActivationError) as caught:
        apply_and_activate(session, "sim", default_parameter_mapping(), fail)

    assert caught.value.snapshot == source.raw
    assert session.raw == source.raw



def test_first_adoption_diff_exposes_channel_and_static_geometry_changes():
    raw = {
        "IOC_PREFIX": "old",
        "HKL": {"SAMPLE_CIRCLE_AXIS_1": {"POSITION": "hand:motor.RBV"}},
    }

    assert requires_adoption_confirmation(raw)
    details = adoption_diff(raw, "new", default_parameter_mapping())

    assert "hand:motor.RBV" in details
    assert "new:Mu:Position" in details
    assert "UB_MATRIX" in details
    assert "<not profile-backed>" in details


def test_partial_canonical_absent_defaults_do_not_request_noop_adoption():
    raw = {
        "IOC_PREFIX": "sim",
        "IOC_RSM_PARAMETER": {
            "ENERGY_SOURCE_PV": "10",
            "ENERGY_UNITS": "keV",
            "SAMPLE_ORIENTATION": "x+",
        },
    }
    profile = profile_from_raw(raw)

    replacement = update_raw_profile(
        raw,
        profile.prefix,
        profile.parameter_mapping(),
        axis_origins={"SAMPLE_AXES": (), "DETECTOR_AXES": ()},
    )

    assert replacement == raw
    assert not requires_adoption_confirmation(raw)


def test_seed_shape_round_trips_through_runtime_profile():
    profile = profile_from_raw(_raw())

    assert profile.schema_version == 1
    assert len(profile.sample_axes) == 4
    assert len(profile.detector_axes) == 2
    assert profile.detector_setup["DISTANCE"] == pytest.approx(400.644)
