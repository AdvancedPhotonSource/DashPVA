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
    ActiveProfileChanged,
    ProfileContentMismatch,
    _classify_activation_mismatch,
    _describe_config_differences,
    _format_detector_value,
    _format_distance_or_pv,
    _format_ub_or_pv,
    _has_pending_rsm_change,
    _parse_detector_value,
    _parse_distance_or_pv,
    _parse_ub_or_pv,
    _pretty_config_path,
    _profile_display_name,
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
    _adoptable_records,
    adoption_diff,
    apply_and_activate,
    default_parameter_mapping,
    profile_from_raw,
    requires_adoption_confirmation,
    update_raw_profile,
    validate_distance,
    validate_parameter_profile,
    validate_ub_matrix,
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


# --- UB matrix / distance: literal vs PV, validation, adoption exclusion ----


def test_ub_matrix_source_pv_defaults_empty_and_round_trips():
    profile = profile_from_raw(_raw())
    assert profile.ub_matrix_source_pv == ""
    assert profile.parameter_mapping()["UB_MATRIX_SOURCE_PV"] == ""

    parameters = default_parameter_mapping()
    parameters["UB_MATRIX_SOURCE_PV"] = "28idbSoft:spec:UB"
    profile = validate_parameter_profile("sim", parameters)

    assert profile.ub_matrix_source_pv == "28idbSoft:spec:UB"
    assert profile.parameter_mapping()["UB_MATRIX_SOURCE_PV"] == "28idbSoft:spec:UB"
    # Setting a live source must not disturb the static UB_MATRIX fallback.
    assert profile.ub_matrix == tuple(default_parameter_mapping()["UB_MATRIX"])


@pytest.mark.parametrize("source", ["nan", "inf", "-inf", "+INF", "500"])
def test_numeric_ub_matrix_source_pv_is_rejected(source):
    # UB_MATRIX_SOURCE_PV is PV-name-only -- UB_MATRIX already has a dedicated
    # literal-fallback key, so a numeric string here (finite or not) would
    # pass validation but the IOC's runtime fallback helper would try to
    # connect to a PV literally named e.g. "500" and never succeed.
    parameters = default_parameter_mapping()
    parameters["UB_MATRIX_SOURCE_PV"] = source

    with pytest.raises(ValueError, match="PV name, not a static number"):
        validate_parameter_profile("sim", parameters)


def test_whitespace_only_ub_matrix_source_pv_is_rejected_not_treated_as_unset():
    parameters = default_parameter_mapping()
    parameters["UB_MATRIX_SOURCE_PV"] = "   "

    with pytest.raises(ValueError, match="UB_MATRIX_SOURCE_PV must not be blank"):
        validate_parameter_profile("sim", parameters)


def test_distance_source_pv_is_optional_and_round_trips():
    parameters = default_parameter_mapping()
    profile = validate_parameter_profile("sim", parameters)
    assert "DISTANCE_SOURCE_PV" not in profile.detector_setup

    parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] = "6idb1:distance.RBV"
    profile = validate_parameter_profile("sim", parameters)

    assert profile.detector_setup["DISTANCE_SOURCE_PV"] == "6idb1:distance.RBV"
    # The static fallback stays intact alongside the source.
    assert profile.detector_setup["DISTANCE"] == pytest.approx(
        default_parameter_mapping()["DETECTOR_SETUP"]["DISTANCE"]
    )


def test_blank_distance_source_pv_is_rejected_not_treated_as_unset():
    parameters = default_parameter_mapping()
    parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] = "   "

    with pytest.raises(ValueError, match="DISTANCE_SOURCE_PV"):
        validate_parameter_profile("sim", parameters)


@pytest.mark.parametrize("source", ["nan", "inf", "-inf", "500"])
def test_numeric_distance_source_pv_is_rejected(source):
    # Same reasoning as UB_MATRIX_SOURCE_PV: DISTANCE already has a dedicated
    # literal-fallback key, so a numeric string here can never work at runtime.
    parameters = default_parameter_mapping()
    parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] = source

    with pytest.raises(ValueError, match="PV name, not a static number"):
        validate_parameter_profile("sim", parameters)


def test_validate_ub_matrix_rejects_singular_matrix():
    with pytest.raises(ValueError, match="full rank"):
        validate_ub_matrix([1, 0, 0, 0, 1, 0, 0, 0, 0])


def test_validate_ub_matrix_rejects_wrong_length():
    with pytest.raises(ValueError, match="9 finite numbers"):
        validate_ub_matrix([1, 2, 3])


@pytest.mark.parametrize("distance", [0, -1.0, float("nan"), float("inf")])
def test_validate_distance_rejects_non_positive_or_non_finite(distance):
    with pytest.raises(ValueError, match="finite and positive"):
        validate_distance(distance)


def test_validate_distance_accepts_positive_finite():
    assert validate_distance(400.644) == pytest.approx(400.644)


def test_parse_ub_or_pv_accepts_valid_literal():
    matrix, source = _parse_ub_or_pv("[1,0,0,0,1,0,0,0,1]", (0, 0, 0, 0, 0, 0, 0, 0, 0))
    assert matrix == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert source == ""


def test_parse_ub_or_pv_treats_non_json_text_as_a_pv_name():
    fallback = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    matrix, source = _parse_ub_or_pv("28idbSoft:spec:UB", fallback)
    assert source == "28idbSoft:spec:UB"
    assert matrix == fallback


def test_parse_ub_or_pv_rejects_malformed_json_instead_of_treating_it_as_a_pv_name():
    fallback = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # JSON-shaped but wrong length / singular / non-finite: must error, never
    # silently fall through to "this must be a PV name".
    with pytest.raises(ValueError, match="9 finite numbers"):
        _parse_ub_or_pv("[1,2,3]", fallback)
    with pytest.raises(ValueError, match="full rank"):
        _parse_ub_or_pv("[1,0,0,0,1,0,0,0,0]", fallback)
    with pytest.raises(ValueError):
        _parse_ub_or_pv("[NaN,0,0,0,1,0,0,0,1]", fallback)


def test_parse_ub_or_pv_rejects_blank():
    with pytest.raises(ValueError, match="blank"):
        _parse_ub_or_pv("   ", (1.0,) * 9)


def test_format_ub_or_pv_shows_pv_when_configured_else_literal():
    ub = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert _format_ub_or_pv(ub, "") == "[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]"
    assert _format_ub_or_pv(ub, "some:pv") == "some:pv"


def test_parse_distance_or_pv_accepts_valid_literal():
    distance, source = _parse_distance_or_pv("400.644", 1.0)
    assert distance == pytest.approx(400.644)
    assert source == ""


def test_parse_distance_or_pv_treats_non_numeric_text_as_a_pv_name():
    distance, source = _parse_distance_or_pv("6idb1:distance.RBV", 400.644)
    assert source == "6idb1:distance.RBV"
    assert distance == pytest.approx(400.644)


def test_parse_distance_or_pv_rejects_non_positive_and_non_finite_numbers():
    for text in ("0", "-1", "nan", "inf"):
        with pytest.raises(ValueError, match="finite and positive"):
            _parse_distance_or_pv(text, 1.0)


def test_parse_distance_or_pv_rejects_blank():
    with pytest.raises(ValueError, match="blank"):
        _parse_distance_or_pv("", 1.0)


def test_format_distance_or_pv_shows_pv_when_configured_else_literal():
    assert _format_distance_or_pv(400.644, "") == "400.644"
    assert _format_distance_or_pv(400.644, "some:pv") == "some:pv"


def test_adoptable_records_excludes_ub_matrix_when_source_pv_configured():
    parameters = default_parameter_mapping()
    parameters["UB_MATRIX_SOURCE_PV"] = "28idbSoft:spec:UB"
    profile = validate_parameter_profile("sim:", parameters)

    records = _adoptable_records(profile)

    assert "sim:spec:UB_matrix:Value" not in records


def test_adoptable_records_includes_ub_matrix_when_no_source_configured():
    profile = validate_parameter_profile("sim:", default_parameter_mapping())

    records = _adoptable_records(profile)

    assert records["sim:spec:UB_matrix:Value"] == ("UB_MATRIX",)


def test_adoptable_records_excludes_distance_when_source_pv_configured():
    parameters = default_parameter_mapping()
    parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] = "6idb1:distance.RBV"
    profile = validate_parameter_profile("sim:", parameters)

    records = _adoptable_records(profile)

    assert "sim:DetectorSetup:Distance" not in records


def test_adoptable_records_includes_distance_when_no_source_configured():
    profile = validate_parameter_profile("sim:", default_parameter_mapping())

    records = _adoptable_records(profile)

    assert records["sim:DetectorSetup:Distance"] == ("DETECTOR_SETUP", "DISTANCE")


def test_update_raw_profile_round_trips_ub_matrix_source_pv():
    raw = _raw()
    parameters = default_parameter_mapping()
    parameters["UB_MATRIX_SOURCE_PV"] = "28idbSoft:spec:UB"

    replacement = update_raw_profile(raw, "sim", parameters)

    assert replacement["IOC_RSM_PARAMETER"]["UB_MATRIX_SOURCE_PV"] == "28idbSoft:spec:UB"


def test_update_raw_profile_round_trips_distance_source_pv():
    raw = _raw()
    parameters = default_parameter_mapping()
    parameters["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"] = "6idb1:distance.RBV"

    replacement = update_raw_profile(raw, "sim", parameters)

    assert (
        replacement["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["DISTANCE_SOURCE_PV"]
        == "6idb1:distance.RBV"
    )


def test_adoption_diff_surfaces_ub_matrix_source_pv_change():
    raw = _raw()
    candidate = default_parameter_mapping()
    candidate["UB_MATRIX_SOURCE_PV"] = "28idbSoft:spec:UB"

    details = adoption_diff(raw, "sim", candidate)

    assert "UB_MATRIX_SOURCE_PV" in details
    assert "28idbSoft:spec:UB" in details


# --- Detector-value round trip (unknown keys / original numeric types) -----


def test_format_and_parse_detector_value_round_trip_scalars_and_pairs():
    assert _format_detector_value(None) == ""
    assert _format_detector_value("z-") == "z-"
    assert _format_detector_value(300.0) == "300.0"
    assert _format_detector_value([300, 300]) == "300, 300"

    assert _parse_detector_value("z-") == "z-"
    assert _parse_detector_value("300") == 300
    assert _parse_detector_value("300.0") == 300.0
    assert _parse_detector_value("300, 300") == [300, 300]
    assert _parse_detector_value("1.5, 2") == [1.5, 2]


def test_format_detector_value_preserves_float_type_marker():
    # 300.0 must render as "300.0", not "300" -- otherwise round-tripping
    # through the text field would quietly rewrite a float to an int.
    assert _format_detector_value(300.0) != _format_detector_value(300)


def test_update_raw_profile_preserves_unknown_detector_setup_keys():
    # An unrecognized DETECTOR_SETUP key (e.g. from a newer schema, or a
    # hand-edited TOML) must survive a save even though the validated profile
    # the GUI edits never sees it -- validate_parameter_profile legitimately
    # drops unrecognized keys when building the canonical view, so this
    # preservation has to happen from the raw baseline in update_raw_profile,
    # not from whatever the editor round-trips.
    raw = _raw()
    raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["VENDOR_SPECIFIC_QUIRK"] = "keep-me"
    profile = profile_from_raw(raw)
    assert "VENDOR_SPECIFIC_QUIRK" not in profile.detector_setup

    # The GUI never saw the unknown key, so it submits the validated mapping
    # unchanged plus one real edit elsewhere.
    submitted = profile.parameter_mapping()
    submitted["ENERGY_SOURCE_PV"] = "11"

    replacement = update_raw_profile(raw, "sim", submitted)

    assert (
        replacement["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["VENDOR_SPECIFIC_QUIRK"]
        == "keep-me"
    )
    assert replacement["IOC_RSM_PARAMETER"]["ENERGY_SOURCE_PV"] == "11"


def test_update_raw_profile_preserves_original_int_type_for_detector_shape():
    raw = _raw()
    raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["DETECTOR_SHAPE"] = [512, 512]
    profile = profile_from_raw(raw)
    assert profile.detector_setup["DETECTOR_SHAPE"] == [512, 512]

    replacement = update_raw_profile(raw, "sim", profile.parameter_mapping())

    shape = replacement["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["DETECTOR_SHAPE"]
    assert shape == [512, 512]
    assert all(isinstance(value, int) for value in shape)


# --- Change-review collapse: covered end-to-end via the GUI test module ----
# (unsaved_changes_rows/apply_change_decisions need a constructed
# SimulatorWindow; see tests/unit/test_rsm_parameter_ioc_gui.py)


# --- Resolved-profile-identity activation mismatch -------------------------


def test_classify_activation_mismatch_detects_moved_profile():
    error = _classify_activation_mismatch(5, 7, saved={}, reloaded={})

    assert isinstance(error, ActiveProfileChanged)
    assert error.expected == 5
    assert error.current == 7


def test_classify_activation_mismatch_detects_concurrent_content_change():
    saved = {"IOC_RSM_PARAMETER": {"SAMPLE_AXES": [{"DIRECTION": "x+"}]}}
    reloaded = {"IOC_RSM_PARAMETER": {"SAMPLE_AXES": [{"DIRECTION": "y+"}]}}

    error = _classify_activation_mismatch(5, 5, saved=saved, reloaded=reloaded)

    assert isinstance(error, ProfileContentMismatch)
    assert error.count == 1
    assert any("y+" in line for line in error.differences)


def test_describe_config_differences_caps_at_limit_with_overflow_line():
    saved = {f"k{i}": i for i in range(20)}
    reloaded = {f"k{i}": i + 1 for i in range(20)}

    lines, total = _describe_config_differences(saved, reloaded, limit=5)

    assert total == 20
    assert len(lines) == 6
    assert lines[-1] == "...and 15 more"


def test_describe_config_differences_detects_empty_section_vs_absent_key():
    # Regression: an empty dict and an absent key used to flatten identically,
    # so a real difference (one side has {}, the other side dropped the key
    # entirely) silently vanished -- the caller only invokes this because the
    # two raw configs are already known to be unequal, so reporting zero
    # differences here is always wrong when it happens.
    saved = {"IOC_RSM_PARAMETER": {"X": 1}, "METADATA_CA": {}}
    reloaded = {"IOC_RSM_PARAMETER": {"X": 1}}
    assert saved != reloaded

    lines, total = _describe_config_differences(saved, reloaded)

    assert total == 1
    assert lines and "METADATA_CA" in lines[0]


def test_pretty_config_path_names_axis_rows_one_indexed():
    assert (
        _pretty_config_path("IOC_RSM_PARAMETER.SAMPLE_AXES.0.DIRECTION")
        == "Sample axis 1 - DIRECTION"
    )
    assert (
        _pretty_config_path("IOC_RSM_PARAMETER.DETECTOR_AXES.2.SOURCE_PV")
        == "Detector axis 3 - SOURCE_PV"
    )


def test_profile_display_name_never_shows_a_bare_database_id_for_a_path():
    assert _profile_display_name(None) == "the auto-selected profile"
    assert _profile_display_name("/path/to/my_profile.toml") == "my_profile.toml"
