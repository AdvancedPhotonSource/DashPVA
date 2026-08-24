# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Raw-preserving persistence and live IOC three-way merge contracts."""

from __future__ import annotations

import copy

from dashpva.utils.rsm_parameter_config import (
    default_parameter_mapping,
    merge_live_records,
    profile_from_raw,
    update_raw_profile,
)


def _raw() -> dict:
    parameters = default_parameter_mapping()
    parameters["SAMPLE_AXES"] = parameters["SAMPLE_AXES"][:2]
    parameters["DETECTOR_AXES"] = parameters["DETECTOR_AXES"][:1]
    return {
        "IOC_PREFIX": "sim",
        "IOC_RSM_PARAMETER": parameters,
        "VENDOR": {"keep": [1, 2, 3]},
    }


def _origins(sample=(0, 1), detector=(0,)):
    return {"SAMPLE_AXES": sample, "DETECTOR_AXES": detector}


def test_unchanged_normalized_form_preserves_exact_raw_document():
    raw = _raw()
    del raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]["ANGLE_UNITS"]
    raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]["VENDOR_AXIS"] = {"mode": 2}
    raw["IOC_RSM_PARAMETER"]["VENDOR_PARAMETER"] = (1, 2)
    raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["SIZE"] = [28, 28]
    del raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["CENTER_CHANNEL_PIXEL"]

    profile = profile_from_raw(raw)
    replacement = update_raw_profile(
        raw,
        profile.prefix,
        profile.parameter_mapping(),
        axis_origins=_origins(),
    )

    assert replacement == raw


def test_pixel_size_only_detector_setup_does_not_grow_default_size():
    raw = _raw()
    setup = raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]
    del setup["SIZE"]
    setup["PIXEL_SIZE"] = [0.075, 0.075]
    profile = profile_from_raw(raw)

    replacement = update_raw_profile(
        raw,
        profile.prefix,
        profile.parameter_mapping(),
        axis_origins=_origins(),
    )
    stored = replacement["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]

    assert "SIZE" not in stored
    assert stored["PIXEL_SIZE"] == [0.075, 0.075]


def test_unchanged_absent_optional_tables_and_axis_lists_stay_absent():
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


def test_reorder_and_rename_follow_stable_axis_identity_and_keep_extensions():
    raw = _raw()
    raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]["VENDOR_AXIS"] = "mu"
    raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][1]["VENDOR_AXIS"] = "eta"
    form = profile_from_raw(raw).parameter_mapping()
    form["SAMPLE_AXES"] = [form["SAMPLE_AXES"][1], form["SAMPLE_AXES"][0]]
    form["SAMPLE_AXES"][0]["LABEL"] = "renamed eta"
    form["SAMPLE_AXES"][0]["RECORD_NAME"] = "EtaRenamed"

    replacement = update_raw_profile(
        raw,
        "sim:",
        form,
        axis_origins=_origins(sample=(1, 0)),
    )
    axes = replacement["IOC_RSM_PARAMETER"]["SAMPLE_AXES"]

    assert axes[0]["VENDOR_AXIS"] == "eta"
    assert axes[0]["LABEL"] == "renamed eta"
    assert axes[0]["RECORD_NAME"] == "EtaRenamed"
    assert axes[1]["VENDOR_AXIS"] == "mu"


def test_live_axis_change_follows_original_identity_after_reorder_and_rename():
    raw = _raw()
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()
    form["SAMPLE_AXES"] = [form["SAMPLE_AXES"][1], form["SAMPLE_AXES"][0]]
    form["SAMPLE_AXES"][0]["RECORD_NAME"] = "EtaRenamed"

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {"sim:Eta:DirectionAxis": "y-"},
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(sample=(1, 0)),
    )

    assert conflicts == []
    assert adopted
    assert form["SAMPLE_AXES"][0]["DIRECTION"] == "y-"
    assert form["SAMPLE_AXES"][1]["DIRECTION"] == "x+"


def test_live_merge_compares_normalized_semantics_not_raw_spelling():
    raw = _raw()
    raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][1]["DIRECTION"] = "Z-"
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {"sim:Eta:DirectionAxis": "y-"},
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(),
    )

    assert conflicts == []
    assert adopted
    assert form["SAMPLE_AXES"][1]["DIRECTION"] == "y-"


def test_divergent_form_and_live_axis_edits_conflict_after_reorder():
    raw = _raw()
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()
    form["SAMPLE_AXES"] = [form["SAMPLE_AXES"][1], form["SAMPLE_AXES"][0]]
    form["SAMPLE_AXES"][0]["DIRECTION"] = "y+"

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {"sim:Eta:DirectionAxis": "x+"},
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(sample=(1, 0)),
    )

    assert adopted == []
    assert len(conflicts) == 1
    assert form["SAMPLE_AXES"][0]["DIRECTION"] == "y+"


def test_identical_concurrent_axis_edit_is_accepted_without_conflict():
    raw = _raw()
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()
    form["SAMPLE_AXES"][1]["DIRECTION"] = "x+"

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {"sim:Eta:DirectionAxis": "x+"},
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(),
    )

    assert adopted == []
    assert conflicts == []


def test_live_spec_motor_name_adopts_distinct_field_without_touching_label():
    raw = _raw()
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {"sim:Mu:SpecMotorName": "th"},
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(),
    )

    assert conflicts == []
    assert adopted
    assert form["SAMPLE_AXES"][0]["SPEC_MOTOR_NAME"] == "th"
    assert form["SAMPLE_AXES"][0]["LABEL"] == "Mu"


def test_absent_spec_motor_name_label_fallback_is_not_a_live_change():
    raw = _raw()
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {"sim:Mu:SpecMotorName": "Mu"},
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(),
    )

    assert adopted == []
    assert conflicts == []
    assert "SPEC_MOTOR_NAME" not in raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]


def test_removed_axis_with_concurrent_live_edit_is_a_conflict():
    raw = _raw()
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()
    del form["SAMPLE_AXES"][1]

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {"sim:Eta:DirectionAxis": "x+"},
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(sample=(0,)),
    )

    assert adopted == []
    assert len(conflicts) == 1
    assert "axis removed" in conflicts[0]


def test_live_adoption_retains_existing_scalar_type_and_introduces_missing_key():
    raw = _raw()
    raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["DISTANCE"] = 400
    del raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["SIZE"]
    baseline = profile_from_raw(raw)
    form = baseline.parameter_mapping()

    adopted, conflicts = merge_live_records(
        form,
        baseline,
        {
            "sim:DetectorSetup:Distance": 401.0,
            "sim:DetectorSetup:Size": [29.0, 30.0],
        },
        raw["IOC_RSM_PARAMETER"],
        baseline.parameter_mapping(),
        axis_origins=_origins(),
    )
    replacement = update_raw_profile(
        raw,
        "sim:",
        copy.deepcopy(form),
        axis_origins=_origins(),
    )
    setup = replacement["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]

    assert conflicts == []
    assert len(adopted) == 2
    assert setup["DISTANCE"] == 401
    assert type(setup["DISTANCE"]) is int
    assert setup["SIZE"] == [29.0, 30.0]


def test_non_integral_edit_is_never_truncated_to_match_raw_int_type():
    raw = _raw()
    raw["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["DISTANCE"] = 400
    form = profile_from_raw(raw).parameter_mapping()
    form["DETECTOR_SETUP"]["DISTANCE"] = 400.5

    replacement = update_raw_profile(
        raw,
        "sim:",
        form,
        axis_origins=_origins(),
    )

    distance = replacement["IOC_RSM_PARAMETER"]["DETECTOR_SETUP"]["DISTANCE"]
    assert distance == 400.5
    assert type(distance) is float
