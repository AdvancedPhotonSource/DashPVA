# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Semantic HKL channel discovery contracts."""

import pytest

from dashpva.utils.config.hkl import (
    axis_field_channels,
    get_hkl_section,
    iter_hkl_axes,
    required_rsm_channels,
    semantic_hkl_channels,
)


def _static_hkl():
    return {
        "PRIMARY_BEAM_DIRECTION": {
            f"AXIS_NUMBER_{index}": f"primary.{index}"
            for index in range(1, 4)
        },
        "INPLANE_REFERENCE_DIRECTION": {
            f"AXIS_NUMBER_{index}": f"inplane.{index}"
            for index in range(1, 4)
        },
        "SAMPLE_SURFACE_NORMAL_DIRECITON": {
            f"AXIS_NUMBER_{index}": f"normal.{index}"
            for index in range(1, 4)
        },
        "SPEC": {
            "ENERGY_VALUE": "energy.RBV",
            "UB_MATRIX_VALUE": "ub.RBV",
            "ENERGY_UNITS": "energy.units",
            "VENDOR_EXTENSION": "not:a:channel",
        },
        "DETECTOR_SETUP": {
            "CENTER_CHANNEL_PIXEL": "detector.center",
            "DISTANCE": "detector.distance",
            "PIXEL_DIRECTION_1": "detector.direction1",
            "PIXEL_DIRECTION_2": "detector.direction2",
            "SIZE": "detector.size",
            "UNITS": "detector.units",
            "COMMENT": "ignore me",
        },
        "CUSTOM_EXTENSION": {"ENABLED": True},
    }


def _axis(index, stem):
    return {
        "AXIS_NUMBER": f"{stem}.axis.RBV",
        "DIRECTION_AXIS": f"{stem}.direction.RBV",
        "POSITION": f"{stem}.position.RBV",
        "SPEC_MOTOR_NAME": f"{stem}.name.RBV",
        "VENDOR_EXTENSION": f"{stem}.ignore.RBV",
    }


def test_axes_are_numeric_sorted_and_fields_are_dispatched_semantically():
    hkl = _static_hkl()
    hkl["SAMPLE_CIRCLE_AXIS_10"] = _axis(10, "ten")
    hkl["SAMPLE_CIRCLE_AXIS_2"] = _axis(2, "two")
    hkl["SAMPLE_CIRCLE_AXIS_1"] = _axis(1, "one")

    assert [axis.index for axis in iter_hkl_axes(hkl, "sample")] == [1, 2, 10]
    assert axis_field_channels(hkl, "sample", "POSITION") == (
        "one.position.RBV",
        "two.position.RBV",
        "ten.position.RBV",
    )
    assert axis_field_channels(hkl, "sample", "DIRECTION_AXIS") == (
        "one.direction.RBV",
        "two.direction.RBV",
        "ten.direction.RBV",
    )


def test_only_known_fields_become_metadata_channels():
    hkl = _static_hkl()
    hkl["SAMPLE_CIRCLE_AXIS_1"] = _axis(1, "sample")

    channels = semantic_hkl_channels(hkl)

    assert "sample.position.RBV" in channels
    assert "sample.direction.RBV" in channels
    assert "sample.ignore.RBV" not in channels
    assert "not:a:channel" not in channels
    assert "ignore me" not in channels
    assert True not in channels


def test_required_q_channels_exclude_names_units_and_extensions():
    hkl = _static_hkl()
    hkl["DETECTOR_CIRCLE_AXIS_1"] = _axis(1, "detector")

    channels = required_rsm_channels(hkl)

    assert "detector.direction.RBV" in channels
    assert "detector.position.RBV" in channels
    assert "detector.axis.RBV" not in channels
    assert "detector.name.RBV" not in channels
    assert "energy.units" not in channels
    assert "detector.units" not in channels


def test_zero_axis_roles_are_valid_for_channel_discovery():
    hkl = _static_hkl()

    assert iter_hkl_axes(hkl, "sample") == ()
    assert iter_hkl_axes(hkl, "detector") == ()
    assert required_rsm_channels(hkl)


def test_canonical_vector_name_wins_and_legacy_typo_remains_supported():
    hkl = _static_hkl()
    canonical = hkl["INPLANE_REFERENCE_DIRECTION"]
    hkl["INPLANE_REFERENCE_DIRECITON"] = {
        "AXIS_NUMBER_1": "legacy.1",
        "AXIS_NUMBER_2": "legacy.2",
        "AXIS_NUMBER_3": "legacy.3",
    }

    assert get_hkl_section(hkl, "INPLANE_REFERENCE_DIRECTION") is canonical

    del hkl["INPLANE_REFERENCE_DIRECTION"]
    assert get_hkl_section(hkl, "INPLANE_REFERENCE_DIRECTION") == {
        "AXIS_NUMBER_1": "legacy.1",
        "AXIS_NUMBER_2": "legacy.2",
        "AXIS_NUMBER_3": "legacy.3",
    }


def test_missing_semantic_field_fails_loudly():
    hkl = _static_hkl()
    hkl["SAMPLE_CIRCLE_AXIS_1"] = _axis(1, "sample")
    del hkl["SAMPLE_CIRCLE_AXIS_1"]["POSITION"]

    with pytest.raises(ValueError, match="POSITION"):
        required_rsm_channels(hkl)


class TestDerivedCircleAccessors:
    """settings.HKL_SAMPLE_CIRCLES / HKL_DETECTOR_CIRCLES.

    A convenience view for call sites that only need "the sample circles, in
    order". It must never disagree with the geometry the Q conversion builds,
    so it goes through the same resolution rsm_geometry uses.
    """

    def test_splits_by_role_in_numeric_order(self):
        from dashpva.settings import _circles_by_role

        hkl = {
            "SAMPLE_CIRCLE_AXIS_10": {"POSITION": "ioc:s10"},
            "SAMPLE_CIRCLE_AXIS_2": {"POSITION": "ioc:s2"},
            "SAMPLE_CIRCLE_AXIS_1": {"POSITION": "ioc:s1"},
            "DETECTOR_CIRCLE_AXIS_1": {"POSITION": "ioc:d1"},
            "SPEC": {"ENERGY_VALUE": "ioc:e"},
        }
        sample = _circles_by_role(hkl, "sample")
        assert [name for name, _ in sample] == [
            "SAMPLE_CIRCLE_AXIS_1", "SAMPLE_CIRCLE_AXIS_2", "SAMPLE_CIRCLE_AXIS_10"
        ]
        assert sample[0][1]["POSITION"] == "ioc:s1"
        assert [name for name, _ in _circles_by_role(hkl, "detector")] == [
            "DETECTOR_CIRCLE_AXIS_1"
        ]

    def test_falls_back_to_legacy_named_groups(self):
        from dashpva.settings import _circles_by_role

        hkl = {"MU": {"POSITION": "ioc:mu"}, "ETA": {"POSITION": "ioc:eta"},
               "NU": {"POSITION": "ioc:nu"}}
        assert [name for name, _ in _circles_by_role(hkl, "sample")] == ["MU", "ETA"]
        assert [name for name, _ in _circles_by_role(hkl, "detector")] == ["NU"]

    def test_empty_config_yields_empty_views(self):
        from dashpva.settings import _circles_by_role

        assert _circles_by_role({}, "sample") == []
        assert _circles_by_role({}, "detector") == []
