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

"""Contracts for persisted (raw) versus runtime-effective RSM profiles."""

import copy
from pathlib import Path

import pytest
import toml

from dashpva.utils.config.resolver import resolve_profile_config


def _canonical_profile():
    return {
        "IOC_PREFIX": "6idb",
        "HKL": {
            "SAMPLE_CIRCLE_AXIS_1": {
                "POSITION": "hand-authored:position",
                "VENDOR_EXTENSION": "keep-me",
            },
            "SAMPLE_CIRCLE_AXIS_2": {"POSITION": "obsolete:position"},
            "CUSTOM_EXTENSION": {"ENABLED": True},
        },
        "IOC_RSM_PARAMETER": {
            "SAMPLE_AXES": [
                {
                    "LABEL": "Mu",
                    "RECORD_NAME": "Mu",
                    "DIRECTION": "x+",
                    "SOURCE_PV": "6idb1:m28.RBV",
                    "ANGLE_UNITS": "deg",
                }
            ],
            "DETECTOR_AXES": [
                {
                    "LABEL": "Delta",
                    "RECORD_NAME": "Delta",
                    "DIRECTION": "z-",
                    "SOURCE_PV": "6idb1:m18.RBV",
                    "ANGLE_UNITS": "deg",
                }
            ],
            "ENERGY_SOURCE_PV": "6idb:spec:Energy",
            "ENERGY_UNITS": "keV",
        },
    }


def test_legacy_profile_is_equal_but_deeply_detached():
    raw = {
        "IOC_PREFIX": "legacy",
        "HKL": {"SPEC": {"ENERGY_VALUE": "hand-authored:energy"}},
    }

    effective = resolve_profile_config(raw)

    assert effective == raw
    assert effective is not raw
    assert effective["HKL"] is not raw["HKL"]
    effective["HKL"]["SPEC"]["ENERGY_VALUE"] = "changed"
    assert raw["HKL"]["SPEC"]["ENERGY_VALUE"] == "hand-authored:energy"


def test_canonical_profile_generates_managed_hkl_without_mutating_raw():
    raw = _canonical_profile()
    before = copy.deepcopy(raw)

    effective = resolve_profile_config(raw)

    assert raw == before
    assert effective["IOC_PREFIX"] == "6idb:"
    assert effective["HKL"]["SAMPLE_CIRCLE_AXIS_1"] == {
        "VENDOR_EXTENSION": "keep-me",
        "AXIS_NUMBER": "6idb:Mu:AxisNumber",
        "DIRECTION_AXIS": "6idb:Mu:DirectionAxis",
        "POSITION": "6idb:Mu:Position",
    }
    assert effective["HKL"]["DETECTOR_CIRCLE_AXIS_1"]["POSITION"] == (
        "6idb:Delta:Position"
    )
    assert "SAMPLE_CIRCLE_AXIS_2" not in effective["HKL"]
    assert effective["HKL"]["CUSTOM_EXTENSION"] == {"ENABLED": True}
    assert effective["HKL"]["SPEC"]["ENERGY_VALUE"] == "6idb:spec:Energy:Value"
    assert "INPLANE_REFERENCE_DIRECTION" in effective["HKL"]
    assert "INPLANE_REFERENCE_DIRECITON" not in effective["HKL"]
    assert "SAMPLE_SURFACE_NORMAL_DIRECTION" in effective["HKL"]
    assert "SAMPLE_SURFACE_NORMAL_DIRECITON" not in effective["HKL"]


def test_empty_canonical_axis_lists_remove_all_legacy_managed_axes():
    raw = _canonical_profile()
    raw["HKL"]["MU"] = {"POSITION": "legacy:mu"}
    raw["HKL"]["NU"] = {"POSITION": "legacy:nu"}
    raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"] = []
    raw["IOC_RSM_PARAMETER"]["DETECTOR_AXES"] = []

    effective = resolve_profile_config(raw)

    assert not any("_CIRCLE_AXIS_" in name for name in effective["HKL"])
    assert "MU" not in effective["HKL"]
    assert "NU" not in effective["HKL"]
    assert "SAMPLE_CIRCLE_AXIS_1" in raw["HKL"]
    assert "MU" in raw["HKL"]
    assert "NU" in raw["HKL"]


def test_record_names_must_be_unique_across_sample_and_detector_roles():
    raw = _canonical_profile()
    raw["IOC_RSM_PARAMETER"]["DETECTOR_AXES"][0]["RECORD_NAME"] = "Mu"

    with pytest.raises(ValueError, match="duplicate RECORD_NAME"):
        resolve_profile_config(raw)


def test_prefixed_record_name_is_rejected_instead_of_double_prefixing():
    raw = _canonical_profile()
    raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]["RECORD_NAME"] = "other:Mu"

    with pytest.raises(ValueError, match="unprefixed record stem"):
        resolve_profile_config(raw)


def test_record_name_rejects_whitespace_that_would_create_an_invalid_pv():
    raw = _canonical_profile()
    raw["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]["RECORD_NAME"] = "Mu Position"

    with pytest.raises(ValueError, match="record stem"):
        resolve_profile_config(raw)


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_unsupported_canonical_schema_version_fails_before_hkl_generation(version):
    raw = _canonical_profile()
    raw["IOC_RSM_PARAMETER"]["SCHEMA_VERSION"] = version

    with pytest.raises(ValueError, match="SCHEMA_VERSION"):
        resolve_profile_config(raw)


def test_canonical_output_never_borrows_detector_prefix():
    raw = _canonical_profile()
    raw.pop("IOC_PREFIX")
    raw["DETECTOR_PREFIX"] = "detector"

    effective = resolve_profile_config(raw)

    assert effective["IOC_PREFIX"] == ""
    assert effective["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["POSITION"] == "Mu:Position"


def test_settings_keeps_raw_profile_and_exports_effective_snapshot(tmp_path):
    from dashpva.settings import Settings

    raw = _canonical_profile()
    profile_path = tmp_path / "canonical.toml"
    profile_path.write_text(toml.dumps(raw))

    settings = Settings.from_toml(str(profile_path))
    snapshot_path = Path(settings.ensure_path())
    try:
        snapshot = toml.load(snapshot_path)
        assert settings.RAW_CONFIG == raw
        assert settings.CONFIG["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["POSITION"] == (
            "6idb:Mu:Position"
        )
        assert toml.load(profile_path) == raw
        assert snapshot_path != profile_path
        assert snapshot == settings.CONFIG
    finally:
        snapshot_path.unlink(missing_ok=True)
