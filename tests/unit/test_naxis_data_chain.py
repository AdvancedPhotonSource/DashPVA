# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Arbitrary-axis contracts for HDF5 persistence and discovery."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import toml

from dashpva.utils.config.resolver import resolve_profile_config
from dashpva.utils.hdf5_loader import discover_hkl_axis_labels
from dashpva.utils.hdf5_writer import HDF5Writer
from dashpva.utils.metadata_converter import _convert_single_file, convert_files_or_dir
from dashpva.utils.rsm_converter import RSMConverter
from dashpva.utils.rsm_geometry import direction_vector

from ._synthetic_hdf5 import make_synthetic_scan_h5


class _Reader:
    def __init__(self, config):
        self.config = config


def _axis(role: str, number: int) -> dict[str, str]:
    return {
        "LABEL": f"{role.title()} {number}",
        "RECORD_NAME": f"{role[0].upper()}{number}",
        "SOURCE_PV": f"motor:{role}:{number}",
        "DIRECTION": "x+" if number % 2 else "z-",
        "ANGLE_UNITS": "deg",
    }


def _config(sample_count: int, detector_count: int) -> dict:
    raw = {
        "IOC_PREFIX": "test:",
        "IOC_RSM_PARAMETER": {
            "SAMPLE_AXES": [_axis("sample", number) for number in range(1, sample_count + 1)],
            "DETECTOR_AXES": [
                _axis("detector", number) for number in range(1, detector_count + 1)
            ],
            "ENERGY_UNITS": "keV",
            "SAMPLE_ORIENTATION": "sam",
        },
    }
    return resolve_profile_config(raw)


def _writer_data(config: dict) -> dict:
    metadata = {}
    for role in ("SAMPLE", "DETECTOR"):
        number = 1
        while (section := config["HKL"].get(f"{role}_CIRCLE_AXIS_{number}")):
            metadata[section["AXIS_NUMBER"]] = [number]
            metadata[section["DIRECTION_AXIS"]] = ["x+" if number % 2 else "z-"]
            metadata[section["POSITION"]] = [float(number)]
            metadata[section["SPEC_MOTOR_NAME"]] = [f"{role.title()} {number}"]
            number += 1
    return {
        "images": [np.arange(4, dtype=np.uint16)],
        "attributes": [{}],
        "rsm": None,
        "shape": (2, 2),
        "len_images": 1,
        "len_attributes": 1,
        "HKL_IN_CONFIG": True,
        "metadata": metadata,
    }


def _write_scan(path: Path, sample_count: int, detector_count: int) -> None:
    config = _config(sample_count, detector_count)
    HDF5Writer(str(path), _Reader(config)).h5_save(
        str(path), _writer_data(config), compress=False
    )


@pytest.mark.parametrize("count", [0, 3, 8, 10])
def test_numbered_axis_discovery_has_no_fixed_limit(tmp_path, count):
    path = tmp_path / f"{count}_axes.h5"
    with h5py.File(path, "w") as h5_file:
        hkl = h5_file.create_group("entry/data/metadata/HKL")
        for number in reversed(range(1, count + 1)):
            hkl.create_group(f"SAMPLE_CIRCLE_AXIS_{number}")

    with h5py.File(path, "r") as h5_file:
        sample_paths, detector_paths = RSMConverter()._resolve_circle_paths(h5_file)

    assert sample_paths == [
        f"entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_{number}"
        for number in range(1, count + 1)
    ]
    assert detector_paths == []


def test_writer_round_trips_fifth_axis_and_builds_both_nexus_chains(tmp_path):
    path = tmp_path / "many_axes.h5"
    config = _config(sample_count=10, detector_count=8)
    config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["VENDOR_EXTENSION"] = (
        config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["POSITION"]
    )
    HDF5Writer(str(path), _Reader(config)).h5_save(
        str(path), _writer_data(config), compress=False
    )

    converter = RSMConverter()
    with h5py.File(path, "r") as h5_file:
        sample_paths, detector_paths = converter._resolve_circle_paths(h5_file)
        sample_directions, sample_positions, detector_directions, detector_positions = (
            converter.get_sample_and_detector_circles(h5_file, frame=0)
        )

        assert len(sample_paths) == len(sample_directions) == len(sample_positions) == 10
        assert len(detector_paths) == len(detector_directions) == len(detector_positions) == 8
        assert sample_positions[4] == 5.0
        assert detector_positions[7] == 8.0

        sample_link = h5_file.get("entry/sample/geometry/S10", getlink=True)
        detector_link = h5_file.get(
            "entry/instrument/detector/transformations/D8", getlink=True
        )
        assert isinstance(sample_link, h5py.SoftLink)
        assert sample_link.path.endswith("/SAMPLE_CIRCLE_AXIS_10/POSITION")
        assert isinstance(detector_link, h5py.SoftLink)
        assert detector_link.path.endswith("/DETECTOR_CIRCLE_AXIS_8/POSITION")

        sample_10 = h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_10/POSITION"
        ]
        assert sample_10.attrs["depends_on"] == "/entry/sample/geometry/S9"
        np.testing.assert_array_equal(sample_10.attrs["vector"], [0.0, 0.0, -1.0])
        assert sample_10.attrs["transformation_type"] == "rotation"
        assert sample_10.attrs["long_name"] == "Sample 10"
        assert h5_file["entry/sample/depends_on"].asstr()[()] == (
            "/entry/sample/geometry/S10"
        )
        assert h5_file["entry/instrument/detector/depends_on"].asstr()[()] == (
            "/entry/instrument/detector/transformations/D8"
        )
        assert "VENDOR_EXTENSION" not in h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1"
        ]
        assert h5_file["entry/data/metadata/HKL/SAMPLE_ORIENTATION"].asstr()[()] == (
            "sam"
        )
        assert h5_file["entry/data/metadata/HKL/SPEC/ENERGY_UNITS"].asstr()[()] == (
            "keV"
        )
        assert "entry/data/metadata/HKL/INPLANE_REFERENCE_DIRECTION" in h5_file
        assert "entry/data/metadata/HKL/INPLANE_REFERENCE_DIRECITON" not in h5_file


def test_canonical_axis_number_is_scalar_ordinal_not_observed_channel(tmp_path):
    path = tmp_path / "canonical_axis_number.h5"
    config = _config(sample_count=2, detector_count=1)
    data = _writer_data(config)
    data["images"] *= 3
    data["len_images"] = data["len_attributes"] = 3
    for group_name in (
        "SAMPLE_CIRCLE_AXIS_1",
        "SAMPLE_CIRCLE_AXIS_2",
        "DETECTOR_CIRCLE_AXIS_1",
    ):
        channel = config["HKL"][group_name]["AXIS_NUMBER"]
        data["metadata"][channel] = [90, 91, 92]

    HDF5Writer(str(path), _Reader(config)).h5_save(
        str(path), data, compress=False
    )

    with h5py.File(path, "r") as h5_file:
        hkl = h5_file["entry/data/metadata/HKL"]
        assert hkl["SAMPLE_CIRCLE_AXIS_1/AXIS_NUMBER"].shape == ()
        assert hkl["SAMPLE_CIRCLE_AXIS_1/AXIS_NUMBER"][()] == 1
        assert hkl["SAMPLE_CIRCLE_AXIS_2/AXIS_NUMBER"][()] == 2
        assert hkl["DETECTOR_CIRCLE_AXIS_1/AXIS_NUMBER"][()] == 1


def test_partial_canonical_profile_still_uses_scalar_axis_ordinals(tmp_path):
    path = tmp_path / "partial_canonical_axis_number.h5"
    config = _config(sample_count=2, detector_count=1)
    parameters = config["IOC_RSM_PARAMETER"]
    del parameters["SAMPLE_AXES"]
    del parameters["DETECTOR_AXES"]
    data = _writer_data(config)
    data["images"] *= 2
    data["len_images"] = data["len_attributes"] = 2
    sample = config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]
    data["metadata"][sample["AXIS_NUMBER"]] = [90, 91]

    HDF5Writer(str(path), _Reader(config)).h5_save(
        str(path), data, compress=False
    )

    with h5py.File(path, "r") as h5_file:
        axis_number = h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1/AXIS_NUMBER"
        ]
        assert axis_number.shape == ()
        assert axis_number[()] == 1


def test_legacy_axis_number_channel_is_validated_and_collapsed_to_scalar(tmp_path):
    path = tmp_path / "legacy_axis_number.h5"
    config = _config(sample_count=1, detector_count=1)
    del config["IOC_RSM_PARAMETER"]
    data = _writer_data(config)
    data["images"] *= 3
    data["len_images"] = data["len_attributes"] = 3
    sample = config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]
    data["metadata"][sample["AXIS_NUMBER"]] = [7, 7, 7]

    HDF5Writer(str(path), _Reader(config)).h5_save(
        str(path), data, compress=False
    )

    with h5py.File(path, "r") as h5_file:
        axis_number = h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1/AXIS_NUMBER"
        ]
        assert axis_number.shape == ()
        assert axis_number[()] == 7


def test_varying_legacy_axis_number_fails_before_destination_is_opened(tmp_path):
    path = tmp_path / "existing_axis_number.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("sentinel", data=42)
    config = _config(sample_count=1, detector_count=1)
    del config["IOC_RSM_PARAMETER"]
    data = _writer_data(config)
    data["images"] *= 3
    data["len_images"] = data["len_attributes"] = 3
    sample = config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]
    data["metadata"][sample["AXIS_NUMBER"]] = [1, 2, 3]

    with pytest.raises(ValueError, match="AXIS_NUMBER changes within the scan"):
        HDF5Writer(str(path), _Reader(config)).h5_save(
            str(path), data, compress=False
        )

    with h5py.File(path, "r") as h5_file:
        assert h5_file["sentinel"][()] == 42


def test_canonical_spec_motor_name_is_not_replaced_by_label(tmp_path):
    path = tmp_path / "spec_motor_name.h5"
    config = _config(sample_count=1, detector_count=1)
    axis = config["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]
    axis["LABEL"] = "friendly label"
    axis["SPEC_MOTOR_NAME"] = "th"
    data = _writer_data(config)
    spec_channel = config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["SPEC_MOTOR_NAME"]
    del data["metadata"][spec_channel]

    HDF5Writer(str(path), _Reader(config)).h5_save(
        str(path), data, compress=False
    )

    with h5py.File(path, "r") as h5_file:
        axis_group = h5_file["entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1"]
        assert axis_group["LABEL"].asstr()[()] == "friendly label"
        assert axis_group["SPEC_MOTOR_NAME"].asstr()[()] == "th"


def test_absent_canonical_spec_motor_name_uses_label_compatibility_value(tmp_path):
    path = tmp_path / "spec_motor_name_fallback.h5"
    config = _config(sample_count=1, detector_count=1)
    axis = config["IOC_RSM_PARAMETER"]["SAMPLE_AXES"][0]
    axis["LABEL"] = "historical motor label"
    data = _writer_data(config)
    spec_channel = config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["SPEC_MOTOR_NAME"]
    del data["metadata"][spec_channel]

    HDF5Writer(str(path), _Reader(config)).h5_save(
        str(path), data, compress=False
    )

    with h5py.File(path, "r") as h5_file:
        axis_group = h5_file["entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1"]
        assert axis_group["SPEC_MOTOR_NAME"].asstr()[()] == (
            "historical motor label"
        )


def test_kappa_direction_has_a_nexus_vector(tmp_path):
    path = tmp_path / "kappa.h5"
    config = _config(sample_count=1, detector_count=1)
    data = _writer_data(config)
    direction_channel = config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["DIRECTION_AXIS"]
    data["metadata"][direction_channel] = ["k+"]

    HDF5Writer(str(path), _Reader(config)).h5_save(str(path), data, compress=False)

    with h5py.File(path, "r") as h5_file:
        np.testing.assert_allclose(
            h5_file[
                "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1/POSITION"
            ].attrs["vector"],
            direction_vector("k+"),
        )


def test_axis_preflight_does_not_truncate_existing_file(tmp_path):
    path = tmp_path / "existing.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("sentinel", data=42)

    config = _config(sample_count=5, detector_count=1)
    data = _writer_data(config)
    missing_channel = config["HKL"]["SAMPLE_CIRCLE_AXIS_5"]["POSITION"]
    del data["metadata"][missing_channel]

    with pytest.raises(ValueError, match="SAMPLE_CIRCLE_AXIS_5"):
        HDF5Writer(str(path), _Reader(config)).h5_save(
            str(path), data, compress=False
        )

    with h5py.File(path, "r") as h5_file:
        assert h5_file["sentinel"][()] == 42


def test_axis_preflight_rejects_varying_and_role_invalid_directions(tmp_path):
    config = _config(sample_count=1, detector_count=1)
    data = _writer_data(config)
    data["images"] = data["images"] * 2
    data["len_images"] = 2
    data["len_attributes"] = 2
    sample_direction = config["HKL"]["SAMPLE_CIRCLE_AXIS_1"]["DIRECTION_AXIS"]
    data["metadata"][sample_direction] = ["x+", "z+"]

    with pytest.raises(ValueError, match="changes within the scan"):
        HDF5Writer(str(tmp_path / "varying.h5"), _Reader(config)).h5_save(
            str(tmp_path / "varying.h5"), data, compress=False
        )

    data = _writer_data(config)
    detector_direction = config["HKL"]["DETECTOR_CIRCLE_AXIS_1"][
        "DIRECTION_AXIS"
    ]
    data["metadata"][detector_direction] = ["k+"]
    with pytest.raises(ValueError, match="Invalid detector rotation direction"):
        HDF5Writer(str(tmp_path / "invalid.h5"), _Reader(config)).h5_save(
            str(tmp_path / "invalid.h5"), data, compress=False
        )


def test_numbered_groups_take_precedence_per_role_over_named_legacy_groups(tmp_path):
    path = tmp_path / "mixed.h5"
    with h5py.File(path, "w") as h5_file:
        hkl = h5_file.create_group("entry/data/metadata/HKL")
        hkl.create_group("SAMPLE_CIRCLE_AXIS_2")
        hkl.create_group("SAMPLE_CIRCLE_AXIS_1")
        hkl.create_group("MU")
        hkl.create_group("NU")
        hkl.create_group("DELTA")

    with h5py.File(path, "r") as h5_file:
        sample_paths, detector_paths = RSMConverter()._resolve_circle_paths(h5_file)

    assert [path.rsplit("/", 1)[-1] for path in sample_paths] == [
        "SAMPLE_CIRCLE_AXIS_1",
        "SAMPLE_CIRCLE_AXIS_2",
    ]
    assert [path.rsplit("/", 1)[-1] for path in detector_paths] == ["NU", "DELTA"]


def test_converter_reads_canonical_vector_names_and_energy_units(tmp_path):
    path = tmp_path / "canonical_physics.h5"
    make_synthetic_scan_h5(str(path), n_frames=1, shape=(2, 2))
    with h5py.File(path, "r+") as h5_file:
        hkl = h5_file["entry/data/metadata/HKL"]
        hkl.move("INPLANE_REFERENCE_DIRECITON", "INPLANE_REFERENCE_DIRECTION")
        hkl.move(
            "SAMPLE_SURFACE_NORMAL_DIRECITON",
            "SAMPLE_SURFACE_NORMAL_DIRECTION",
        )
        spec = hkl["SPEC"]
        del spec["ENERGY_VALUE"]
        spec.create_dataset("ENERGY_VALUE", data=10_000.0)
        spec.create_dataset("ENERGY_UNITS", data="eV")

    with h5py.File(path, "r") as h5_file:
        _, inplane, surface, _, energy = RSMConverter().get_physics_params(h5_file)

    assert inplane == [1.0, 0.0, 0.0]
    assert surface == [0.0, 0.0, 1.0]
    assert energy == 10_000.0


def test_axis_label_discovery_sorts_suffix_10_numerically(tmp_path):
    path = tmp_path / "labels.h5"
    with h5py.File(path, "w") as h5_file:
        hkl = h5_file.create_group("entry/data/metadata/HKL")
        for number in (10, 2, 1):
            group = hkl.create_group(f"SAMPLE_CIRCLE_AXIS_{number}")
            group.create_dataset("NAME", data=f"S{number}")

    labels = discover_hkl_axis_labels(str(path))

    assert labels["sample_axes"] == ["S1", "S2", "S10"]


def test_metadata_converter_uses_effective_canonical_hkl_mapping(tmp_path):
    h5_path = tmp_path / "canonical.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset(
            "entry/data/data", data=np.ones((2, 2, 2), dtype=np.float32)
        )
        motor_positions = h5_file.create_group("entry/data/metadata/motor_positions")
        motor_positions.create_dataset("motor:S1.RBV", data=[1.0, 2.0])
        energy = h5_file.create_dataset(
            "entry/data/metadata/ca/Energy", data=[10.0, 10.0]
        )
        energy.attrs["pv_name"] = "beam:energy"

    sample_axis = _axis("sample", 1)
    sample_axis["SOURCE_PV"] = "motor:S1.RBV"
    static_axis = _axis("sample", 2)
    static_axis["SOURCE_PV"] = "3.5"
    raw = {
        "IOC_PREFIX": "test:",
        "HKL": {
            "VENDOR_EXTENSION": {"COPY": "motor:S1.RBV"},
            "SAMPLE_CIRCLE_AXIS_1": {"CUSTOM": "motor:S1.RBV"},
        },
        "IOC_RSM_PARAMETER": {
            "SAMPLE_AXES": [sample_axis, static_axis],
            "DETECTOR_AXES": [],
            "ENERGY_SOURCE_PV": "beam:energy",
            "ENERGY_UNITS": "keV",
            "SAMPLE_ORIENTATION": "sam",
            "UB_MATRIX": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "PRIMARY_BEAM_DIRECTION": [0.0, 1.0, 0.0],
            "INPLANE_REFERENCE_DIRECTION": [1.0, 0.0, 0.0],
            "SAMPLE_SURFACE_NORMAL_DIRECTION": [0.0, 0.0, 1.0],
            "DETECTOR_SETUP": {
                "PIXEL_DIRECTION_1": "x+",
                "PIXEL_DIRECTION_2": "z+",
                "CENTER_CHANNEL_PIXEL": [1.0, 1.0],
                "SIZE": [2.0, 2.0],
                "DISTANCE": 500.0,
                "UNITS": "mm",
            },
        },
    }
    toml_path = tmp_path / "canonical.toml"
    with toml_path.open("w", encoding="utf-8") as stream:
        toml.dump(raw, stream)

    assert convert_files_or_dir(
        str(toml_path), str(h5_path), include=True, in_place=True
    ) == [str(h5_path)]

    with h5py.File(h5_path, "r") as h5_file:
        position = h5_file.get(
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1/POSITION", getlink=True
        )
        assert isinstance(position, h5py.SoftLink)
        assert position.path == "/entry/data/metadata/motor_positions/SAMPLE 1"
        assert h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1/NAME"
        ].asstr()[()] == "Sample 1"
        assert h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1/DIRECTION_AXIS"
        ].asstr()[()] == "x+"
        assert h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_2/POSITION"
        ][()] == 3.5
        assert "entry/data/metadata/HKL/VENDOR_EXTENSION" not in h5_file
        assert "CUSTOM" not in h5_file[
            "entry/data/metadata/HKL/SAMPLE_CIRCLE_AXIS_1"
        ]

        converter = RSMConverter()
        geometry = converter.build_file_geometry(h5_file)
        qx, qy, qz = converter.q_for_frames(geometry, h5_file, 0, 2)

    assert qx.shape == qy.shape == qz.shape == (2, 2, 2)


def test_canonical_converter_fails_when_energy_source_is_missing(tmp_path):
    h5_path = tmp_path / "missing_energy.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_group("entry/data/metadata")

    raw = {
        "IOC_PREFIX": "test:",
        "IOC_RSM_PARAMETER": {
            "SAMPLE_AXES": [],
            "DETECTOR_AXES": [],
            "ENERGY_SOURCE_PV": "beam:energy",
            "ENERGY_UNITS": "keV",
            "SAMPLE_ORIENTATION": "x+",
            "UB_MATRIX": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "PRIMARY_BEAM_DIRECTION": [0.0, 1.0, 0.0],
            "INPLANE_REFERENCE_DIRECTION": [1.0, 0.0, 0.0],
            "SAMPLE_SURFACE_NORMAL_DIRECTION": [0.0, 0.0, 1.0],
            "DETECTOR_SETUP": {
                "PIXEL_DIRECTION_1": "x+",
                "PIXEL_DIRECTION_2": "z+",
                "CENTER_CHANNEL_PIXEL": [1.0, 1.0],
                "SIZE": [2.0, 2.0],
                "DISTANCE": 500.0,
                "UNITS": "mm",
            },
        },
    }
    toml_path = tmp_path / "missing_energy.toml"
    with toml_path.open("w", encoding="utf-8") as stream:
        toml.dump(raw, stream)

    with pytest.raises(ValueError, match="Missing required photon energy source data"):
        _convert_single_file(
            h5_path,
            toml_path,
            "entry/data/metadata",
            True,
            True,
            tmp_path,
            False,
        )
