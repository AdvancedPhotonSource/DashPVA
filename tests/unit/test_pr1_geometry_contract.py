# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Behavior-preservation contracts for the shared PR 1 RSM geometry path."""

import h5py
import numpy as np
import pytest
import xrayutilities as xu

from dashpva.utils.rsm_converter import RSMConverter
from dashpva.utils.rsm_geometry import (
    DetectorModel,
    RotationAxis,
    RSMGeometry,
    build_hxrd,
    calculate_q,
)

from ._synthetic_hdf5 import make_synthetic_scan_h5


def _model(
    *,
    sample_axes=(RotationAxis("sample", "z-"),),
    detector_axes=(RotationAxis("detector", "z-"),),
    sample_orientation="det",
    primary_beam=(0.0, 1.0, 0.0),
    inplane=(1.0, 0.0, 0.0),
    surface=(0.0, 0.0, 1.0),
):
    return RSMGeometry(
        sample_axes=sample_axes,
        detector_axes=detector_axes,
        primary_beam_direction=primary_beam,
        inplane_reference_direction=inplane,
        sample_surface_normal_direction=surface,
        energy_eV=10000.0,
        ub_matrix=np.eye(3),
        detector=DetectorModel(
            "x+",
            "z+",
            (1.0, 2.0),
            (3, 5),
            (1.0, 1.0),
            500.0,
            (0, 3, 0, 5),
        ),
        sample_orientation=sample_orientation,
    )


def test_legacy_six_axis_math_remains_bit_identical(tmp_path):
    """The shared builder must not perturb the legacy xrayutilities call."""
    path = str(tmp_path / "legacy_geometry.h5")
    make_synthetic_scan_h5(path, n_frames=4, shape=(3, 5))

    sample_directions = ["x+", "z-", "y+", "z-"]
    detector_directions = ["x+", "z-"]
    sample_angles = [
        np.linspace(0.0, 10.0, 4),
        np.full(4, 2.0),
        np.full(4, 3.0),
        np.full(4, 4.0),
    ]
    detector_angles = [np.full(4, 20.0), np.full(4, 25.0)]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "r+") as h5_file:
        hkl = h5_file["entry/data/metadata/HKL"]
        del hkl["SAMPLE_CIRCLE_AXIS_1/DIRECTION_AXIS"]
        hkl["SAMPLE_CIRCLE_AXIS_1"].create_dataset(
            "DIRECTION_AXIS", data=np.array(["x+"], dtype=object), dtype=string_dtype
        )
        for number, (direction, positions) in enumerate(
            zip(sample_directions[1:], sample_angles[1:]), start=2
        ):
            axis = hkl.create_group(f"SAMPLE_CIRCLE_AXIS_{number}")
            axis.create_dataset(
                "DIRECTION_AXIS",
                data=np.array([direction], dtype=object),
                dtype=string_dtype,
            )
            axis.create_dataset("POSITION", data=positions)

        del hkl["DETECTOR_CIRCLE_AXIS_1/DIRECTION_AXIS"]
        hkl["DETECTOR_CIRCLE_AXIS_1"].create_dataset(
            "DIRECTION_AXIS", data=np.array(["x+"], dtype=object), dtype=string_dtype
        )
        detector_axis_2 = hkl.create_group("DETECTOR_CIRCLE_AXIS_2")
        detector_axis_2.create_dataset(
            "DIRECTION_AXIS", data=np.array(["z-"], dtype=object), dtype=string_dtype
        )
        detector_axis_2.create_dataset("POSITION", data=detector_angles[1])

    converter = RSMConverter()
    with h5py.File(path, "r") as h5_file:
        geometry = converter.build_file_geometry(h5_file)
        actual = converter.q_for_frames(geometry, h5_file, 0, 4)

    qconv = xu.experiment.QConversion(
        sample_directions,
        detector_directions,
        [0.0, 1.0, 0.0],
    )
    reference = xu.HXRD(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        en=10000.0,
        qconv=qconv,
    )
    reference.Ang2Q.init_area(
        "x+",
        "z+",
        cch1=1,
        cch2=2,
        Nch1=3,
        Nch2=5,
        pwidth1=1.0,
        pwidth2=1.0,
        distance=500.0,
        roi=[0, 3, 0, 5],
    )
    expected = reference.Ang2Q.area(
        *sample_angles,
        *detector_angles,
        UB=np.eye(3),
    )

    for actual_axis, expected_axis in zip(actual, expected):
        np.testing.assert_array_equal(actual_axis, expected_axis)


def test_shared_builder_is_bit_identical_to_direct_xrayutilities():
    model = _model()
    sample_angle = np.linspace(0.0, 10.0, 4)
    detector_angle = np.full(4, 20.0)

    actual = calculate_q(build_hxrd(model), [sample_angle], [detector_angle])

    qconv = xu.experiment.QConversion(["z-"], ["z-"], [0.0, 1.0, 0.0])
    reference = xu.HXRD(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        en=10000.0,
        qconv=qconv,
        sampleor="det",
    )
    reference.Ang2Q.init_area(
        "x+",
        "z+",
        cch1=1.0,
        cch2=2.0,
        Nch1=3,
        Nch2=5,
        pwidth1=1.0,
        pwidth2=1.0,
        distance=500.0,
        roi=[0, 3, 0, 5],
    )
    expected = reference.Ang2Q.area(
        sample_angle,
        detector_angle,
        UB=np.eye(3),
        deg=True,
    )

    for actual_axis, expected_axis in zip(actual, expected):
        np.testing.assert_array_equal(actual_axis, expected_axis)


@pytest.mark.parametrize(
    "detector_axes",
    [(), (RotationAxis("detector", "y+"),)],
)
def test_det_orientation_rejects_geometry_without_usable_detector_axis(
    detector_axes,
):
    with pytest.raises(ValueError, match="SAMPLE_ORIENTATION='det'"):
        _model(detector_axes=detector_axes)


def test_det_orientation_ignores_one_innermost_beam_axis():
    model = _model(
        detector_axes=(
            RotationAxis("detector", "z-"),
            RotationAxis("detector", "y+"),
        )
    )

    built = build_hxrd(model)

    assert built.model.sample_orientation == "det"


def test_sam_orientation_requires_a_sample_axis():
    with pytest.raises(ValueError, match="requires a sample rotation axis"):
        _model(sample_axes=(), sample_orientation="sam")


def test_sam_orientation_rejects_beam_parallel_innermost_axis():
    with pytest.raises(ValueError, match="innermost sample axis"):
        _model(
            sample_axes=(RotationAxis("sample", "y+"),),
            sample_orientation="sam",
        )


def test_sam_orientation_warns_that_innermost_axis_must_be_azimuth():
    with pytest.warns(UserWarning, match="azimuth motor"):
        model = _model(sample_orientation="sam")

    assert model.sample_orientation == "sam"


def test_explicit_orientation_rejects_primary_beam_parallel_direction():
    with pytest.raises(ValueError, match="silently substitute"):
        _model(sample_orientation="y+")


def test_inplane_and_surface_directions_must_be_perpendicular():
    with pytest.raises(ValueError, match="perpendicular"):
        _model(inplane=(1.0, 0.0, 1.0))


def test_kappa_is_sample_only_and_roles_cannot_cross_lists():
    kappa = RotationAxis("sample", "k+")
    assert kappa.direction == "k+"

    with pytest.raises(ValueError, match="Invalid detector rotation direction"):
        RotationAxis("detector", "k+")
    with pytest.raises(ValueError, match="sample_axes"):
        _model(sample_axes=(RotationAxis("detector", "z-"),))
