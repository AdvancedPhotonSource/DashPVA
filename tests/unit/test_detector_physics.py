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

"""Detector calibration, unit normalization, and direct-xrayutilities parity.

Every Q assertion here is checked against a hand-built ``xu.HXRD`` rather than
against a stored value, so the tests fail if the shared builder stops agreeing
with xrayutilities rather than merely changing.
"""

import numpy as np
import pytest
import xrayutilities as xu

from dashpva.utils.rsm_geometry import (
    FRAME_AXIS_DIRECT,
    FRAME_AXIS_SWAPPED,
    DetectorModel,
    RotationAxis,
    RSMGeometry,
    build_hxrd,
    calculate_q,
    detector_model_from_setup,
    detector_setup_from_channels,
    pixel_width_from_size,
)
from dashpva.utils.units import to_deg, to_eV, to_mm

# Deliberately asymmetric: with a square detector a transposed frame axis or a
# swapped ROI pair still produces a correctly-shaped array.
FRAME = (4, 6)
SAMPLE_ANGLES = [11.0]
DETECTOR_ANGLES = [23.0]


def _detector(**overrides) -> DetectorModel:
    params = {
        "pixel_direction_1": "z-",
        "pixel_direction_2": "x-",
        "center_channel": (1.5, 2.5),
        "shape": FRAME,
        "pixel_width": (0.075, 0.075),
        "distance": 400.644,
    }
    params.update(overrides)
    return DetectorModel(**params)


def _geometry(detector: DetectorModel, *, energy_eV: float = 10000.0) -> RSMGeometry:
    return RSMGeometry(
        sample_axes=(RotationAxis("sample", "z-"),),
        detector_axes=(RotationAxis("detector", "z-"),),
        primary_beam_direction=(0.0, 1.0, 0.0),
        inplane_reference_direction=(1.0, 0.0, 0.0),
        sample_surface_normal_direction=(0.0, 0.0, 1.0),
        energy_eV=energy_eV,
        ub_matrix=np.eye(3),
        detector=detector,
    )


def _reference_q(detector: DetectorModel, energy_eV: float = 10000.0):
    """Q from a hand-built xrayutilities geometry, bypassing our builder."""
    qconv = xu.experiment.QConversion(["z-"], ["z-"], (0.0, 1.0, 0.0))
    hxrd = xu.HXRD(
        (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), en=energy_eV, qconv=qconv, sampleor="det"
    )
    hxrd.Ang2Q.init_area(
        detector.pixel_direction_1,
        detector.pixel_direction_2,
        cch1=detector.center_channel[0],
        cch2=detector.center_channel[1],
        Nch1=detector.shape[0],
        Nch2=detector.shape[1],
        pwidth1=detector.pixel_width[0],
        pwidth2=detector.pixel_width[1],
        distance=detector.distance,
        detrot=detector.detrot,
        tilt=detector.tilt,
        tiltazimuth=detector.tiltazimuth,
        roi=list(detector.roi),
        Nav=list(detector.binning),
    )
    return hxrd.Ang2Q.area(*SAMPLE_ANGLES, *DETECTOR_ANGLES, UB=np.eye(3), deg=True)


def _our_q(detector: DetectorModel, energy_eV: float = 10000.0):
    return calculate_q(
        build_hxrd(_geometry(detector, energy_eV=energy_eV)),
        SAMPLE_ANGLES,
        DETECTOR_ANGLES,
    )


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def test_unit_tables_convert_to_canonical_ev_mm_deg():
    assert to_eV(10.0, "keV") == pytest.approx(10000.0)
    assert to_eV(10000.0, "eV") == pytest.approx(10000.0)
    assert to_mm(40.0644, "cm") == pytest.approx(400.644)
    assert to_mm(0.4, "m") == pytest.approx(400.0)
    assert to_mm(75.0, "um") == pytest.approx(0.075)
    assert to_deg(np.pi, "rad") == pytest.approx(180.0)


@pytest.mark.parametrize(
    "value, units",
    [(400.644, "furlong"), (400.644, ""), (400.644, None)],
)
def test_unknown_length_units_raise_rather_than_defaulting(value, units):
    with pytest.raises(ValueError, match="unsupported length units"):
        to_mm(value, units, "DETECTOR_SETUP.DISTANCE")


def test_equivalent_energy_units_produce_identical_q():
    """10 keV and 10000 eV are the same photon; Q must not know the difference."""
    kev = _our_q(_detector(), energy_eV=to_eV(10.0, "keV"))
    ev = _our_q(_detector(), energy_eV=to_eV(10000.0, "eV"))
    for left, right in zip(kev, ev):
        np.testing.assert_array_equal(left, right)


def test_equivalent_length_units_produce_identical_q():
    """A distance of 40.0644 cm is 400.644 mm; Q must be bit-identical."""
    base = {
        "PIXEL_DIRECTION_1": "z-",
        "PIXEL_DIRECTION_2": "x-",
        "CENTER_CHANNEL_PIXEL": [1.5, 2.5],
        "PIXEL_SIZE": [0.075, 0.075],
    }
    in_mm = detector_model_from_setup(
        {**base, "DISTANCE": 400.644, "UNITS": "mm"}, FRAME
    )
    in_cm = detector_model_from_setup(
        {**base, "DISTANCE": 40.0644, "UNITS": "cm", "PIXEL_SIZE": [0.0075, 0.0075]},
        FRAME,
    )
    assert in_cm.distance == pytest.approx(in_mm.distance)
    for left, right in zip(_our_q(in_mm), _our_q(in_cm)):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)


def test_per_field_length_units_override_the_section_default():
    model = detector_model_from_setup(
        {
            "PIXEL_DIRECTION_1": "z-",
            "PIXEL_DIRECTION_2": "x-",
            "CENTER_CHANNEL_PIXEL": [1.5, 2.5],
            "UNITS": "mm",
            "DISTANCE": 40.0644,
            "DISTANCE_UNITS": "cm",
            "PIXEL_SIZE": [75.0, 75.0],
            "PIXEL_SIZE_UNITS": "um",
        },
        FRAME,
    )
    assert model.distance == pytest.approx(400.644)
    assert model.pixel_width == pytest.approx((0.075, 0.075))


# --------------------------------------------------------------------------
# Pixel size, ROI, binning
# --------------------------------------------------------------------------

def test_pixel_width_from_size_allowed_only_full_frame_unbinned():
    assert pixel_width_from_size((0.3, 0.45), (4, 6)) == pytest.approx((0.075, 0.075))
    with pytest.raises(ValueError, match="absorb the binning factor"):
        pixel_width_from_size((0.3, 0.45), (4, 6), binning=(2, 2))
    with pytest.raises(ValueError, match="cropped channel count"):
        pixel_width_from_size((0.3, 0.45), (4, 6), roi=(0, 4, 0, 3))


def test_center_and_pixel_size_are_not_pre_scaled_for_binning():
    """The calibration stays unbinned; xrayutilities applies Nav itself.

    Pre-dividing the centre or pre-multiplying the pixel width here would
    double-apply the correction that _get_detparam_area already performs.
    """
    model = detector_model_from_setup(
        {
            "PIXEL_DIRECTION_1": "z-",
            "PIXEL_DIRECTION_2": "x-",
            "CENTER_CHANNEL_PIXEL": [16.0, 24.0],
            "DISTANCE": 400.644,
            "PIXEL_SIZE": [0.075, 0.075],
            "DETECTOR_SHAPE": [8, 12],
            "BINNING": [2, 2],
        },
        (4, 6),
    )
    assert model.center_channel == (16.0, 24.0)
    assert model.pixel_width == pytest.approx((0.075, 0.075))
    assert model.shape == (8, 12)
    assert model.acquired_shape == (4, 6)


def test_binned_and_cropped_geometry_matches_direct_xrayutilities():
    detector = _detector(
        shape=(8, 12), center_channel=(3.0, 5.0), roi=(2, 6, 0, 12), binning=(2, 2)
    )
    for ours, reference in zip(_our_q(detector), _reference_q(detector)):
        np.testing.assert_allclose(ours, reference, rtol=0, atol=0)


def test_roi_span_must_be_exactly_divisible_by_binning():
    """xrayutilities would ceil the span and report a frame that never existed."""
    with pytest.raises(ValueError, match="not divisible by binning"):
        _detector(shape=(8, 12), roi=(0, 5, 0, 12), binning=(2, 1))


def test_binning_must_be_positive_integers():
    with pytest.raises(ValueError, match="two positive integers"):
        _detector(binning=(0, 1))


def test_require_frame_shape_rejects_a_calibration_data_mismatch():
    detector = _detector(shape=(8, 12), roi=(0, 8, 0, 12), binning=(2, 2))
    assert detector.acquired_shape == (4, 6)
    detector.require_frame_shape((4, 6))
    with pytest.raises(ValueError, match="does not match the calibration"):
        detector.require_frame_shape((8, 12))


# --------------------------------------------------------------------------
# Frame axis order
# --------------------------------------------------------------------------

def test_frame_axis_order_transposes_q_to_match_readout():
    direct = _detector()
    swapped = _detector(frame_axis_order=FRAME_AXIS_SWAPPED)

    assert direct.acquired_shape == FRAME
    assert swapped.acquired_shape == (FRAME[1], FRAME[0])

    for straight, turned in zip(_our_q(direct), _our_q(swapped)):
        np.testing.assert_array_equal(turned, straight.T)


def test_swapped_frame_axis_order_reads_roi_in_detector_coordinates():
    """A transposed readout must not silently transpose the ROI as well."""
    model = detector_model_from_setup(
        {
            "PIXEL_DIRECTION_1": "z-",
            "PIXEL_DIRECTION_2": "x-",
            "CENTER_CHANNEL_PIXEL": [1.5, 2.5],
            "DISTANCE": 400.644,
            "PIXEL_SIZE": [0.075, 0.075],
            "FRAME_AXIS_ORDER": FRAME_AXIS_SWAPPED,
        },
        (6, 4),  # acquired frame is (direction2, direction1)
    )
    assert model.shape == (4, 6)
    assert model.acquired_shape == (6, 4)


# --------------------------------------------------------------------------
# Tilt / detrot
# --------------------------------------------------------------------------

def test_tilt_and_detrot_match_direct_xrayutilities():
    detector = _detector(detrot=1.25, tilt=0.75, tiltazimuth=30.0)
    for ours, reference in zip(_our_q(detector), _reference_q(detector)):
        np.testing.assert_allclose(ours, reference, rtol=0, atol=0)


def test_detrot_does_not_change_the_angle_argument_count():
    """xrayutilities appends the detrot axis internally.

    ``init_area`` adds a rotation about the primary beam to the detector axis
    list when detrot != 0, but supplies its value itself -- so callers keep
    passing only their own circles. If that ever changes, this fails loudly
    rather than silently mis-assigning angles.
    """
    detector = _detector(detrot=2.0)
    qx, _, _ = _our_q(detector)
    assert qx.shape == FRAME


def test_sampleor_det_is_continuous_as_detrot_approaches_zero():
    """'det' ignores rotation about the primary beam, so the limit must be smooth.

    detrot appends a beam-axis rotation as the innermost detector axis, and
    sampleor='det' derives the surface orientation from the innermost detector
    rotation. If it did not skip beam-parallel axes, Q would jump
    discontinuously the moment detrot became nonzero.
    """
    baseline = _our_q(_detector(detrot=0.0))
    for epsilon in (1e-3, 1e-5, 1e-7):
        perturbed = _our_q(_detector(detrot=epsilon))
        for ours, reference in zip(perturbed, baseline):
            np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-3)

    # ...and the limit is approached, not merely bounded.
    def _max_deviation(epsilon):
        return max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(_our_q(_detector(detrot=epsilon)), baseline)
        )

    assert _max_deviation(1e-7) < _max_deviation(1e-3)


# --------------------------------------------------------------------------
# Legacy behavior preservation
# --------------------------------------------------------------------------

def test_persisted_calibration_round_trips_as_numbers_not_reprs(tmp_path):
    """Numeric calibration must survive HDF5 as numbers.

    Commit 687e943 fixed readers that were getting reprs like '[6 6 6]' back.
    Writing PIXEL_SIZE/ROI/BINNING through the plain string writer would
    reintroduce that, and the failure is silent until Q comes out wrong.
    """
    import h5py

    from dashpva.utils.hdf5_writer import HDF5Writer
    from dashpva.utils.rsm_converter import RSMConverter

    path = tmp_path / "calibration.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/data/metadata/HKL/DETECTOR_SETUP")
        literals = {
            "PIXEL_DIRECTION_1": "z-",
            "PIXEL_DIRECTION_2": "x-",
            "CENTER_CHANNEL_PIXEL": [16.0, 24.0],
            "DISTANCE": 400.644,
            "PIXEL_SIZE": [0.075, 0.075],
            "DETECTOR_SHAPE": [8, 12],
            "BINNING": [2, 2],
            "ROI": [0, 8, 0, 12],
            "TILT": 0.75,
            "TILTAZIMUTH": 30.0,
            "UNITS": "mm",
            "ANGLE_UNITS": "deg",
        }
        for key, value in literals.items():
            HDF5Writer._write_typed_literal_dataset(group, key, value)

    with h5py.File(path, "r") as handle:
        stored = handle["entry/data/metadata/HKL/DETECTOR_SETUP"]
        assert stored["PIXEL_SIZE"].dtype.kind == "f"
        assert stored["BINNING"].dtype.kind in ("i", "u")
        assert h5py.check_string_dtype(stored["UNITS"].dtype) is not None
        setup = RSMConverter()._detector_setup_mapping(handle)

    model = detector_model_from_setup(setup, (4, 6))
    assert model.pixel_width == pytest.approx((0.075, 0.075))
    assert model.binning == (2, 2)
    assert model.shape == (8, 12)
    assert model.tilt == pytest.approx(0.75)
    assert model.acquired_shape == (4, 6)


def test_setup_without_any_pr3_keys_reproduces_legacy_full_frame_model():
    """An untouched beamline profile must convert exactly as it did before."""
    legacy = detector_model_from_setup(
        {
            "PIXEL_DIRECTION_1": "z-",
            "PIXEL_DIRECTION_2": "x-",
            "CENTER_CHANNEL_PIXEL": [1.5, 2.5],
            "DISTANCE": 400.644,
            "SIZE": [0.3, 0.45],
        },
        FRAME,
    )
    assert legacy.shape == FRAME
    assert legacy.roi == (0, FRAME[0], 0, FRAME[1])
    assert legacy.binning == (1, 1)
    assert (legacy.detrot, legacy.tilt, legacy.tiltazimuth) == (0.0, 0.0, 0.0)
    assert legacy.pixel_width == pytest.approx((0.075, 0.075))
    for ours, reference in zip(_our_q(legacy), _reference_q(legacy)):
        np.testing.assert_allclose(ours, reference, rtol=0, atol=0)


def test_live_channels_overlay_the_canonical_profile_literals():
    """Calibration the IOC never publishes must still reach live geometry."""
    canonical = {
        "PIXEL_DIRECTION_1": "z-",
        "PIXEL_DIRECTION_2": "x-",
        "CENTER_CHANNEL_PIXEL": [1.5, 2.5],
        "DISTANCE": 400.644,
        "UNITS": "mm",
        "PIXEL_SIZE": [0.0375, 0.0375],
        "BINNING": [2, 2],
        "ROI": [0, 8, 0, 12],
        "DETROT": 1.25,
        "TILT": 0.75,
        "TILTAZIMUTH": 30.0,
        "FRAME_AXIS_ORDER": FRAME_AXIS_SWAPPED,
    }
    section = {
        "PIXEL_DIRECTION_1": "sim:DetectorSetup:PixelDirection1",
        "PIXEL_DIRECTION_2": "sim:DetectorSetup:PixelDirection2",
        "CENTER_CHANNEL_PIXEL": "sim:DetectorSetup:CenterChannelPixel",
        "DISTANCE": "sim:DetectorSetup:Distance",
        "SIZE": "sim:DetectorSetup:Size",
        "UNITS": "sim:DetectorSetup:Units",
    }
    live = {
        "sim:DetectorSetup:PixelDirection1": "z-",
        "sim:DetectorSetup:PixelDirection2": "x-",
        "sim:DetectorSetup:CenterChannelPixel": [3.0, 4.0],
        "sim:DetectorSetup:Distance": 500.0,
        "sim:DetectorSetup:Size": [0.3, 0.45],
        "sim:DetectorSetup:Units": "mm",
    }

    setup = detector_setup_from_channels(section, live, canonical)

    # Live records win where the IOC publishes them...
    assert setup["DISTANCE"] == 500.0
    assert setup["CENTER_CHANNEL_PIXEL"] == [3.0, 4.0]
    # ...and the profile-only calibration survives.
    assert setup["PIXEL_SIZE"] == [0.0375, 0.0375]
    assert setup["ROI"] == [0, 8, 0, 12]
    assert setup["FRAME_AXIS_ORDER"] == FRAME_AXIS_SWAPPED

    model = detector_model_from_setup(setup, (6, 4))
    assert model.pixel_width == pytest.approx((0.0375, 0.0375))
    assert model.binning == (2, 2)
    assert model.tilt == pytest.approx(0.75)
    assert model.detrot == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("frame_axis_order", "acquired_frame"),
    ((FRAME_AXIS_DIRECT, (4, 6)), (FRAME_AXIS_SWAPPED, (6, 4))),
)
def test_live_and_offline_models_share_the_complete_detector_setup(
    frame_axis_order, acquired_frame
):
    canonical = {
        "PIXEL_DIRECTION_1": "z-",
        "PIXEL_DIRECTION_2": "x-",
        "CENTER_CHANNEL_PIXEL": [4.0, 6.0],
        "DISTANCE": 400.644,
        "UNITS": "mm",
        "PIXEL_SIZE": [0.075, 0.075],
        "DETECTOR_SHAPE": [8, 12],
        "BINNING": [2, 2],
        "ROI": [0, 8, 0, 12],
        "DETROT": 1.25,
        "TILT": 0.75,
        "TILTAZIMUTH": 30.0,
        "FRAME_AXIS_ORDER": frame_axis_order,
    }
    section = {
        field: f"sim:DetectorSetup:{field}"
        for field in (
            "PIXEL_DIRECTION_1",
            "PIXEL_DIRECTION_2",
            "CENTER_CHANNEL_PIXEL",
            "DISTANCE",
            "SIZE",
            "UNITS",
        )
    }
    live = {
        section[field]: canonical[field]
        for field in (
            "PIXEL_DIRECTION_1",
            "PIXEL_DIRECTION_2",
            "CENTER_CHANNEL_PIXEL",
            "DISTANCE",
            "UNITS",
        )
    }
    live[section["SIZE"]] = [0.6, 0.9]

    live_model = detector_model_from_setup(
        detector_setup_from_channels(section, live, canonical), acquired_frame
    )
    offline_model = detector_model_from_setup(canonical, acquired_frame)

    assert live_model == offline_model
    for live_q, offline_q in zip(_our_q(live_model), _our_q(offline_model)):
        np.testing.assert_array_equal(live_q, offline_q)


def test_live_resolution_copies_inputs_and_ignores_non_ioc_overlays():
    canonical = {
        "PIXEL_DIRECTION_1": "z-",
        "PIXEL_DIRECTION_2": "x-",
        "CENTER_CHANNEL_PIXEL": [1.5, 2.5],
        "DISTANCE": 400.644,
        "PIXEL_SIZE": [0.075, 0.075],
        "ROI": [0, 4, 0, 6],
        "TILT": 0.75,
        "FUTURE_CALIBRATION": {"coefficients": [1.0, 2.0]},
    }
    live_center = [3.0, 4.0]
    setup = detector_setup_from_channels(
        {
            "CENTER_CHANNEL_PIXEL": "sim:Center",
            "ROI": "sim:UnsupportedRoi",
            "TILT": "sim:UnsupportedTilt",
        },
        {
            "sim:Center": live_center,
            "sim:UnsupportedRoi": [1, 3, 1, 5],
            "sim:UnsupportedTilt": 90.0,
        },
        canonical,
    )

    assert setup["ROI"] == [0, 4, 0, 6]
    assert setup["TILT"] == 0.75
    assert "SIZE" not in setup
    assert setup["FUTURE_CALIBRATION"] == {"coefficients": [1.0, 2.0]}

    setup["CENTER_CHANNEL_PIXEL"][0] = 99.0
    setup["ROI"][0] = 99
    setup["FUTURE_CALIBRATION"]["coefficients"][0] = 99.0
    assert live_center == [3.0, 4.0]
    assert canonical["CENTER_CHANNEL_PIXEL"] == [1.5, 2.5]
    assert canonical["ROI"] == [0, 4, 0, 6]
    assert canonical["FUTURE_CALIBRATION"] == {"coefficients": [1.0, 2.0]}


def test_canonical_pixel_size_survives_a_live_size_record():
    """The IOC republishes a defaulted SIZE; PIXEL_SIZE must still decide."""
    setup = detector_setup_from_channels(
        {"SIZE": "sim:DetectorSetup:Size"},
        {"sim:DetectorSetup:Size": [28.38, 28.38]},
        {
            "PIXEL_DIRECTION_1": "z-",
            "PIXEL_DIRECTION_2": "x-",
            "CENTER_CHANNEL_PIXEL": [1.5, 2.5],
            "DISTANCE": 400.644,
            "PIXEL_SIZE": [0.075, 0.075],
        },
    )

    assert detector_model_from_setup(setup, FRAME).pixel_width == pytest.approx(
        (0.075, 0.075)
    )


def test_channels_alone_still_resolve_without_a_canonical_table():
    setup = detector_setup_from_channels(
        {
            "PIXEL_DIRECTION_1": "sim:D1",
            "PIXEL_DIRECTION_2": "sim:D2",
            "CENTER_CHANNEL_PIXEL": "sim:Center",
            "DISTANCE": "sim:Distance",
            "SIZE": "sim:Size",
        },
        {
            "sim:D1": "z-",
            "sim:D2": "x-",
            "sim:Center": [1.5, 2.5],
            "sim:Distance": 400.644,
            "sim:Size": [0.3, 0.45],
        },
    )

    assert set(setup) == {
        "PIXEL_DIRECTION_1",
        "PIXEL_DIRECTION_2",
        "CENTER_CHANNEL_PIXEL",
        "DISTANCE",
        "SIZE",
    }
