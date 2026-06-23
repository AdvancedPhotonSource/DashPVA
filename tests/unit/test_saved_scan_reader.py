"""Tests for ``SavedScanReader`` — the saved-scan → reader-shaped bridge.

Builds a tiny synthetic scan ``.h5`` mirroring ``HDF5Writer.h5_save``'s schema,
then asserts (a) the reader reconstructs the expected caches and (b) the
*existing* analysis/PV tools run over it unchanged and return sane dicts.
"""

from __future__ import annotations

import json
import types
from collections import deque

import h5py
import numpy as np
import pytest

from dashpva.agent.saved_scan_reader import SavedScanReader, _LazyFrame
from dashpva.analysis.tools.analysis_tools import AnalysisTools
from dashpva.analysis.tools.pv_tools import PvTools

ENERGY_PV = "6idb1:energy.VAL"
N_FRAMES = 8
RING_RADIUS = 16


def _ring_image(n=64, radius=RING_RADIUS, width=2.0, amp=100.0):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2)
    return (amp * np.exp(-((r - radius) ** 2) / (2 * width ** 2))).astype(np.float64)


def _feature_vector(frame_id, ts, snr, total):
    return {
        "frame_id": frame_id,
        "timestamp": ts,
        "n_blobs": 1,
        "blobs": [{"cx": 32, "cy": 32, "w": 10, "h": 10}],
        "frame": {"snr": snr, "total_intensity": total, "com_x": 32.0, "com_y": 32.0},
    }


@pytest.fixture
def scan_file(tmp_path):
    """Write a synthetic scan .h5 in the HDF5Writer schema; return its path."""
    path = tmp_path / "scan.h5"
    frame_ids = [100 + i for i in range(N_FRAMES)]
    timestamps = [1000.0 + i for i in range(N_FRAMES)]
    snrs = [float(i) for i in range(N_FRAMES)]
    totals = [10.0 * i + 5.0 for i in range(N_FRAMES)]
    energies = [8.0 + 0.1 * i for i in range(N_FRAMES)]
    images = np.array([_ring_image() for _ in range(N_FRAMES)])

    with h5py.File(path, "w") as h5:
        dgrp = h5.create_group("entry/data")
        dgrp.create_dataset("data", data=images)
        dgrp.create_dataset("frame_ids", data=np.asarray(frame_ids, dtype=np.int64))
        dgrp.create_dataset("timestamps", data=np.asarray(timestamps, dtype=np.float64))

        str_dt = h5py.special_dtype(vlen=str)
        fvs = [json.dumps(_feature_vector(frame_ids[i], timestamps[i], snrs[i], totals[i]))
               for i in range(N_FRAMES)]
        dgrp.create_dataset("feature_vectors", data=fvs, dtype=str_dt)

        # Blob detections (n, maxN=2, 5), NaN-padded: one real blob per frame.
        padded = np.full((N_FRAMES, 2, 5), np.nan)
        for i in range(N_FRAMES):
            padded[i, 0] = [27, 27, 37, 37, 0.9]
        dgrp.create_dataset("blob_detections", data=padded)

        ca = dgrp.create_group("metadata/ca")
        ds = ca.create_dataset("energy", data=np.asarray(energies, dtype=np.float64))
        ds.attrs["pv_name"] = ENERGY_PV
        ca.create_dataset("energy__frame_ids", data=np.asarray(frame_ids, dtype=np.int64))
        ca.create_dataset("energy__timestamps", data=np.asarray(timestamps, dtype=np.float64))

    return path


def _settings():
    return types.SimpleNamespace(
        CONFIG={"METADATA": {"CA": {"energy": ENERGY_PV}}},
        METADATA_CA={"energy": ENERGY_PV},
        IOC_PREFIX="6idb1:",
        DETECTOR_PREFIX="6idb1:",
        CHAT_TOOLS={"HISTORY_MAX_POINTS": 500, "FRAME_TOOL_MAX_CALLS_PER_TURN": 20},
        SESSION_ANALYSIS={},
    )


# ----------------------------------------------------------------------
# Reader reconstruction
# ----------------------------------------------------------------------

class TestReaderShape:

    def test_basic_caches(self, scan_file):
        r = SavedScanReader(scan_file, settings=_settings())
        assert r.frames_received == N_FRAMES
        assert r.shape == (64, 64)
        assert r.CACHING_MODE == "saved"
        assert isinstance(r.cached_images, deque)
        assert isinstance(r.cached_frame_ids, deque)
        assert list(r.cached_frame_ids) == [100 + i for i in range(N_FRAMES)]
        assert len(r.feature_vector_cache) == N_FRAMES
        assert r.feature_vector_cache[0]["frame"]["snr"] == 0.0

    def test_images_are_lazy(self, scan_file):
        r = SavedScanReader(scan_file, settings=_settings())
        # Elements are proxies, not arrays — nothing materialized yet.
        first = r.cached_images[0]
        assert isinstance(first, _LazyFrame)
        # Materializes to a raveled 1-D C-order frame matching the shape.
        arr = np.asarray(first)
        assert arr.ndim == 1
        assert arr.size == 64 * 64

    def test_blob_detections_unpadded(self, scan_file):
        r = SavedScanReader(scan_file, settings=_settings())
        assert len(r.blob_detections_cache) == N_FRAMES
        # One real detection per frame; NaN pad row dropped.
        assert r.blob_detections_cache[0].shape == (1, 5)

    def test_ca_keyed_by_pv_name(self, scan_file):
        r = SavedScanReader(scan_file, settings=_settings())
        assert ENERGY_PV in r.cached_ca
        assert len(r.cached_ca[ENERGY_PV]) == N_FRAMES
        assert r.cached_ca_frame_ids[ENERGY_PV][0] == 100

    def test_rejects_non_scan_file(self, tmp_path):
        bad = tmp_path / "bad.h5"
        with h5py.File(bad, "w") as h5:
            h5.create_dataset("something", data=[1, 2, 3])
        with pytest.raises(ValueError):
            SavedScanReader(bad, settings=_settings())


# ----------------------------------------------------------------------
# Existing tools run unchanged over the saved reader
# ----------------------------------------------------------------------

class TestToolsOverSavedScan:

    def test_list_available_frames(self, scan_file):
        at = AnalysisTools(SavedScanReader(scan_file, settings=_settings()), _settings())
        out = at.list_available_frames()
        assert out["n_frames"] == N_FRAMES
        assert out["frame_id_min"] == 100
        assert out["has_images"] is True
        assert out["caching_mode"] == "saved"

    def test_feature_timeseries(self, scan_file):
        at = AnalysisTools(SavedScanReader(scan_file, settings=_settings()), _settings())
        out = at.get_feature_timeseries("snr")
        assert out["n"] == N_FRAMES
        assert out["values"] == [float(i) for i in range(N_FRAMES)]
        assert out["frame_ids"][0] == 100

    def test_radial_profile_finds_ring(self, scan_file):
        at = AnalysisTools(SavedScanReader(scan_file, settings=_settings()), _settings())
        out = at.compute_radial_profile(100, n_bins=64)
        assert out["rings"]
        assert abs(out["rings"][0]["r_px"] - RING_RADIUS) <= 3

    def test_check_saturation(self, scan_file):
        at = AnalysisTools(SavedScanReader(scan_file, settings=_settings()), _settings())
        out = at.check_saturation(103)
        assert out["frame_id"] == 103
        assert out["max_value"] == pytest.approx(100.0, abs=1.0)

    def test_pv_history_over_saved_scan(self, scan_file):
        pv = PvTools(SavedScanReader(scan_file, settings=_settings()), _settings())
        out = pv.read_pv_at_frame("energy", 105, source="live")
        assert out["value"] == pytest.approx(8.5, abs=1e-6)
        assert out["matched_frame_id"] == 105

    def test_correlate_feature_with_pv(self, scan_file):
        s = _settings()
        reader = SavedScanReader(scan_file, settings=s)
        pv = PvTools(reader, s)
        at = AnalysisTools(reader, s, pv_tools=pv)
        # snr ramps 0..7, energy ramps 8.0..8.7 — both monotone → rho == 1.
        out = at.correlate_series("snr", "pv:energy")
        assert out["n_paired"] == N_FRAMES
        assert out["spearman_rho"] == pytest.approx(1.0, abs=1e-9)