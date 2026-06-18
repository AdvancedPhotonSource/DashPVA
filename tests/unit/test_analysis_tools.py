"""Tests for AnalysisTools — feature series/stats/correlation/anomaly, on-demand
frame computation, budgets, and the vision toggle."""

from __future__ import annotations

import types
from collections import deque
from unittest.mock import patch

import numpy as np
import pytest

from dashpva.analysis.tools.analysis_tools import AnalysisTools


def _settings(chat_tools=None, session=None):
    return types.SimpleNamespace(
        CHAT_TOOLS=chat_tools or {'HISTORY_MAX_POINTS': 500,
                                  'FRAME_TOOL_MAX_CALLS_PER_TURN': 20,
                                  'VLM_TOOL_MAX_CALLS_PER_TURN': 3,
                                  'VLM_TOOL_MODEL': 'claudesonnet46'},
        SESSION_ANALYSIS=session or {},
    )


def _fv(frame_id, ts, **frame_vals):
    return {'frame_id': frame_id, 'timestamp': ts,
            'n_blobs': frame_vals.pop('n_blobs', 0),
            'blobs': frame_vals.pop('blobs', []),
            'frame': frame_vals}


class _Reader:
    def __init__(self, fvs=None, images=None, frame_ids=None,
                 shape=(0, 0), blob_dets=None):
        self.feature_vector_cache = fvs or []
        self.cached_images = deque(images) if images is not None else None
        self.cached_frame_ids = deque(frame_ids) if frame_ids is not None else None
        self.cached_timestamps = deque(
            [float(i) for i in range(len(frame_ids))]) if frame_ids else None
        self.blob_detections_cache = blob_dets or []
        self.shape = shape
        self.image_is_transposed = False
        self.CACHING_MODE = 'alignment'
        self.frames_received = len(frame_ids) if frame_ids else 0
        self.config = {}


# ----------------------------------------------------------------------
# Feature time-series / statistics
# ----------------------------------------------------------------------

class TestFeatureSeries:

    def _ramp_reader(self):
        fvs = [_fv(100 + i, float(i), snr=float(i), total_intensity=2.0 * i)
               for i in range(10)]
        return _Reader(fvs=fvs)

    def test_timeseries_alignment(self):
        at = AnalysisTools(self._ramp_reader(), _settings())
        out = at.get_feature_timeseries('snr')
        assert out['n'] == 10
        assert out['frame_ids'][0] == 100
        assert out['values'] == [float(i) for i in range(10)]

    def test_timeseries_window_by_frame(self):
        at = AnalysisTools(self._ramp_reader(), _settings())
        out = at.get_feature_timeseries('snr', start=103, end=105, by='frame')
        assert out['frame_ids'] == [103, 104, 105]

    def test_unknown_feature_lists_available(self):
        at = AnalysisTools(self._ramp_reader(), _settings())
        out = at.get_feature_timeseries('nonsense')
        assert 'error' in out
        assert 'snr' in out['available_features']

    def test_statistics_match_numpy(self):
        at = AnalysisTools(self._ramp_reader(), _settings())
        out = at.get_feature_statistics('snr')
        vals = np.arange(10.0)
        assert out['mean'] == pytest.approx(vals.mean())
        assert out['std'] == pytest.approx(vals.std())
        assert out['trend_slope_per_frame'] == pytest.approx(1.0, abs=1e-9)
        assert out['trend_r2'] == pytest.approx(1.0, abs=1e-9)

    def test_statistics_flags_outlier(self):
        fvs = [_fv(i, float(i), snr=1.0) for i in range(20)]
        fvs[10]['frame']['snr'] = 100.0  # big spike
        at = AnalysisTools(_Reader(fvs=fvs), _settings())
        out = at.get_feature_statistics('snr')
        assert 10 in out['outlier_frame_ids']


class TestCorrelation:

    def test_perfectly_correlated_features(self):
        fvs = [_fv(i, float(i), snr=float(i), total_intensity=3.0 * i + 1)
               for i in range(12)]
        at = AnalysisTools(_Reader(fvs=fvs), _settings())
        out = at.correlate_series('snr', 'total_intensity')
        assert out['n_paired'] == 12
        assert out['pearson_r'] == pytest.approx(1.0, abs=1e-9)
        assert out['spearman_rho'] == pytest.approx(1.0, abs=1e-9)

    def test_constant_series_undefined(self):
        fvs = [_fv(i, float(i), snr=5.0, total_intensity=float(i)) for i in range(6)]
        at = AnalysisTools(_Reader(fvs=fvs), _settings())
        out = at.correlate_series('snr', 'total_intensity')
        assert out['pearson_r'] is None


class TestAnomalies:

    def test_detects_step(self):
        fvs = [_fv(i, float(i), snr=(1.0 if i < 15 else 20.0)) for i in range(30)]
        at = AnalysisTools(_Reader(fvs=fvs), _settings())
        out = at.detect_anomalies('snr')
        assert out['drift']['present'] is True
        assert 10 <= out['change_point']['index'] <= 20


# ----------------------------------------------------------------------
# On-demand frame computation
# ----------------------------------------------------------------------

def _ring_image(n=64, radius=18, width=2.0):
    """A ring with a smooth (Gaussian) radial profile — like a real powder ring,
    so the radial profile has a clear single maximum rather than a flat top."""
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2)
    img = 100.0 * np.exp(-((r - radius) ** 2) / (2 * width ** 2))
    return img.astype(np.float64)


def _gauss_image(n=64, cx=40, cy=24, sigma=3.0, amp=500.0):
    yy, xx = np.mgrid[0:n, 0:n]
    img = amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return img.astype(np.float64)


def _reader_with_image(img, frame_id=500):
    return _Reader(images=[np.ravel(img)], frame_ids=[frame_id], shape=img.shape)


class TestFrameComputation:

    def test_radial_profile_finds_ring(self):
        at = AnalysisTools(_reader_with_image(_ring_image(radius=18)), _settings())
        out = at.compute_radial_profile(500, n_bins=64)
        assert 'rings' in out and out['rings']
        assert abs(out['rings'][0]['r_px'] - 18) <= 3

    def test_fit_peak_recovers_centroid(self):
        at = AnalysisTools(_reader_with_image(_gauss_image(cx=40, cy=24)), _settings())
        out = at.fit_peak(500, x=30, y=14, w=20, h=20)
        assert out['centroid_x'] == pytest.approx(40, abs=1.5)
        assert out['centroid_y'] == pytest.approx(24, abs=1.5)
        assert out['fwhm_x'] is not None

    def test_check_saturation_plateau(self):
        img = np.zeros((32, 32), dtype=np.float64)
        img[10:20, 10:20] = 1000.0   # plateau at max
        at = AnalysisTools(_reader_with_image(img), _settings())
        out = at.check_saturation(500)
        assert out['max_value'] == 1000.0
        assert out['n_saturated'] == 100

    def test_frame_summary_quadrants(self):
        img = np.zeros((40, 40), dtype=np.float64)
        img[:20, :20] = 50.0   # bright top-left quadrant
        at = AnalysisTools(_reader_with_image(img), _settings())
        out = at.get_frame_image_summary(500)
        assert out['quadrant_means']['tl'] > out['quadrant_means']['br']
        assert out['quadrant_asymmetry'] > 0

    def test_missing_frame_returns_error(self):
        at = AnalysisTools(_reader_with_image(_ring_image(), frame_id=500), _settings())
        out = at.compute_radial_profile(999)
        assert 'error' in out
        assert '999' in out['error']

    def test_fit_peak_by_blob_id(self):
        img = _gauss_image(cx=40, cy=24)
        fvs = [{'frame_id': 500, 'timestamp': 0.0, 'n_blobs': 1,
                'blobs': [{'cx': 40, 'cy': 24, 'w': 20, 'h': 20}], 'frame': {}}]
        r = _Reader(fvs=fvs, images=[np.ravel(img)], frame_ids=[500], shape=img.shape)
        at = AnalysisTools(r, _settings())
        out = at.fit_peak(500, blob_id=0)
        assert out['source'] == 'blob 0'
        assert out['centroid_x'] == pytest.approx(40, abs=2.0)


class TestBudgets:

    def test_frame_budget_enforced(self):
        cfg = {'FRAME_TOOL_MAX_CALLS_PER_TURN': 2, 'HISTORY_MAX_POINTS': 500}
        at = AnalysisTools(_reader_with_image(_ring_image()), _settings(chat_tools=cfg))
        assert 'error' not in at.check_saturation(500)
        assert 'error' not in at.check_saturation(500)
        out = at.check_saturation(500)
        assert 'budget' in out['error']
        at.reset_turn_budgets()
        assert 'error' not in at.check_saturation(500)


class TestVisionTool:

    def test_describe_frame_disabled(self):
        at = AnalysisTools(_reader_with_image(_ring_image()), _settings(),
                           vision_enabled=False)
        out = at.describe_frame(500)
        assert 'error' in out
        assert 'disabled' in out['error']

    def test_describe_frame_calls_argo(self):
        sess = {'ARGO_USER': 'tester', 'ARGO_BASE_URL': 'https://x/argoapi'}
        at = AnalysisTools(_reader_with_image(_ring_image()),
                           _settings(session=sess), vision_enabled=True)

        class _Resp:
            status_code = 200
            text = ''

            def json(self):
                return {'choices': [{'message': {'content': 'a bright ring'}}]}

        with patch('dashpva.analysis.tools.analysis_tools.requests.post',
                   return_value=_Resp()) as mocked:
            out = at.describe_frame(500, question='what is this?')
        assert out['description'] == 'a bright ring'
        body = mocked.call_args.kwargs['json']
        assert body['messages'][0]['content'][1]['type'] == 'image_url'


class TestScanInfo:

    def test_list_available_frames(self):
        at = AnalysisTools(_reader_with_image(_ring_image(), frame_id=7), _settings())
        out = at.list_available_frames()
        assert out['n_frames'] == 1
        assert out['frame_id_min'] == 7
        assert out['has_images'] is True