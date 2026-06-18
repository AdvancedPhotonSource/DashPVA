"""Investigative analysis tools — the agent's quantitative power tools.

Where :mod:`session_tools` only exposes the *latest* feature vector, these tools
let the model investigate over time and compute new quantities on demand:

  * feature time-series / statistics / anomaly+drift detection over the cached
    per-frame feature vectors,
  * correlation between two features, or a feature and a beamline PV,
  * on-demand computation on a specific cached frame image (radial profile,
    peak fit, ROI stats, saturation, image summary),
  * ``describe_frame`` — send the actual cached image to a vision model so the
    agent can *look* at a frame (gated; not registered when vision is off).

All tools return JSON-serializable dicts and never raise (errors come back as
``{'error': ...}``), matching :meth:`ToolRegistry.call`'s contract. Expensive
frame/vision tools are rate-limited per turn.
"""

from __future__ import annotations

import numpy as np
import requests

from dashpva.analysis.history_store import LiveHistoryStore
from dashpva.analysis.tools.base import BaseTool, tool
from dashpva.utils.stats_analysis import calculate_1d_analysis

# ----------------------------------------------------------------------
# Cache helpers (module-private)
# ----------------------------------------------------------------------

def _flatten(cache) -> list:
    """deque -> list, list[deque] (bin mode) -> concatenated list. Mirrors
    :meth:`LiveHistoryStore._flatten` so cache indexing is consistent."""
    return LiveHistoryStore._flatten(cache)


def _fv_cache(reader) -> list:
    return list(getattr(reader, 'feature_vector_cache', None) or [])


def _get_path(fv: dict, path: str):
    """Pull a dotted feature path from one feature-vector dict.

    Frame-level metrics live under ``fv['frame']`` (e.g. ``snr``, ``com_x``);
    ``n_blobs`` is top-level. Bare names resolve against the frame dict first,
    then the top level. Returns None if absent.
    """
    if not isinstance(fv, dict):
        return None
    if path == 'n_blobs':
        return fv.get('n_blobs')
    if path.startswith('frame.'):
        return (fv.get('frame') or {}).get(path[len('frame.'):])
    frame = fv.get('frame') or {}
    if path in frame:
        return frame[path]
    return fv.get(path)


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.number)) and not isinstance(v, bool)


def _frame_keys(reader) -> list:
    """Available numeric feature names, introspected from the latest vector."""
    cache = _fv_cache(reader)
    if not cache:
        return []
    fv = cache[-1]
    keys = ['n_blobs']
    for k, v in (fv.get('frame') or {}).items():
        if _is_number(v):
            keys.append(k)
    return keys


def _aligned_series(reader, path: str):
    """Return (frame_ids, timestamps, values) for a numeric feature path over
    the whole cache, in arrival order. Non-numeric / missing samples dropped."""
    fids, ts, vals = [], [], []
    for fv in _fv_cache(reader):
        v = _get_path(fv, path)
        if not _is_number(v):
            continue
        fids.append(fv.get('frame_id'))
        ts.append(fv.get('timestamp'))
        vals.append(float(v))
    return fids, ts, vals


def _resolve_frame_index(reader, frame_id: int):
    """Cache index for *frame_id* (exact match), or an error dict if it isn't
    available (rolled off the cache, or never seen)."""
    fids = _flatten(getattr(reader, 'cached_frame_ids', None))
    if not fids:
        return {'error': 'no frames are cached yet'}
    try:
        return fids.index(int(frame_id))
    except (ValueError, TypeError):
        return {'error': f'frame_id {frame_id} is not in the image cache '
                         f'(available range [{fids[0]}, {fids[-1]}], '
                         f'{len(fids)} frames; it may have rolled off)'}


def _image_2d(reader, idx: int):
    """Reconstruct the 2-D image for cache index *idx*. ``cached_images`` holds
    C-order raveled frames (pva_reader caches ``np.ravel(self.image)``); reshape
    with ``reader.shape`` (axes swapped when ``image_is_transposed``)."""
    images = _flatten(getattr(reader, 'cached_images', None))
    if idx < 0 or idx >= len(images) or images[idx] is None:
        return None
    raw = np.asarray(images[idx])
    if raw.ndim == 2:
        return raw.astype(np.float64)
    shape = getattr(reader, 'shape', None)
    if not shape or len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        return None
    h, w = int(shape[0]), int(shape[1])
    if getattr(reader, 'image_is_transposed', False):
        h, w = w, h
    if raw.size != h * w:
        return None
    return raw.reshape((h, w)).astype(np.float64)


def _stride_cap(items: list, max_points: int):
    """Even-stride downsample so a long window is still represented end-to-end."""
    if max_points and len(items) > max_points:
        stride = max(1, len(items) // max_points)
        return items[::stride][:max_points], True
    return items, False


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho without scipy: Pearson on ranks (ties broken by position)."""
    def rank(x):
        order = np.argsort(x, kind='mergesort')
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        return r
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


# ----------------------------------------------------------------------
# Tool class
# ----------------------------------------------------------------------

class AnalysisTools(BaseTool):
    def __init__(self, pva_reader, settings_module, pv_tools=None,
                 vision_enabled: bool = True):
        super().__init__()
        self.reader = pva_reader
        self.settings = settings_module
        self.pv_tools = pv_tools
        self.vision_enabled = bool(vision_enabled)
        self._frame_calls = 0
        self._vlm_calls = 0

    # -- wiring (called by the window, not the LLM) --

    def set_reader(self, pva_reader) -> None:
        self.reader = pva_reader

    def reset_turn_budgets(self) -> None:
        self._frame_calls = 0
        self._vlm_calls = 0

    def _chat_cfg(self) -> dict:
        return getattr(self.settings, 'CHAT_TOOLS', {}) or {}

    def _history_max_points(self) -> int:
        return int(self._chat_cfg().get('HISTORY_MAX_POINTS', 500))

    def _spend_frame(self) -> dict | None:
        cap = int(self._chat_cfg().get('FRAME_TOOL_MAX_CALLS_PER_TURN', 20))
        if self._frame_calls >= cap:
            return {'error': f'per-turn frame-tool budget exhausted ({cap}). '
                             f'Summarize what you found from frames already inspected.'}
        self._frame_calls += 1
        return None

    # ------------------------------------------------------------------
    # Feature time-series / statistics
    # ------------------------------------------------------------------

    @tool(description="Pull a per-frame feature over a window as aligned "
                      "(frame_ids, timestamps, values). Feature paths: 'n_blobs', "
                      "'snr', 'total_intensity', 'com_x', 'com_y', 'background', "
                      "'active_fraction', 'background_texture', 'peak_x', 'peak_y', "
                      "or any frame metric. Use this to see how something evolves.")
    def get_feature_timeseries(self, feature_path: str,
                               start: float | None = None,
                               end: float | None = None,
                               by: str = 'frame') -> dict:
        """Return a feature's time-series.

        Args:
            feature_path: dotted feature name (e.g. 'snr', 'frame.com_x').
            start: window start (frame id if by='frame', else POSIX time).
            end: window end. Omit start/end for the whole cache.
            by: 'frame' or 'time'.
        """
        fids, ts, vals = _aligned_series(self.reader, feature_path)
        if not vals:
            return {'error': f'no numeric data for {feature_path!r}',
                    'available_features': _frame_keys(self.reader)}
        keys = fids if by == 'frame' else ts
        rows = list(zip(fids, ts, vals))
        if start is not None or end is not None:
            lo = -np.inf if start is None else min(start, end if end is not None else start)
            hi = np.inf if end is None else max(start if start is not None else end, end)
            rows = [r for r, k in zip(rows, keys) if k is not None and lo <= k <= hi]
        rows, truncated = _stride_cap(rows, self._history_max_points())
        return {
            'feature_path': feature_path, 'by': by, 'n': len(rows),
            'frame_ids': [r[0] for r in rows],
            'timestamps': [r[1] for r in rows],
            'values': [r[2] for r in rows],
            'truncated': truncated,
        }

    @tool(description="Summary statistics for a feature over the most recent "
                      "`window` frames (or all): mean, std, min, max, median, "
                      "p10, p90, linear trend slope per frame (with r^2), and the "
                      "frame_ids of outliers (robust z > 3).")
    def get_feature_statistics(self, feature_path: str,
                               window: int | None = None) -> dict:
        """Compute statistics for a feature path.

        Args:
            feature_path: dotted feature name (e.g. 'snr').
            window: number of most-recent frames to include (default: all).
        """
        fids, ts, vals = _aligned_series(self.reader, feature_path)
        if not vals:
            return {'error': f'no numeric data for {feature_path!r}',
                    'available_features': _frame_keys(self.reader)}
        if window:
            fids, vals = fids[-int(window):], vals[-int(window):]
        arr = np.asarray(vals, dtype=np.float64)
        idx = np.arange(arr.size, dtype=np.float64)
        out = {
            'feature_path': feature_path, 'n': int(arr.size),
            'mean': float(arr.mean()), 'std': float(arr.std()),
            'min': float(arr.min()), 'max': float(arr.max()),
            'median': float(np.median(arr)),
            'p10': float(np.percentile(arr, 10)),
            'p90': float(np.percentile(arr, 90)),
        }
        if arr.size >= 2 and float(arr.std()) > 0:
            slope, intercept = np.polyfit(idx, arr, 1)
            fit = slope * idx + intercept
            ss_res = float(np.sum((arr - fit) ** 2))
            ss_tot = float(np.sum((arr - arr.mean()) ** 2))
            out['trend_slope_per_frame'] = float(slope)
            out['trend_r2'] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            out['trend_slope_per_frame'] = 0.0
            out['trend_r2'] = 0.0
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) or (float(arr.std()) or 1.0)
        z = 0.6745 * (arr - med) / mad
        mask = np.abs(z) > 3.0
        out['outlier_frame_ids'] = [fids[i] for i in np.where(mask)[0]]
        out['outlier_values'] = [float(arr[i]) for i in np.where(mask)[0]]
        return out

    @tool(description="Correlate two series over a window (Pearson + Spearman). "
                      "Each side is a feature path (e.g. 'snr') OR a PV prefixed "
                      "'pv:' (e.g. 'pv:energy'). Aligns by frame id. Use to relate "
                      "detector response to beamline conditions.")
    def correlate_series(self, a: str, b: str,
                         start: float | None = None,
                         end: float | None = None,
                         by: str = 'frame') -> dict:
        """Correlate two series.

        Args:
            a: feature path or 'pv:<name>'.
            b: feature path or 'pv:<name>'.
            start: window start (frame id if by='frame', else POSIX time).
            end: window end.
            by: 'frame' or 'time'.
        """
        ma = self._series_map(a)
        if 'error' in ma:
            return ma
        mb = self._series_map(b)
        if 'error' in mb:
            return mb
        common = sorted(set(ma['map']) & set(mb['map']))
        if len(common) < 2:
            return {'a': a, 'b': b, 'n_paired': len(common),
                    'error': 'fewer than 2 paired samples after aligning by frame id'}
        xa = np.array([ma['map'][k] for k in common], dtype=np.float64)
        xb = np.array([mb['map'][k] for k in common], dtype=np.float64)
        result = {'a': a, 'b': b, 'by': by, 'n_paired': len(common),
                  'caveat': 'correlation is not causation; small n is unreliable'}
        if xa.std() == 0 or xb.std() == 0:
            result['pearson_r'] = None
            result['spearman_rho'] = None
            result['note'] = 'one series is constant; correlation undefined'
            return result
        result['pearson_r'] = float(np.corrcoef(xa, xb)[0, 1])
        result['spearman_rho'] = _spearman(xa, xb)
        return result

    def _series_map(self, spec: str) -> dict:
        """Resolve a series spec to {'map': {frame_id: value}} (or error)."""
        if spec.startswith('pv:'):
            name = spec[3:].strip()
            if self.pv_tools is None:
                return {'error': "PV correlation unavailable (no PV tools wired)"}
            resolved = self.pv_tools.resolve_and_check(name)
            if resolved is None:
                return {'error': f'{name!r} is not in the chat-tool allowlist'}
            store = self.pv_tools.history_store('live')
            rng = store.get_range(resolved, -1e18, 1e18, by='frame', max_points=None)
            if 'error' in rng:
                return rng
            m = {p['frame_id']: float(p['value']) for p in rng['points']
                 if _is_number(p['value']) and p['frame_id'] is not None}
            return {'map': m}
        fids, _, vals = _aligned_series(self.reader, spec)
        if not vals:
            return {'error': f'no numeric data for feature {spec!r}'}
        return {'map': {f: v for f, v in zip(fids, vals) if f is not None}}

    @tool(description="Detect change points, drift, and spikes in a feature over "
                      "a window. Returns drift (slope/direction), the dominant "
                      "change point (CUSUM), and flagged spike frame_ids.")
    def detect_anomalies(self, feature_path: str,
                         window: int | None = None,
                         sensitivity: float = 3.0) -> dict:
        """Detect anomalies in a feature path.

        Args:
            feature_path: dotted feature name (e.g. 'snr').
            window: number of most-recent frames to include (default: all).
            sensitivity: robust z threshold for spikes (default 3.0).
        """
        fids, ts, vals = _aligned_series(self.reader, feature_path)
        if len(vals) < 3:
            return {'error': f'need >=3 numeric samples for {feature_path!r}, '
                             f'have {len(vals)}'}
        if window:
            fids, vals = fids[-int(window):], vals[-int(window):]
        arr = np.asarray(vals, dtype=np.float64)
        idx = np.arange(arr.size, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) or (float(arr.std()) or 1.0)
        z = 0.6745 * (arr - med) / mad
        spikes = [{'frame_id': fids[i], 'value': float(arr[i]), 'z': float(z[i])}
                  for i in np.where(np.abs(z) > float(sensitivity))[0]]
        slope = float(np.polyfit(idx, arr, 1)[0]) if arr.std() > 0 else 0.0
        noise = mad / 0.6745 if mad else (float(arr.std()) or 1.0)
        drift_present = abs(slope) * arr.size > 2.0 * noise
        cusum = np.cumsum(arr - arr.mean())
        cp = int(np.argmax(np.abs(cusum)))
        return {
            'feature_path': feature_path, 'n': int(arr.size),
            'drift': {'present': bool(drift_present),
                      'slope_per_frame': slope,
                      'direction': ('rising' if slope > 0 else
                                    'falling' if slope < 0 else 'flat')},
            'change_point': {'frame_id': fids[cp], 'index': cp,
                             'score': float(np.abs(cusum)[cp])},
            'spikes': spikes,
            'method': 'rolling-MAD spikes + CUSUM change point (numpy)',
        }

    # ------------------------------------------------------------------
    # On-demand frame computation
    # ------------------------------------------------------------------

    @tool(description="Compute the full radial intensity profile I(r) from the "
                      "image center for a cached frame, plus detected rings with "
                      "their FWHM and integrated intensity. Use to characterize "
                      "powder/diffraction rings beyond the top-5 peaks.")
    def compute_radial_profile(self, frame_id: int, n_bins: int = 256) -> dict:
        """Radial profile for a frame.

        Args:
            frame_id: detector frame id (must still be in the image cache).
            n_bins: number of radial bins (default 256).
        """
        over = self._spend_frame()
        if over:
            return over
        img = self._frame_image(frame_id)
        if isinstance(img, dict):
            return img
        h, w = img.shape
        cy, cx = h / 2.0, w / 2.0
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        r_max = float(min(cx, cy))
        n_bins = max(8, int(n_bins))
        bins = np.linspace(0, r_max, n_bins + 1)
        which = np.clip(np.digitize(r.ravel(), bins) - 1, 0, n_bins - 1)
        flat = img.ravel()
        profile = np.zeros(n_bins)
        counts = np.bincount(which, minlength=n_bins).astype(np.float64)
        sums = np.bincount(which, weights=flat, minlength=n_bins)
        nz = counts > 0
        profile[nz] = sums[nz] / counts[nz]
        centers = 0.5 * (bins[:-1] + bins[1:])
        rings = self._rings_from_profile(centers, profile)
        return {
            'frame_id': int(frame_id), 'n_bins': n_bins,
            'r_px': [float(c) for c in centers],
            'intensity': [round(float(p), 4) for p in profile],
            'rings': rings,
        }

    @staticmethod
    def _rings_from_profile(centers: np.ndarray, profile: np.ndarray) -> list:
        bg = float(np.median(profile))
        thresh = bg * 1.5
        rings = []
        for i in range(1, len(profile) - 1):
            if profile[i] > thresh and profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
                lo = max(0, i - 8)
                hi = min(len(profile), i + 9)
                fit = calculate_1d_analysis(centers[lo:hi], profile[lo:hi])
                fwhm = float(fit['fwhm_value']) if fit and fit.get('fwhm_value') else None
                integ = float(np.sum(profile[lo:hi] - bg))
                rings.append({'r_px': int(round(float(centers[i]))),
                              'intensity': round(float(profile[i]), 4),
                              'fwhm_px': fwhm,
                              'integrated_intensity': round(integ, 4)})
        rings.sort(key=lambda d: d['intensity'], reverse=True)
        return rings[:8]

    @tool(description="Fit a peak in a cached frame to sub-pixel accuracy via 1-D "
                      "marginal analysis: returns centroid, FWHM and amplitude in "
                      "x and y. Give either a blob_id (index into that frame's "
                      "blobs) or an explicit ROI (x,y,w,h).")
    def fit_peak(self, frame_id: int, blob_id: int | None = None,
                 x: int | None = None, y: int | None = None,
                 w: int | None = None, h: int | None = None) -> dict:
        """Fit a peak.

        Args:
            frame_id: detector frame id (must still be in the image cache).
            blob_id: index into the frame's detected blobs (alternative to ROI).
            x: ROI left (with y,w,h).
            y: ROI top.
            w: ROI width.
            h: ROI height.
        """
        over = self._spend_frame()
        if over:
            return over
        img = self._frame_image(frame_id)
        if isinstance(img, dict):
            return img
        ih, iw = img.shape
        if blob_id is not None:
            bbox = self._blob_bbox(frame_id, int(blob_id))
            if isinstance(bbox, dict):
                return bbox
            x0, y0, x1, y1 = bbox
            source = f'blob {blob_id}'
        elif None not in (x, y, w, h):
            x0, y0, x1, y1 = int(x), int(y), int(x) + int(w), int(y) + int(h)
            source = 'roi'
        else:
            return {'error': 'provide either blob_id or all of x,y,w,h'}
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(iw, x1), min(ih, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return {'error': 'ROI too small or out of bounds'}
        crop = img[y0:y1, x0:x1]
        col = crop.sum(axis=0)   # profile along x
        row = crop.sum(axis=1)   # profile along y
        fx = calculate_1d_analysis(np.arange(x0, x1), col)
        fy = calculate_1d_analysis(np.arange(y0, y1), row)
        if not fx or not fy:
            return {'error': 'peak fit failed (flat or empty ROI)'}

        def _fwhm_to_sigma(fwhm):
            return (float(fwhm) / 2.3548) if fwhm else None
        return {
            'frame_id': int(frame_id), 'source': source,
            'centroid_x': float(fx.get('fwhm_center') or fx['com_pos']),
            'centroid_y': float(fy.get('fwhm_center') or fy['com_pos']),
            'com_x': float(fx['com_pos']), 'com_y': float(fy['com_pos']),
            'fwhm_x': (float(fx['fwhm_value']) if fx.get('fwhm_value') else None),
            'fwhm_y': (float(fy['fwhm_value']) if fy.get('fwhm_value') else None),
            'sigma_x': _fwhm_to_sigma(fx.get('fwhm_value')),
            'sigma_y': _fwhm_to_sigma(fy.get('fwhm_value')),
            'amplitude': float(crop.max()),
            'baseline': float(np.median(crop)),
            'method': '1D marginal half-max (calculate_1d_analysis)',
        }

    @tool(description="Statistics (mean/min/max/std/sum, saturated count) inside "
                      "an arbitrary rectangle of a cached frame.")
    def get_roi_statistics(self, frame_id: int, x: int, y: int,
                           w: int, h: int) -> dict:
        """ROI statistics.

        Args:
            frame_id: detector frame id (must still be in the image cache).
            x: ROI left. y: ROI top. w: width. h: height.
        """
        over = self._spend_frame()
        if over:
            return over
        img = self._frame_image(frame_id)
        if isinstance(img, dict):
            return img
        ih, iw = img.shape
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(iw, int(x) + int(w)), min(ih, int(y) + int(h))
        if x1 <= x0 or y1 <= y0:
            return {'error': 'ROI out of bounds'}
        crop = img[y0:y1, x0:x1]
        sat_max = self._saturation_level(img)
        return {
            'frame_id': int(frame_id), 'roi': [x0, y0, x1 - x0, y1 - y0],
            'mean': float(crop.mean()), 'min': float(crop.min()),
            'max': float(crop.max()), 'std': float(crop.std()),
            'sum': float(crop.sum()),
            'n_saturated': int(np.count_nonzero(crop >= sat_max)),
        }

    @tool(description="Detect detector saturation in a cached frame: the max "
                      "value, the inferred saturation level, the saturated-pixel "
                      "count/fraction, and a boolean flag.")
    def check_saturation(self, frame_id: int) -> dict:
        """Saturation check for a frame.

        Args:
            frame_id: detector frame id (must still be in the image cache).
        """
        over = self._spend_frame()
        if over:
            return over
        img = self._frame_image(frame_id)
        if isinstance(img, dict):
            return img
        sat = self._saturation_level(img)
        n_sat = int(np.count_nonzero(img >= sat))
        frac = n_sat / float(img.size)
        return {
            'frame_id': int(frame_id), 'max_value': float(img.max()),
            'saturation_level': float(sat), 'n_saturated': n_sat,
            'saturated_fraction': round(frac, 8),
            'is_saturated': bool(frac > 1e-4),
        }

    @tool(description="Compact summary of a cached frame: intensity histogram, "
                      "per-quadrant mean intensities and an asymmetry score, and "
                      "global max/p99. Cheaper than pulling the whole image.")
    def get_frame_image_summary(self, frame_id: int, n_bins: int = 32) -> dict:
        """Image summary for a frame.

        Args:
            frame_id: detector frame id (must still be in the image cache).
            n_bins: histogram bin count (default 32).
        """
        over = self._spend_frame()
        if over:
            return over
        img = self._frame_image(frame_id)
        if isinstance(img, dict):
            return img
        h, w = img.shape
        my, mx = h // 2, w // 2
        quads = {
            'tl': float(img[:my, :mx].mean()), 'tr': float(img[:my, mx:].mean()),
            'bl': float(img[my:, :mx].mean()), 'br': float(img[my:, mx:].mean()),
        }
        qvals = np.array(list(quads.values()))
        asym = float((qvals.max() - qvals.min()) / (qvals.mean() + 1e-9))
        counts, edges = np.histogram(img.ravel(), bins=max(4, int(n_bins)))
        return {
            'frame_id': int(frame_id),
            'histogram': {'counts': [int(c) for c in counts],
                          'edges': [float(e) for e in edges]},
            'quadrant_means': {k: round(v, 4) for k, v in quads.items()},
            'quadrant_asymmetry': round(asym, 4),
            'max': float(img.max()), 'p99': float(np.percentile(img, 99)),
        }

    @tool(description="Send the actual cached detector image for a frame to a "
                      "vision model and return its description. Use when numeric "
                      "features are ambiguous and you need to SEE the frame. "
                      "Rate-limited per turn.")
    def describe_frame(self, frame_id: int, question: str | None = None) -> dict:
        """Describe a frame with a vision model.

        Args:
            frame_id: detector frame id (must still be in the image cache).
            question: optional specific question about the image.
        """
        if not self.vision_enabled:
            return {'error': 'vision is disabled for this session'}
        cap = int(self._chat_cfg().get('VLM_TOOL_MAX_CALLS_PER_TURN', 3))
        if self._vlm_calls >= cap:
            return {'error': f'per-turn vision budget exhausted ({cap})'}
        img = self._frame_image(frame_id)
        if isinstance(img, dict):
            return img
        self._vlm_calls += 1
        try:
            from dashpva.analysis.vlm_util import encode_frame_for_vlm
            b64 = encode_frame_for_vlm(img)
            text = self._argo_vision(b64, question)
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}'}
        return {
            'frame_id': int(frame_id),
            'model': self._chat_cfg().get('VLM_TOOL_MODEL', 'claudesonnet46'),
            'question': question,
            'description': text,
            'note': 'VLM output is qualitative; verify quantitative claims with '
                    'other tools.',
        }

    # ------------------------------------------------------------------
    # Scan / context
    # ------------------------------------------------------------------

    @tool(description="List the frames currently available for inspection: count "
                      "and frame-id / timestamp range, and whether images and "
                      "feature vectors are present.")
    def list_available_frames(self) -> dict:
        fids = _flatten(getattr(self.reader, 'cached_frame_ids', None))
        ts = _flatten(getattr(self.reader, 'cached_timestamps', None))
        images = _flatten(getattr(self.reader, 'cached_images', None))
        fv = _fv_cache(self.reader)
        return {
            'n_frames': len(fids),
            'frame_id_min': fids[0] if fids else None,
            'frame_id_max': fids[-1] if fids else None,
            't_min': ts[0] if ts else None,
            't_max': ts[-1] if ts else None,
            'has_images': len(images) > 0,
            'n_feature_vectors': len(fv),
            'caching_mode': getattr(self.reader, 'CACHING_MODE', ''),
        }

    @tool(description="Describe the current scan/session context: caching mode, "
                      "frames received, image shape, and the configured metadata "
                      "PVs available for correlation.")
    def get_scan_info(self) -> dict:
        cfg = getattr(self.reader, 'config', {}) or {}
        ca = (cfg.get('METADATA', {}) or {}).get('CA', {}) or {}
        return {
            'caching_mode': getattr(self.reader, 'CACHING_MODE', ''),
            'frames_received': getattr(self.reader, 'frames_received', None),
            'image_shape': list(getattr(self.reader, 'shape', []) or []),
            'metadata_pvs': sorted(ca.keys()),
            'cached_frames': len(_flatten(getattr(self.reader, 'cached_frame_ids', None))),
        }

    # ------------------------------------------------------------------
    # Internal frame helpers
    # ------------------------------------------------------------------

    def _frame_image(self, frame_id: int):
        idx = _resolve_frame_index(self.reader, frame_id)
        if isinstance(idx, dict):
            return idx
        img = _image_2d(self.reader, idx)
        if img is None:
            return {'error': f'frame {frame_id} image is unavailable or could not '
                             f'be reshaped to {getattr(self.reader, "shape", None)}'}
        return img

    def _blob_bbox(self, frame_id: int, blob_id: int):
        """Bbox (x0,y0,x1,y1) for a blob, preferring the feature vector's blobs,
        falling back to blob_detections_cache."""
        idx = _resolve_frame_index(self.reader, frame_id)
        # Try the feature vector (has cx,cy,w,h).
        for fv in _fv_cache(self.reader):
            if fv.get('frame_id') == int(frame_id):
                blobs = fv.get('blobs') or []
                if 0 <= blob_id < len(blobs):
                    b = blobs[blob_id]
                    cx, cy, bw, bh = b['cx'], b['cy'], b['w'], b['h']
                    return (int(cx - bw / 2), int(cy - bh / 2),
                            int(cx + bw / 2), int(cy + bh / 2))
        # Fall back to raw detections [x1,y1,x2,y2,score].
        if not isinstance(idx, dict):
            dets = _flatten(getattr(self.reader, 'blob_detections_cache', None))
            if 0 <= idx < len(dets):
                arr = np.asarray(dets[idx])
                if blob_id < len(arr):
                    x1, y1, x2, y2 = arr[blob_id][:4]
                    return (int(x1), int(y1), int(x2), int(y2))
        return {'error': f'blob {blob_id} not found for frame {frame_id}'}

    @staticmethod
    def _saturation_level(img: np.ndarray) -> float:
        """Best-effort saturation level for a float image whose source dtype was
        lost. Uses the frame max as a proxy (anything at the max is 'saturated')."""
        return float(img.max())

    def _argo_vision(self, b64: str, question: str | None) -> str:
        """One-shot multimodal call to an Argo model via the OpenAI-compatible
        endpoint (independent of the chat backend, so vision works even when
        chatting through a text model)."""
        import os
        cfg = getattr(self.settings, 'SESSION_ANALYSIS', {}) or {}
        user = (cfg.get('ARGO_USER') or os.environ.get('ARGO_USER', '') or '').strip()
        if not user:
            raise RuntimeError('ARGO_USER not configured for vision calls')
        base = (cfg.get('ARGO_BASE_URL')
                or 'https://apps.inside.anl.gov/argoapi').rstrip('/')
        model = self._chat_cfg().get('VLM_TOOL_MODEL', 'claudesonnet46')
        prompt = question or (
            'This is an X-ray area detector frame from a synchrotron experiment. '
            'Describe what you observe: spots, rings, diffuse scatter, halos, '
            'edges, saturation, or artifacts.')
        body = {
            'model': model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                ],
            }],
            'max_tokens': int(cfg.get('ARGO_MAX_TOKENS', 512)),
        }
        resp = requests.post(
            f'{base}/v1/chat/completions',
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {user}'},
            json=body, timeout=int(cfg.get('ARGO_TIMEOUT', 120)))
        if resp.status_code != 200:
            raise RuntimeError(f'Argo vision HTTP {resp.status_code}: {resp.text[:200]}')
        data = resp.json()
        choices = data.get('choices') or []
        if not choices:
            raise RuntimeError(f'Argo vision returned no choices: {data!r}')
        return (choices[0].get('message', {}).get('content') or '').strip()
