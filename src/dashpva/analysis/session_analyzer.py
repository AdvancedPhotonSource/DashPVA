"""
SessionAnalyzer — assembles a prompt from cached feature vectors, VLM
descriptions, PV history and optional experiment context, then sends it to a
configured LLM backend.

Designed to be called from a worker QThread so the GUI stays responsive
during the (potentially multi-second) LLM round-trip.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

from dashpva.analysis.experiment_context import ExperimentContext
from dashpva.analysis.llm_backend import LLMBackend


def _peak_to_median(frame: dict) -> float:
    """Read the peak-to-median ratio from a frame dict, tolerating the legacy
    ``snr`` key emitted by older recorded scans (the field was renamed because
    peak/median is not a true signal-to-noise ratio)."""
    v = frame.get('peak_to_median', frame.get('snr', 0.0))
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


_SYSTEM_PROMPT = (
    "You are an expert beamline scientist analyzing data from a synchrotron "
    "X-ray experiment.  You will receive quantitative measurements from a "
    "live area detector session.  Based on this data, provide a concise "
    "scientific interpretation:\n"
    "- What was observed (spot patterns, ring patterns, changes over time)\n"
    "- Whether the data quality was good or degraded\n"
    "- Any notable events or anomalies\n"
    "- What this suggests about the sample or experiment conditions\n"
    "Be specific with numbers.  Flag anything that looks like an artifact "
    "or problem."
)


class SessionAnalyzer:
    def __init__(self,
                 pva_reader,
                 backend: LLMBackend,
                 context: Optional[ExperimentContext] = None,
                 n_snapshots: int = 10):
        self.reader = pva_reader
        self.backend = backend
        self.context = context
        self.n_snapshots = max(1, int(n_snapshots))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> str:
        prompt = self._build_prompt()
        return self.backend.complete(prompt=prompt, system=_SYSTEM_PROMPT)

    def build_prompt(self) -> str:
        """Public alias so the UI can preview the prompt without calling the LLM."""
        return self._build_prompt()

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        sections = []

        fv_cache = list(getattr(self.reader, 'feature_vector_cache', []) or [])
        if fv_cache:
            sections.append(self._section_trend_stats(fv_cache))
            sections.append(self._section_snapshots(fv_cache))
            change_events = self._detect_change_events(fv_cache)
            if change_events:
                sections.append("Change events:\n" + "\n".join(f"  - {e}" for e in change_events))
        else:
            sections.append("No feature vectors were cached for this session.")

        vlm = list(getattr(self.reader, 'sampled_descriptions', []) or [])
        if vlm:
            sections.append(self._section_vlm(vlm))

        pv_history = self._section_pv_history()
        if pv_history:
            sections.append(pv_history)

        if self.context is not None:
            ctx_text = self.context.to_text()
            if ctx_text:
                sections.append(ctx_text)

        sections.append(
            "Provide a concise interpretation of this session as described in the "
            "system prompt."
        )
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _section_trend_stats(self, fv_cache: list[dict]) -> str:
        n = len(fv_cache)
        n_blobs = [int(fv.get('n_blobs', 0)) for fv in fv_cache]
        peak_med = [_peak_to_median(fv.get('frame', {})) for fv in fv_cache]
        totals = [float(fv.get('frame', {}).get('total_intensity', 0.0)) for fv in fv_cache]
        active = [float(fv.get('frame', {}).get('active_fraction', 0.0)) for fv in fv_cache]
        bg_tex = [float(fv.get('frame', {}).get('background_texture', 0.0)) for fv in fv_cache]

        ring_count = sum(1 for fv in fv_cache
                         if fv.get('frame', {}).get('radial_profile_peaks'))
        # Most-common radius if rings were ever detected
        radii = []
        for fv in fv_cache:
            peaks = fv.get('frame', {}).get('radial_profile_peaks', []) or []
            for p in peaks:
                if 'r_px' in p:
                    radii.append(int(p['r_px']))
        common_r = ''
        if radii:
            # Round-bin radii by 5 px so jitter doesn't fragment the histogram.
            binned = [r // 5 * 5 for r in radii]
            most = max(set(binned), key=binned.count)
            common_r = f"; most common radius ≈ {most}px"

        lines = [f"Scan statistics over {n} frames:"]
        lines.append(self._fmt_stat("Blob count",         n_blobs,  fmt='{:.1f}'))
        lines.append(self._fmt_stat("Peak/median",        peak_med, fmt='{:.2f}'))
        lines.append(self._fmt_stat("Total intensity",    totals,   fmt='{:.0f}'))
        lines.append(self._fmt_stat("Active fraction",    active,  fmt='{:.4f}'))
        lines.append(self._fmt_stat("Background texture", bg_tex,  fmt='{:.2f}'))
        lines.append(f"  Radial ring peaks:  {ring_count}/{n} frames had ring peaks"
                     f"{common_r}")
        return "\n".join(lines)

    def _section_snapshots(self, fv_cache: list[dict]) -> str:
        n = len(fv_cache)
        if n <= self.n_snapshots:
            indices = list(range(n))
        else:
            # Evenly-spaced indices including first and last frames.
            indices = np.linspace(0, n - 1, self.n_snapshots).astype(int).tolist()
            # Deduplicate while preserving order (linspace may repeat ends).
            seen = set()
            indices = [i for i in indices if not (i in seen or seen.add(i))]

        lines = [f"Snapshots ({len(indices)} of {n} frames):"]
        for idx in indices:
            fv = fv_cache[idx]
            frame = fv.get('frame', {}) or {}
            peaks = frame.get('radial_profile_peaks', []) or []
            peak_str = (', '.join(f"r={p.get('r_px', '?')}px@{p.get('intensity', 0):.0f}"
                                   for p in peaks[:3])
                        if peaks else 'none')
            lines.append(
                f"  Frame {idx}: blobs={fv.get('n_blobs', 0)}, "
                f"peak/med={_peak_to_median(frame):.2f}, "
                f"total={frame.get('total_intensity', 0):,.0f}, "
                f"bg={frame.get('background', 0)}, "
                f"peak=({frame.get('peak_x', '?')},{frame.get('peak_y', '?')}), "
                f"active_frac={frame.get('active_fraction', 0)}, "
                f"bg_texture={frame.get('background_texture', 0)}, "
                f"rings=[{peak_str}]"
            )
        return "\n".join(lines)

    def _detect_change_events(self, fv_cache: list[dict]) -> list[str]:
        events: list[str] = []
        if len(fv_cache) < 2:
            return events

        prev_blobs = int(fv_cache[0].get('n_blobs', 0))
        prev_ptm = _peak_to_median(fv_cache[0].get('frame', {}))
        for i in range(1, len(fv_cache)):
            fv = fv_cache[i]
            n = int(fv.get('n_blobs', 0))
            ptm = _peak_to_median(fv.get('frame', {}))
            if abs(n - prev_blobs) > 2:
                events.append(
                    f"Frame {i}: blob count changed from {prev_blobs} → {n} "
                    f"(possible sample drift or beam intensity change)"
                )
            if prev_ptm > 0 and (prev_ptm - ptm) / prev_ptm > 0.30:
                events.append(
                    f"Frame {i}: peak/median dropped from {prev_ptm:.2f} → {ptm:.2f} "
                    f"({(prev_ptm - ptm) / prev_ptm * 100:.0f}% loss)"
                )
            prev_blobs = n
            prev_ptm = ptm
        return events

    def _section_vlm(self, vlm: list[dict]) -> str:
        if not vlm:
            return ''
        model = vlm[0].get('model', 'vlm')
        first_t = vlm[0].get('timestamp', time.time())
        lines = [f"VLM observations ({model}, sampled periodically):"]
        for entry in vlm:
            ts = entry.get('timestamp', first_t) - first_t
            fid = entry.get('frame_id', '?')
            text = (entry.get('text') or '').strip().replace('\n', ' ')
            if len(text) > 240:
                text = text[:240] + '…'
            lines.append(f"  t={ts:+.0f}s frame {fid}: \"{text}\"")
        return "\n".join(lines)

    def _section_pv_history(self) -> str:
        cached_attrs = getattr(self.reader, 'cached_attributes', None)
        if not cached_attrs:
            return ''
        # cached_attributes can be a deque (alignment/scan mode) or a list of
        # deques (bin mode) — flatten the bin case.
        if isinstance(cached_attrs, (list, tuple)) and cached_attrs and \
                isinstance(cached_attrs[0], deque):
            frames = [d for sub in cached_attrs for d in sub]
        else:
            frames = list(cached_attrs)
        if not frames:
            return ''

        custom_ca = (self.reader.config.get('METADATA', {}) or {}).get('CA', {}) or {}
        if not custom_ca:
            return ''

        lines = ["Process variables over the session:"]
        for friendly, pv_name in custom_ca.items():
            vals = [f.get(pv_name) for f in frames if isinstance(f, dict)]
            numeric = [float(v) for v in vals
                       if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)]
            if not numeric:
                continue
            arr = np.asarray(numeric)
            lines.append(
                f"  {friendly} ({pv_name}): mean={arr.mean():.4g}, "
                f"range=[{arr.min():.4g}, {arr.max():.4g}], n={len(arr)}"
            )
        return "\n".join(lines) if len(lines) > 1 else ''

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_stat(label: str, values: list, fmt: str = '{:.2f}') -> str:
        if not values:
            return f"  {label}: (no data)"
        arr = np.asarray(values, dtype=np.float64)
        return (f"  {label}: mean={fmt.format(arr.mean())}, "
                f"min={fmt.format(arr.min())}, "
                f"max={fmt.format(arr.max())}, "
                f"std={fmt.format(arr.std())}")