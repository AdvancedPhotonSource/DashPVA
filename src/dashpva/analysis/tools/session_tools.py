"""
Session-data tools the LLM can call during chat.

``SessionTools`` exposes read-only slices of the cached frame-analysis data so
the model can ask for just what it needs instead of re-receiving the whole
six-section summarize prompt:

  * get_latest_features      — the most recent feature_vector entry
  * get_recent_change_events — blob/SNR change events over the last N frames
  * get_vlm_samples          — the last N moondream descriptions (if any)

Change-event detection reuses ``SessionAnalyzer._detect_change_events`` so the
chat and the summarize prompt agree on what counts as an event.
"""

from __future__ import annotations

from dashpva.analysis.tools.base import BaseTool, tool


class SessionTools(BaseTool):
    def __init__(self, pva_reader, analyzer):
        super().__init__()
        self.reader = pva_reader
        self.analyzer = analyzer

    def set_reader(self, pva_reader) -> None:
        self.reader = pva_reader

    def set_analyzer(self, analyzer) -> None:
        self.analyzer = analyzer

    @tool(description="Return the most recent per-frame feature vector "
                      "(n_blobs, SNR, total intensity, COM, ring peaks, etc.) "
                      "with its detector frame_id and timestamp.")
    def get_latest_features(self) -> dict:
        fv = list(getattr(self.reader, 'feature_vector_cache', []) or [])
        if not fv:
            return {'error': 'feature_vector_cache is empty — no frames processed yet.'}
        latest = fv[-1]
        return {
            'cache_index': len(fv) - 1,
            'frame_id': latest.get('frame_id'),
            'timestamp': latest.get('timestamp'),
            'features': latest,
        }

    @tool(description="Detect blob-count and SNR change events over the last "
                      "`window` cached frames. Returns human-readable event "
                      "strings (same logic as the session summary).")
    def get_recent_change_events(self, window: int = 50) -> dict:
        fv = list(getattr(self.reader, 'feature_vector_cache', []) or [])
        if len(fv) < 2:
            return {'window': len(fv), 'events': []}
        sliced = fv[-max(2, int(window)):]
        events = []
        if self.analyzer is not None:
            try:
                events = self.analyzer._detect_change_events(sliced)
            except Exception as e:
                return {'window': len(sliced),
                        'error': f'{type(e).__name__}: {e}', 'events': []}
        return {'window': len(sliced), 'events': events}

    @tool(description="Return the last `n` VLM (moondream) text samples taken "
                      "during the session, if the background sampler was running.")
    def get_vlm_samples(self, n: int = 5) -> dict:
        vlm = list(getattr(self.reader, 'sampled_descriptions', []) or [])
        n = max(0, int(n))
        return {'total': len(vlm), 'samples': vlm[-n:] if n else []}
