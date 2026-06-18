"""
PV-access tools the LLM can call during chat.

``PvTools`` exposes read-only EPICS access:

  * live value         — read_pv (caget, with a cached_ca short-circuit)
  * point in history   — read_pv_at_frame / read_pv_at_time
  * window of history  — read_pv_history (capped at HISTORY_MAX_POINTS)
  * window summary     — get_pv_summary (mean/min/max/std/n)
  * discovery          — list_known_pvs

Every name is validated against a prefix allowlist before any access:
``IOC_PREFIX`` + ``DETECTOR_PREFIX`` + user-extensible ``CHAT_TOOLS.EXTRA_PV_PREFIXES``.
Friendly names from ``[METADATA.CA]`` are always allowed and are resolved to
their real PV name automatically.

Historical lookups go through a :class:`~dashpva.analysis.history_store.HistoryStore`:
``source='live'`` reads the live reader caches; ``source='h5'`` reads a saved
scan file the user loaded via :meth:`set_history_file`.
"""

from __future__ import annotations

from dashpva.analysis.history_store import (
    H5HistoryStore,
    LiveHistoryStore,
)
from dashpva.analysis.tools.base import BaseTool, tool


class PvTools(BaseTool):
    def __init__(self, pva_reader, settings_module):
        super().__init__()
        self.reader = pva_reader
        self.settings = settings_module
        self._h5_store: H5HistoryStore | None = None
        # friendly_name -> pv_name, built once from METADATA.CA
        self._friendly_map: dict[str, str] = {
            str(k): str(v)
            for k, v in (getattr(settings_module, 'METADATA_CA', {}) or {}).items()
        }

    # ------------------------------------------------------------------
    # Wiring (called by the window, not the LLM)
    # ------------------------------------------------------------------

    def set_reader(self, pva_reader) -> None:
        self.reader = pva_reader

    def set_history_file(self, path: str | None) -> None:
        """Point ``source='h5'`` lookups at *path* (or clear with None)."""
        self._h5_store = H5HistoryStore(path) if path else None

    def resolve_and_check(self, pv_name: str) -> str | None:
        """Resolve a friendly name and apply the chat-tool allowlist. Returns the
        canonical PV name, or None if it is not allowed. Lets sibling tools
        (e.g. AnalysisTools.correlate_series) reuse the same gate."""
        name = self._resolve(pv_name)
        return name if self._is_allowed(name) else None

    def history_store(self, source: str = 'live'):
        """Expose the configured history store (live or loaded h5) for reuse."""
        return self._store(source)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool(description="List the EPICS PVs the chat is allowed to read: the "
                      "friendly-name map from METADATA.CA, the prefix allowlist, "
                      "and whether a saved history file is loaded. Call this "
                      "first if you don't know what PVs are available.")
    def list_known_pvs(self) -> dict:
        return {
            'friendly_names': dict(self._friendly_map),
            'allowed_prefixes': sorted(self._allowed_prefixes()),
            'history_file_loaded': self._h5_store is not None,
            'h5_pvs': self._h5_store.available_pvs() if self._h5_store else [],
            'note': 'Pass either a friendly name (a key above) or any PV name '
                    'whose prefix is in allowed_prefixes. For history, '
                    "source='live' reads the current session; source='h5' reads "
                    'the loaded file.',
        }

    @tool(description="Read the current live value of a single EPICS PV. "
                      "Returns {name, value, source} or {name, error}.")
    def read_pv(self, pv_name: str) -> dict:
        """Read a PV's current value.

        Args:
            pv_name: EPICS PV name (e.g. '6idb1:m17.RBV') or a friendly name
                from METADATA.CA. Friendly names resolve automatically.
        """
        name = self._resolve(pv_name)
        if not self._is_allowed(name):
            return self._denied(name)

        # Short-circuit: if we already observed this PV this session, return the
        # latest cached value (avoids a network round-trip during chat).
        cached = getattr(self.reader, 'cached_ca', None) or {}
        if name in cached and cached[name]:
            return {
                'name': name, 'value': cached[name][-1],
                'source': 'cached_ca',
                'samples_in_session': len(cached[name]),
            }

        try:
            from epics import caget
            timeout = min(float((self.settings.CHAT_TOOLS or {}).get('TOOL_TIMEOUT_S', 5.0)), 1.0)
            value = caget(name, timeout=timeout)
        except Exception as e:
            return {'name': name, 'error': f'{type(e).__name__}: {e}'}
        if value is None:
            return {'name': name,
                    'error': 'caget returned None (PV disconnected or timed out)'}
        return {'name': name, 'value': value, 'source': 'live_caget'}

    @tool(description="Read a PV's value at a specific detector frame id. "
                      "Returns the nearest sample with matched_frame_id + an "
                      "'exact' flag when the requested frame was dropped.")
    def read_pv_at_frame(self, pv_name: str, frame_id: int, source: str = 'live') -> dict:
        """Read a PV's value at a detector frame.

        Args:
            pv_name: PV name or friendly name.
            frame_id: Detector uniqueId to look up.
            source: 'live' (current session cache) or 'h5' (loaded file).
        """
        name = self._resolve(pv_name)
        if not self._is_allowed(name):
            return self._denied(name)
        store = self._store(source)
        if store is None:
            return self._no_h5()
        return store.get_at_frame(name, int(frame_id))

    @tool(description="Read a PV's value at (nearest to) a POSIX timestamp. "
                      "Returns the nearest sample with an 'exact' flag.")
    def read_pv_at_time(self, pv_name: str, timestamp: float, source: str = 'live') -> dict:
        """Read a PV's value at a timestamp.

        Args:
            pv_name: PV name or friendly name.
            timestamp: POSIX seconds.
            source: 'live' or 'h5'.
        """
        name = self._resolve(pv_name)
        if not self._is_allowed(name):
            return self._denied(name)
        store = self._store(source)
        if store is None:
            return self._no_h5()
        return store.get_at_time(name, float(timestamp))

    @tool(description="Return a PV's samples across a [start, end] window. "
                      "by='frame' interprets start/end as frame ids; by='time' "
                      "as POSIX timestamps. Capped at HISTORY_MAX_POINTS — for "
                      "large windows prefer get_pv_summary.")
    def read_pv_history(self, pv_name: str, start: float, end: float,
                        by: str = 'frame', source: str = 'live') -> dict:
        """Return a PV's history over a window.

        Args:
            pv_name: PV name or friendly name.
            start: Window start (frame id if by='frame', else POSIX seconds).
            end: Window end.
            by: 'frame' or 'time'.
            source: 'live' or 'h5'.
        """
        name = self._resolve(pv_name)
        if not self._is_allowed(name):
            return self._denied(name)
        store = self._store(source)
        if store is None:
            return self._no_h5()
        max_points = int((self.settings.CHAT_TOOLS or {}).get('HISTORY_MAX_POINTS', 500))
        return store.get_range(name, start, end, by=by, max_points=max_points)

    @tool(description="Summarise a PV over a [start, end] window: "
                      "mean/min/max/std/count. Cheaper than read_pv_history for "
                      "large windows. by='frame' or 'time'.")
    def get_pv_summary(self, pv_name: str, start: float, end: float,
                       by: str = 'frame', source: str = 'live') -> dict:
        """Summarise a PV over a window.

        Args:
            pv_name: PV name or friendly name.
            start: Window start (frame id if by='frame', else POSIX seconds).
            end: Window end.
            by: 'frame' or 'time'.
            source: 'live' or 'h5'.
        """
        name = self._resolve(pv_name)
        if not self._is_allowed(name):
            return self._denied(name)
        store = self._store(source)
        if store is None:
            return self._no_h5()
        return store.get_summary(name, start, end, by=by)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(self, pv_name: str) -> str:
        return self._friendly_map.get(pv_name, pv_name)

    def _store(self, source: str):
        if source == 'h5':
            return self._h5_store
        return LiveHistoryStore(self.reader)

    def _allowed_prefixes(self) -> set[str]:
        s = self.settings
        prefixes: set[str] = set()
        ioc = getattr(s, 'IOC_PREFIX', None)
        if ioc:
            prefixes.add(ioc)
        det = getattr(s, 'DETECTOR_PREFIX', None)
        if det:
            prefixes.add(det if det.endswith(':') else det + ':')
        extra = (getattr(s, 'CHAT_TOOLS', {}) or {}).get('EXTRA_PV_PREFIXES') or []
        for p in extra:
            if p:
                prefixes.add(p if p.endswith(':') else p + ':')
        return prefixes

    def _is_allowed(self, pv_name: str) -> bool:
        # Configured friendly-mapped PVs are always allowed.
        if pv_name in self._friendly_map.values():
            return True
        return any(pv_name.startswith(p) for p in self._allowed_prefixes())

    def _denied(self, pv_name: str) -> dict:
        return {
            'name': pv_name,
            'error': f'{pv_name!r} is not in the chat-tool allowlist. '
                     f'Allowed prefixes: {sorted(self._allowed_prefixes())}. '
                     f'Call list_known_pvs to see configured names.',
        }

    def _no_h5(self) -> dict:
        return {'error': "source='h5' requested but no history file is loaded. "
                         "Load one in the Session Analysis window, or use source='live'."}
