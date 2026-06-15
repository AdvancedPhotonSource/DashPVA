"""
Background VLM sampler — periodically grabs the latest detector frame, sends
it to a local vision-language model (moondream by default) via ollama, and
appends the resulting text description to ``pva_reader.sampled_descriptions``.

Runs in a dedicated QThread so the model round-trip never blocks the GUI.
Off by default — controlled via the Frame Features dock checkbox.
"""

from __future__ import annotations

import base64
import io
import threading
import time

import numpy as np
import requests
from PyQt5.QtCore import QThread, pyqtSignal


class BackgroundVlmSampler(QThread):
    status_updated = pyqtSignal(str)

    def __init__(self,
                 pva_reader,
                 interval_s: int = 30,
                 ollama_url: str = 'http://localhost:11434',
                 model: str = 'moondream',
                 timeout_s: int = 60,
                 parent=None):
        super().__init__(parent)
        self.reader = pva_reader
        self._interval_s = max(1, int(interval_s))
        self._ollama_url = ollama_url.rstrip('/')
        self._model = model
        self._timeout_s = int(timeout_s)
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    def set_interval(self, interval_s: int) -> None:
        self._interval_s = max(1, int(interval_s))

    def stop(self) -> None:
        self._stop_event.set()
        # wait() blocks the caller until run() returns; safe to call from the
        # GUI thread because the worker self-terminates on the stop event.
        if self.isRunning():
            self.wait(int((self._timeout_s + 5) * 1000))

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.status_updated.emit(
            f"VLM sampler started ({self._model}, every {self._interval_s}s)"
        )
        # Initial short delay so the first sample fires reasonably soon after
        # the user enables it rather than after a full interval.
        if self._stop_event.wait(timeout=2.0):
            return

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                self.status_updated.emit(f"VLM sample failed: {e}")
            # Use the stop event as the sleep so stop() takes effect immediately.
            if self._stop_event.wait(timeout=self._interval_s):
                break

        self.status_updated.emit("VLM sampler stopped")

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        image = getattr(self.reader, 'image', None)
        if image is None:
            self.status_updated.emit("VLM sampler waiting for first frame…")
            return
        # Copy under the assumption that callbackSuccess may replace the array
        # at any moment; the numpy buffer is the lightweight reference.
        img = np.asarray(image).copy()
        fv = dict(getattr(self.reader, 'feature_vector', {}) or {})
        frame_id = getattr(self.reader, 'frames_received', 0)

        text = self._interpret(img, fv)
        if not text:
            self.status_updated.emit("VLM returned empty response")
            return

        entry = {
            'timestamp': time.time(),
            'frame_id': frame_id,
            'text': text,
            'model': self._model,
        }
        # Append to the reader-side cache so HDF5Writer can persist it.
        self.reader.sampled_descriptions.append(entry)
        preview = text[:80].replace('\n', ' ') + ('…' if len(text) > 80 else '')
        self.status_updated.emit(f"frame {frame_id}: {preview}")

    def _interpret(self, image: np.ndarray, features: dict) -> str:
        # Avoid importing PIL until the user actually enables sampling — keeps
        # the dock cheap to construct on startup.
        from PIL import Image

        # Reduce dimensionality (some PVA frames arrive as (H, W, 1) etc.)
        if image.ndim == 3:
            image = image.mean(axis=2)
        arr = image.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

        img_pil = Image.fromarray(arr, mode='L')
        # moondream uses 378×378 internally; downscale to keep payload + latency low.
        max_side = max(img_pil.size)
        if max_side > 336:
            scale = 336 / max_side
            new_w = max(1, int(img_pil.size[0] * scale))
            new_h = max(1, int(img_pil.size[1] * scale))
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        n = features.get('n_blobs', 0)
        frame_feats = features.get('frame', {}) or {}
        snr = frame_feats.get('snr', 0)
        prompt = (
            f"This is an X-ray area detector frame from a synchrotron experiment. "
            f"Deterministic analysis found {n} bright spots with overall SNR={snr}. "
            f"In one or two sentences, briefly describe what you observe in the "
            f"image (spots, rings, diffuse scatter, halo, edges, artifacts)."
        )

        resp = requests.post(
            f"{self._ollama_url}/api/generate",
            json={
                'model': self._model,
                'prompt': prompt,
                'images': [b64],
                'stream': False,
            },
            timeout=self._timeout_s,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"ollama generate HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        return (data.get('response') or '').strip()