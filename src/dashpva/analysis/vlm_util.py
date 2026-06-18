"""Shared helpers for sending detector frames to a vision-language model.

The same encode path is used by the background sampler
(:class:`~dashpva.analysis.background_vlm_sampler.BackgroundVlmSampler`) and by
the on-demand ``describe_frame`` tool
(:class:`~dashpva.analysis.tools.analysis_tools.AnalysisTools`), so it lives in
one place. Pure function over a numpy image — no Qt, no network.
"""

from __future__ import annotations

import base64
import io

import numpy as np


def encode_frame_for_vlm(image: np.ndarray, max_side: int = 336) -> str:
    """Normalize a detector frame to an 8-bit grayscale JPEG and return it
    base64-encoded (ASCII).

    Handles multi-channel frames (averaged to grayscale) and flat dynamic range
    (returns a black image rather than dividing by zero). Downscales so the
    longest side is at most ``max_side`` px to keep payload + latency low
    (moondream works internally at 378x378).
    """
    from PIL import Image

    if image.ndim == 3:
        image = image.mean(axis=2)
    arr = np.asarray(image, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)

    img_pil = Image.fromarray(arr, mode='L')
    longest = max(img_pil.size)
    if longest > max_side:
        scale = max_side / longest
        new_w = max(1, int(img_pil.size[0] * scale))
        new_h = max(1, int(img_pil.size[1] * scale))
        img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('ascii')
