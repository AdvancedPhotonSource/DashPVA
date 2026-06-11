#!/usr/bin/env python
"""
Gaussian Blob PVA Simulator
============================
Publishes synthetic frames containing drifting Gaussian blobs over a PVA channel.
Designed for testing HpcBlobTrackingProcessor + the area detector blob tracking overlay.

Usage
-----
  python -m dashpva.consumers.caIOC_servers.gaussian_blob_sim [options]

Pipeline example
----------------
  # Terminal 1 — this simulator
  python -m dashpva.consumers.caIOC_servers.gaussian_blob_sim \
      --channel pvapy:image --fps 5 --rows 512 --cols 512 --n-blobs 4

  # Terminal 2 — blob tracking consumer
  python -m pvapy.cli.hpcConsumer \
      --input-channel  pvapy:image \
      --output-channel sim:Pva1:Image \
      --processor-file src/dashpva/consumers/hpc/analysis/hpc_blob_tracking_consumer.py \
      --processor-class HpcBlobTrackingProcessor \
      --n-consumers 1 --report-period 5

  # Viewer: enter "sim" as the PV prefix.
"""

import argparse
import time

import numpy as np
import pvaccess as pva
from pvapy.utility.adImageUtility import AdImageUtility


# ── Blob state ────────────────────────────────────────────────────────────────

class GaussianBlob:
    """
    A single drifting Gaussian blob with optional random birth/death.

    Position drifts sinusoidally so SORT can build multi-frame tracks.
    Amplitude is randomised each frame with Poisson noise to simulate
    photon-counting detectors.
    """

    def __init__(self, rows: int, cols: int, rng: np.random.Generator,
                 sigma: float = 18.0, amplitude: float = 200.0,
                 drift_speed: float = 1.5):
        self.rows = rows
        self.cols = cols
        self.rng = rng
        self.sigma = sigma
        self.amplitude = amplitude
        self.drift_speed = drift_speed  # pixels per frame

        # Start at a random interior position
        margin = int(3 * sigma)
        self.cx = float(rng.integers(margin, cols - margin))
        self.cy = float(rng.integers(margin, rows - margin))

        # Random drift direction (unit vector)
        angle = rng.uniform(0, 2 * np.pi)
        self.vx = np.cos(angle) * drift_speed
        self.vy = np.sin(angle) * drift_speed

        # Sinusoidal wobble adds organic feel
        self.wobble_phase = rng.uniform(0, 2 * np.pi)
        self.wobble_amp = rng.uniform(3.0, 12.0)
        self.wobble_freq = rng.uniform(0.02, 0.07)  # radians per frame

    def step(self, frame_idx: int) -> None:
        """Advance position by one frame."""
        margin = int(3 * self.sigma)
        wobble = self.wobble_amp * np.sin(self.wobble_freq * frame_idx + self.wobble_phase)

        self.cx += self.vx + wobble * np.cos(self.wobble_phase)
        self.cy += self.vy + wobble * np.sin(self.wobble_phase)

        # Bounce off walls
        if self.cx < margin or self.cx > self.cols - margin:
            self.vx *= -1
            self.cx = np.clip(self.cx, margin, self.cols - margin)
        if self.cy < margin or self.cy > self.rows - margin:
            self.vy *= -1
            self.cy = np.clip(self.cy, margin, self.rows - margin)

    def render(self, canvas: np.ndarray) -> None:
        """Add this blob's Gaussian footprint onto *canvas* in-place."""
        sigma = self.sigma
        # Bounding box to avoid evaluating the full grid for every blob
        r0 = max(0, int(self.cy - 4 * sigma))
        r1 = min(canvas.shape[0], int(self.cy + 4 * sigma) + 1)
        c0 = max(0, int(self.cx - 4 * sigma))
        c1 = min(canvas.shape[1], int(self.cx + 4 * sigma) + 1)

        rows = np.arange(r0, r1, dtype=np.float32)
        cols = np.arange(c0, c1, dtype=np.float32)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')

        amp = self.amplitude * self.rng.poisson(5) / 5.0
        patch = amp * np.exp(
            -((rr - self.cy) ** 2 + (cc - self.cx) ** 2) / (2 * sigma ** 2)
        )
        canvas[r0:r1, c0:c1] += patch


# ── Frame generation ──────────────────────────────────────────────────────────

def build_frame(blobs: list, rows: int, cols: int,
                noise_level: float, dtype: np.dtype,
                frame_idx: int) -> np.ndarray:
    """Render all blobs onto a noisy background and return a 2-D array."""
    canvas = np.zeros((rows, cols), dtype=np.float32)

    # Background shot noise
    if noise_level > 0:
        canvas += np.random.poisson(noise_level, canvas.shape).astype(np.float32)

    for blob in blobs:
        blob.step(frame_idx)
        blob.render(canvas)

    # Clip and convert to target dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        canvas = np.clip(canvas, info.min, info.max)
    return canvas.astype(dtype)


# ── PVA publishing ────────────────────────────────────────────────────────────

def publish_frame(server: pva.PvaServer, channel: str,
                  frame: np.ndarray, frame_id: int) -> None:
    """Wrap *frame* in an NtNdArray and update the PVA server record."""
    pv = AdImageUtility.generateNtNdArray2D(frame_id, frame)
    server.update(channel, pv)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Publish synthetic Gaussian-blob frames over PVA.')
    p.add_argument('--channel',     default='pvapy:image',
                   help='PVA output channel name (default: pvapy:image)')
    p.add_argument('--fps',         type=float, default=5.0,
                   help='Target frame rate in Hz (default: 5)')
    p.add_argument('--rows',        type=int, default=512,
                   help='Image height in pixels (default: 512)')
    p.add_argument('--cols',        type=int, default=512,
                   help='Image width in pixels (default: 512)')
    p.add_argument('--n-blobs',     type=int, default=4,
                   help='Number of blobs in the scene (default: 4)')
    p.add_argument('--sigma',       type=float, default=18.0,
                   help='Gaussian sigma (blob radius) in pixels (default: 18)')
    p.add_argument('--amplitude',   type=float, default=200.0,
                   help='Peak blob intensity (default: 200)')
    p.add_argument('--drift-speed', type=float, default=1.5,
                   help='Blob drift speed in pixels/frame (default: 1.5)')
    p.add_argument('--noise',       type=float, default=5.0,
                   help='Background Poisson noise mean (default: 5, 0 = off)')
    p.add_argument('--dtype',       default='uint8',
                   choices=['uint8', 'uint16', 'float32'],
                   help='Output pixel dtype (default: uint8)')
    p.add_argument('--seed',        type=int, default=None,
                   help='Random seed for reproducible blob positions')
    p.add_argument('--nf',          type=int, default=0,
                   help='Number of frames to publish then stop (0 = run forever)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dtype = np.dtype(args.dtype)

    blobs = [
        GaussianBlob(args.rows, args.cols, rng,
                     sigma=args.sigma,
                     amplitude=args.amplitude,
                     drift_speed=args.drift_speed)
        for _ in range(args.n_blobs)
    ]

    server = pva.PvaServer()
    server.addRecord(args.channel, pva.NtNdArray(), None)
    print(f'Serving Gaussian-blob frames on "{args.channel}"')
    print(f'  {args.rows}×{args.cols} px, dtype={args.dtype}, '
          f'{args.n_blobs} blobs, {args.fps:.1f} fps')
    print('  Press Ctrl-C to stop.\n')

    frame_id = 0
    dt = 1.0 / args.fps if args.fps > 0 else 0.0

    try:
        while args.nf == 0 or frame_id < args.nf:
            t0 = time.monotonic()

            frame = build_frame(blobs, args.rows, args.cols,
                                args.noise, dtype, frame_id)
            publish_frame(server, args.channel, frame, frame_id)

            frame_id += 1
            if frame_id % 50 == 0:
                print(f'  Published {frame_id} frames')

            elapsed = time.monotonic() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print(f'\nStopped after {frame_id} frames.')


if __name__ == '__main__':
    main()
