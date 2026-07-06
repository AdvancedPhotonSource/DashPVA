"""Moving 2D probe-beam PVA simulator for testing the Beam Profiler dock.

Publishes an NTNDArray image on a PVA channel (consumable by the area detector
viewer) containing an anisotropic probe beam that drifts and jitters over time.
Optionally publishes ground-truth scalar PVs so a live fit can be validated
quantitatively.

Shapes are built *separable* in the (optionally rotated) beam frame,
``I = amp * f_x(x') * f_y(y') + bg``, so summing the ROI along one axis recovers
a clean 1D profile of the matching model — exactly what the Beam Profiler fits.

    DashPVA sim probe --shape gaussian --fwhm-x 30 --fwhm-y 15 --drift 40

Run alongside ``DashPVA detector`` (which defaults to the same ``pvapy:image``
channel), open the Beam Profiler dock, and draw the red ROI over the beam.
"""

import argparse
import sys
import time

import numpy as np
import pvaccess as pva
from pvapy.utility.adImageUtility import AdImageUtility

SHAPES = ("gaussian", "laplacian", "lorentzian", "zone-plate")

_FWHM_GAUSS = 2.0 * np.sqrt(2.0 * np.log(2.0))
_FWHM_LAPLACE = 2.0 * np.log(2.0)

# Ground-truth scalar PV suffixes (published when --truth-pvs is set).
_TRUTH_KEYS = ("true_cx", "true_cy", "true_fwhm_x", "true_fwhm_y", "true_amp")


def _fwhm_to_width(fwhm, shape):
    """Convert an axis FWHM to the shape's native 1D width parameter."""
    if shape in ("gaussian", "zone-plate"):
        return fwhm / _FWHM_GAUSS
    if shape == "lorentzian":
        return fwhm / 2.0
    if shape == "laplacian":
        return fwhm / _FWHM_LAPLACE
    raise ValueError(shape)


def _profile_1d(u, width, shape):
    """1D factor in [0, 1], peaking at 1 where u == 0."""
    width = max(float(width), 1e-9)
    if shape == "gaussian":
        return np.exp(-0.5 * (u / width) ** 2)
    if shape == "lorentzian":
        return (width * width) / (u * u + width * width)
    if shape == "laplacian":
        return np.exp(-np.abs(u) / width)
    raise ValueError(shape)


def _make_frame(xg, yg, cx, cy, fwhm_x, fwhm_y, amp, bg, angle_rad, shape):
    """Render one 2D beam frame (float counts, before optional Poisson noise)."""
    dx = xg - cx
    dy = yg - cy
    if angle_rad:
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a
    else:
        xr, yr = dx, dy

    if shape == "zone-plate":
        # Fresnel-like concentric rings; FWHM is not meaningful for this pattern
        # (visual / edge-case test only). Scale radius by the two widths.
        sx = _fwhm_to_width(fwhm_x, shape)
        sy = _fwhm_to_width(fwhm_y, shape)
        r2 = (xr / max(sx, 1e-9)) ** 2 + (yr / max(sy, 1e-9)) ** 2
        img = 0.5 * (1.0 + np.cos(r2)) * np.exp(-r2 / 50.0)
        return amp * img + bg

    wx = _fwhm_to_width(fwhm_x, shape)
    wy = _fwhm_to_width(fwhm_y, shape)
    img = _profile_1d(xr, wx, shape) * _profile_1d(yr, wy, shape)
    return amp * img + bg


def _cast(frame, dtype):
    """Clip negatives and cast to the requested output dtype."""
    frame = np.clip(frame, 0, None)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        frame = np.clip(np.rint(frame), info.min, info.max)
    return frame.astype(dtype)


def main():
    parser = argparse.ArgumentParser(description="DashPVA moving-probe PVA simulator")
    parser.add_argument("-cn", "--channel-name", default="pvapy:image", dest="channel")
    parser.add_argument("-fps", "--frame-rate", type=float, default=10.0, dest="fps")
    parser.add_argument("-dt", "--datatype", default="float32", dest="dt")
    parser.add_argument("-rt", "--runtime", type=float, default=3600.0, dest="runtime")
    parser.add_argument("-rp", "--report-period", type=int, default=100, dest="report_period")
    parser.add_argument("-nx", type=int, default=512, dest="nx")
    parser.add_argument("-ny", type=int, default=512, dest="ny")
    parser.add_argument("-shape", "--shape", choices=SHAPES, default="gaussian", dest="shape")
    parser.add_argument("-fwhmx", "--fwhm-x", type=float, default=30.0, dest="fwhm_x")
    parser.add_argument("-fwhmy", "--fwhm-y", type=float, default=18.0, dest="fwhm_y")
    parser.add_argument("-amp", "--amplitude", type=float, default=3000.0, dest="amp")
    parser.add_argument("-bg", "--background", type=float, default=10.0, dest="bg")
    parser.add_argument("-angle", "--angle", type=float, default=0.0, dest="angle",
                        help="Beam rotation in degrees.")
    parser.add_argument("-drift", "--drift", type=float, default=30.0, dest="drift",
                        help="Center drift amplitude in pixels (0 = stationary).")
    parser.add_argument("-driftp", "--drift-period", type=float, default=8.0, dest="drift_period",
                        help="Center drift period in seconds.")
    parser.add_argument("-jitter", "--jitter", type=float, default=0.05, dest="jitter",
                        help="Fractional per-frame jitter on FWHM and amplitude.")
    parser.add_argument("-noise", "--noise", action="store_true", dest="noise",
                        help="Apply Poisson photon noise.")
    parser.add_argument("-truth", "--truth-pvs", action="store_true", dest="truth_pvs",
                        help="Publish ground-truth scalar PVs (<channel>:true_*).")
    args = parser.parse_args()

    dtype = np.dtype(args.dt)
    yg, xg = np.meshgrid(np.arange(args.ny, dtype=float),
                         np.arange(args.nx, dtype=float), indexing="ij")
    cx0, cy0 = args.nx / 2.0, args.ny / 2.0
    angle_rad = np.deg2rad(args.angle)
    rng = np.random.default_rng()

    server = pva.PvaServer()
    server.addRecord(args.channel, pva.NtNdArray(), None)
    truth_channels = {}
    truth_objs = {}
    if args.truth_pvs:
        for key in _TRUTH_KEYS:
            name = f"{args.channel}:{key}"
            server.addRecord(name, pva.NtScalar(pva.DOUBLE), None)
            truth_channels[key] = name
            truth_objs[key] = pva.NtScalar(pva.DOUBLE)
    server.start()

    print(f"[probe-sim] serving '{args.channel}' "
          f"({args.nx}x{args.ny} {dtype}, {args.shape}, {args.fps} fps)")
    if truth_channels:
        print(f"[probe-sim] ground-truth PVs: {', '.join(truth_channels.values())}")
    print("[probe-sim] Ctrl-C to stop.")

    dt_frame = 1.0 / args.fps if args.fps > 0 else 0.1
    start = time.time()
    next_t = start
    frame_id = 0
    try:
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= args.runtime:
                break

            # Lissajous drift of the beam center + small per-frame jitter.
            phase = 2.0 * np.pi * elapsed / max(args.drift_period, 1e-6)
            cx = cx0 + args.drift * np.sin(phase)
            cy = cy0 + args.drift * np.sin(0.7 * phase + 1.0)
            jx = 1.0 + args.jitter * float(rng.standard_normal())
            jy = 1.0 + args.jitter * float(rng.standard_normal())
            ja = 1.0 + args.jitter * float(rng.standard_normal())
            fwhm_x = max(args.fwhm_x * jx, 1.0)
            fwhm_y = max(args.fwhm_y * jy, 1.0)
            amp = max(args.amp * ja, 0.0)

            frame = _make_frame(xg, yg, cx, cy, fwhm_x, fwhm_y, amp, args.bg,
                                angle_rad, args.shape)
            if args.noise:
                frame = rng.poisson(np.clip(frame, 0, None)).astype(float)
            frame = _cast(frame, dtype)

            ntnda = AdImageUtility.generateNtNdArray2D(frame_id, frame, args.nx, args.ny, dtype)
            ntnda["uniqueId"] = frame_id
            ts = pva.PvTimeStamp(now)
            ntnda["timeStamp"] = ts
            ntnda["dataTimeStamp"] = ts
            server.updateUnchecked(args.channel, ntnda)

            if truth_channels:
                truth_vals = {
                    "true_cx": cx, "true_cy": cy,
                    "true_fwhm_x": fwhm_x, "true_fwhm_y": fwhm_y, "true_amp": amp,
                }
                for key, name in truth_channels.items():
                    obj = truth_objs[key]
                    obj["value"] = float(truth_vals[key])
                    server.updateUnchecked(name, obj)

            frame_id += 1
            if args.report_period > 0 and frame_id % args.report_period == 0:
                print(f"[probe-sim] published {frame_id} frames ({frame_id / elapsed:.1f} fps)")

            next_t += dt_frame
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()  # fell behind; resync
    except KeyboardInterrupt:
        print("\n[probe-sim] stopping.")
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
