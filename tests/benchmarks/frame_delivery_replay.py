# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""Deterministic delivery-contract replay; not a beamline throughput benchmark."""

import argparse
import gc
import json
import weakref

import numpy as np

from dashpva.utils.frame_delivery import FramePacket, LatestFrame


def replay_delivery(frames: int = 1000, consume_every: int = 25) -> dict:
    if frames < 4:
        raise ValueError("frames must be at least 4")
    if consume_every <= 0:
        raise ValueError("consume_every must be positive")

    latest = LatestFrame()
    epoch = latest.epoch
    reconnect_at = frames // 2
    notifications = 0
    consumed = []
    rejected_stale = 0
    first_reference = None

    for sequence in range(frames):
        if sequence == reconnect_at:
            stale = FramePacket.capture(
                max_array_bytes=1024,
                stream_epoch=epoch,
                sequence=sequence,
                unique_id=sequence // 2,
                source_timestamp=float(sequence),
                dequeued_monotonic=float(sequence),
                published_monotonic=float(sequence),
                image=np.full((2, 3), sequence, dtype=np.uint16),
                shape=(2, 3),
                pixel_ordering="F",
                attributes={},
                frame_attributes={},
                fallback_attributes={},
                rsm_attributes={},
                fallback_channels=(),
                geometry_revision="geometry-a",
            )
            epoch = latest.reset()
            rejected_stale += int(not latest.publish(stale))

        packet = FramePacket.capture(
            max_array_bytes=1024,
            stream_epoch=epoch,
            sequence=sequence,
            unique_id=None if sequence % 11 == 0 else sequence // 2,
            source_timestamp=float(sequence),
            dequeued_monotonic=float(sequence),
            published_monotonic=float(sequence),
            image=np.full((2, 3), sequence, dtype=np.uint16, order="F"),
            shape=(2, 3),
            pixel_ordering="F",
            attributes={"motor": sequence},
            frame_attributes={"motor": sequence},
            fallback_attributes={},
            rsm_attributes={},
            fallback_channels=(),
            geometry_revision="geometry-a" if sequence < reconnect_at else "geometry-b",
        )
        if sequence == 0:
            first_reference = weakref.ref(packet)
        notifications += int(latest.publish(packet))
        if (sequence + 1) % consume_every == 0:
            consumed.append(latest.take())

    final = latest.take()
    if not consumed or consumed[-1] is not final:
        consumed.append(final)
    gc.collect()
    consumed_ids = [packet.identity for packet in consumed if packet is not None]
    return {
        "scope": __doc__,
        "frames_published": frames,
        "consume_every": consume_every,
        "notifications": notifications,
        "superseded": latest.superseded,
        "consumed": len(consumed_ids),
        "consumed_identities": consumed_ids,
        "final_identity": final.identity,
        "final_source_id": final.unique_id,
        "final_geometry_revision": final.geometry_revision,
        "stale_epoch_rejections": rejected_stale,
        "first_packet_released": first_reference() is None,
        "retained_packet_limit": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--consume-every", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(replay_delivery(args.frames, args.consume_every), indent=2))


if __name__ == "__main__":
    main()
