"""Annotation data model and HDF5 (de)serialization shared across the viewer.

Lives in ``utils`` so both the workbench (``AnnotationManager``) and the live-view
``HDF5Writer`` emit and parse an identical ``/entry/data/annotations/`` schema.

An annotation is one of three kinds:

* ``dataset``     — a note attached to one or more frames (``frames``), no geometry.
* ``coordinate``  — one or more points, each a :class:`Placement` with its own frame.
* ``roi``         — one or more ROIs, each a :class:`Placement` with frame + box.

Usage::

    from dashpva.utils.annotations import Annotation, Placement, write_annotation_groups

    ann = Annotation(text="beam drift", ann_type="coordinate", title="drift", placements=[
        Placement(frame=4, x=234, y=65), Placement(frame=5, x=75, y=100),
    ])
    with h5py.File(path, 'a') as h5f:
        write_annotation_groups(h5f.require_group('entry').require_group('data'), [ann])
"""

from dataclasses import dataclass, field
from typing import List, Optional

import h5py
import numpy as np


@dataclass
class Placement:
    """A single point (coordinate) or box (roi) belonging to an annotation, on one frame.

    ``w``/``h``/``roi_name`` are used only for roi placements.
    """

    frame: int
    x: int = 0
    y: int = 0
    w: Optional[int] = None
    h: Optional[int] = None
    roi_name: Optional[str] = None


@dataclass
class Annotation:
    """A single annotation; may hold multiple :class:`Placement` items of one kind.

    Usage::

        ann = Annotation(text="", ann_type="coordinate", title="drift",
                         placements=[Placement(frame=4, x=234, y=65)])
        ann = Annotation(text="", ann_type="roi", placements=[
            Placement(frame=3, x=10, y=20, w=50, h=30, roi_name="ROI 1")])
        ann = Annotation(text="whole-dataset note", ann_type="dataset", frames=[])
    """

    text: str
    ann_type: str  # 'coordinate' | 'roi' | 'dataset'
    title: str = ""
    tags: List[str] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)          # dataset-type note frames
    placements: List[Placement] = field(default_factory=list)  # coordinate/roi geometry
    visible: bool = True   # per-annotation overlay visibility

    def effective_frames(self) -> List[int]:
        """Frames this annotation touches: note frames (dataset) or placement frames."""
        if self.ann_type == 'dataset':
            return list(self.frames)
        return sorted({p.frame for p in self.placements})

    def applies_to_frame(self, frame_idx: int) -> bool:
        """True if this annotation targets ``frame_idx`` (dataset with no frames = all frames)."""
        if self.ann_type == 'dataset':
            return not self.frames or frame_idx in self.frames
        return frame_idx in self.effective_frames()

    def usage_count(self) -> int:
        """Number of placements (coordinates/ROIs) in this annotation."""
        return len(self.placements)

    def frame_label(self) -> str:
        """Frame(s) column text, e.g. ``[2]``, ``[4, 5]`` or ``[all]``."""
        frames = self.effective_frames()
        if not frames:
            return "[all]" if self.ann_type == 'dataset' else "[-]"
        if len(frames) == 1:
            return f"[{frames[0]}]"
        shown = ", ".join(str(f) for f in frames[:6])
        return f"[{shown}{' …' if len(frames) > 6 else ''}]"

    def display_title(self) -> str:
        """Title column text: explicit title, else first tag, else a text preview."""
        if self.title.strip():
            return self.title.strip()
        if self.tags:
            return self.tags[0]
        if self.text.strip():
            return self.text.strip()[:40].replace("\n", " ")
        return "(untitled)"


def write_annotation_groups(data_grp: h5py.Group, annotations: List[Annotation]) -> None:
    """Write ``annotations`` under ``data_grp/annotations`` (NXcollection of NXnote groups).

    Replaces any existing ``annotations`` group. Shared by the workbench manager and the
    live-view HDF5 writer so both sides emit an identical on-disk schema.
    """
    dt_str = h5py.string_dtype(encoding='utf-8')
    if 'annotations' in data_grp:
        del data_grp['annotations']
    ann_grp = data_grp.create_group('annotations')
    ann_grp.attrs['NX_class'] = 'NXcollection'

    for i, ann in enumerate(annotations):
        grp = ann_grp.create_group(f'ann_{i}')
        grp.attrs['NX_class'] = 'NXnote'
        grp.create_dataset('data', data=np.array(ann.text, dtype=dt_str))
        grp.create_dataset('type', data=np.array('text/plain', dtype=dt_str))
        grp.create_dataset('frames', data=np.array(ann.effective_frames(), dtype=np.int32))
        grp.create_dataset('tags', data=np.array(ann.tags, dtype=dt_str))
        grp.attrs['ann_type'] = ann.ann_type
        grp.attrs['visible'] = bool(ann.visible)
        grp.attrs['title'] = ann.title

        if ann.placements:
            grp.create_dataset('pl_frame', data=np.array([p.frame for p in ann.placements], dtype=np.int32))
            grp.create_dataset('pl_x', data=np.array([int(p.x) for p in ann.placements], dtype=np.int32))
            grp.create_dataset('pl_y', data=np.array([int(p.y) for p in ann.placements], dtype=np.int32))
            if ann.ann_type == 'roi':
                grp.create_dataset('pl_w', data=np.array([int(p.w or 0) for p in ann.placements], dtype=np.int32))
                grp.create_dataset('pl_h', data=np.array([int(p.h or 0) for p in ann.placements], dtype=np.int32))
                grp.create_dataset('pl_roi', data=np.array([p.roi_name or '' for p in ann.placements], dtype=dt_str))


def _str_list(arr) -> List[str]:
    return [(v.decode('utf-8') if isinstance(v, bytes) else str(v)) for v in arr]


def read_annotation_groups(ann_grp: h5py.Group) -> List[Annotation]:
    """Parse an ``annotations`` group written by :func:`write_annotation_groups`."""
    annotations: List[Annotation] = []
    for key in sorted(ann_grp.keys()):
        grp = ann_grp[key]
        if not isinstance(grp, h5py.Group):
            continue
        try:
            raw = grp['data'][()]
            text = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
            frames = [int(f) for f in grp['frames'][()]]
            tags = [t for t in _str_list(grp['tags'][()] if 'tags' in grp else []) if t]
            ann_type = str(grp.attrs.get('ann_type', 'dataset'))
            visible = bool(grp.attrs['visible']) if 'visible' in grp.attrs else True
            title = str(grp.attrs['title']) if 'title' in grp.attrs else ""

            placements: List[Placement] = []
            if 'pl_frame' in grp:
                pf = [int(v) for v in grp['pl_frame'][()]]
                px = [int(v) for v in grp['pl_x'][()]]
                py = [int(v) for v in grp['pl_y'][()]]
                pw = [int(v) for v in grp['pl_w'][()]] if 'pl_w' in grp else None
                ph = [int(v) for v in grp['pl_h'][()]] if 'pl_h' in grp else None
                pr = _str_list(grp['pl_roi'][()]) if 'pl_roi' in grp else None
                for j in range(len(pf)):
                    placements.append(Placement(
                        frame=pf[j], x=px[j], y=py[j],
                        w=pw[j] if pw is not None else None,
                        h=ph[j] if ph is not None else None,
                        roi_name=(pr[j] or None) if pr is not None else None,
                    ))
            elif ann_type in ('coordinate', 'roi') and 'x' in grp.attrs:
                # Legacy single-position annotation -> one placement per frame
                x = int(grp.attrs['x'])
                y = int(grp.attrs['y']) if 'y' in grp.attrs else 0
                w = int(grp.attrs['w']) if 'w' in grp.attrs else None
                h = int(grp.attrs['h']) if 'h' in grp.attrs else None
                roi_name = str(grp.attrs['roi_name']) if 'roi_name' in grp.attrs else None
                for f in (frames or [0]):
                    placements.append(Placement(frame=f, x=x, y=y, w=w, h=h, roi_name=roi_name))

            annotations.append(Annotation(
                text=text, ann_type=ann_type, title=title, tags=tags,
                frames=frames if ann_type == 'dataset' else [],
                placements=placements, visible=visible,
            ))
        except Exception:
            pass
    return annotations
