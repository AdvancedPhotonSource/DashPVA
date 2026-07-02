"""Annotation Manager for Workbench — stores dataset annotations and handles HDF5 I/O."""

from typing import List

import h5py

from dashpva.utils.annotations import (
    Annotation,
    read_annotation_groups,
    write_annotation_groups,
)

# Re-exported so existing imports (e.g. the annotation dock) keep working.
__all__ = ["Annotation", "AnnotationManager"]


class AnnotationManager:
    """Manages annotations for a loaded dataset; saves/loads under /entry/data/annotations/.

    Usage::

        mgr = AnnotationManager(main_window)
        mgr.add(Annotation(text="note", ann_type="dataset", frames=[]))
        mgr.save_to_hdf5("/path/to/file.h5")
    """

    def __init__(self, main_window):
        self.main = main_window
        self._annotations: List[Annotation] = []

    def add(self, ann: Annotation) -> None:
        self._annotations.append(ann)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._annotations):
            self._annotations.pop(index)

    def update(self, index: int, ann: Annotation) -> None:
        if 0 <= index < len(self._annotations):
            self._annotations[index] = ann

    def all_annotations(self) -> List[Annotation]:
        return list(self._annotations)

    def clear(self) -> None:
        self._annotations.clear()

    def notes_for_frame(self, frame_idx: int) -> List[Annotation]:
        """Return annotations that apply to ``frame_idx`` (empty frames = all frames)."""
        return [ann for ann in self._annotations if ann.applies_to_frame(frame_idx)]

    def save_to_hdf5(self, file_path: str) -> None:
        """Write all annotations to /entry/data/annotations/ in append mode."""
        try:
            with h5py.File(file_path, 'a') as h5f:
                entry = h5f.require_group('entry')
                data_grp = entry.require_group('data')
                write_annotation_groups(data_grp, self._annotations)
            self.main.update_status(f"Saved {len(self._annotations)} annotation(s) to HDF5")
        except Exception as e:
            self.main.update_status(f"Error saving annotations: {e}", level='error')

    def load_from_hdf5(self, file_path: str) -> None:
        """Read annotations from /entry/data/annotations/ and replace in-memory store."""
        self.clear()
        try:
            with h5py.File(file_path, 'r') as h5f:
                ann_grp = h5f.get('entry/data/annotations')
                if ann_grp is None:
                    return
                self._annotations.extend(read_annotation_groups(ann_grp))
        except Exception:
            pass
