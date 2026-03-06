from pathlib import Path
from typing import List, Optional
import shutil

import h5py
import numpy as np
import toml
import settings
from utils.log_manager import get_default_manager

# Use central LogManager rather than local handler configuration
logger = get_default_manager().get_logger(__name__)


def is_numeric(value):
    return isinstance(value, (int, float, bool, np.number))


def is_position_pv(pv: str) -> bool:
    if not isinstance(pv, str):
        return False
    return (":Position" in pv) or (".RBV" in pv) or ("_RBV" in pv)


def _ensure_parent_group(h5_file: h5py.File, full_path: str):
    parent_path = "/".join(full_path.split("/")[:-1])
    if parent_path:
        h5_file.require_group(parent_path)


def _find_dataset_path_by_name(h5_file: h5py.File, base_group_path: str, dataset_name: str):
    """
    Walk under base_group_path and return the full path of the first dataset whose leaf name equals dataset_name.
    """
    if base_group_path not in h5_file:
        return None

    found_path = None

    def visitor(name, obj):
        nonlocal found_path
        if found_path is not None:
            return
        try:
            if isinstance(obj, h5py.Dataset):
                leaf = name.split('/')[-1]
                if leaf == dataset_name:
                    found_path = f"{base_group_path}/{name}" if not name.startswith(base_group_path) else name
        except Exception:
            pass

    h5_file[base_group_path].visititems(visitor)
    return found_path


def resolve_pv_dataset(h5_file: h5py.File, pv: str, base_group: str = "entry/data/metadata"):
    """
    Resolve a PV string to an existing dataset path inside the saved HDF5 file using known locations.
    - Position PVs -> {base_group}/motor_positions/{pv}
    - Other PVs -> {base_group}/{pv}
    - Fallback: search under base_group for a dataset whose leaf name == pv
    Returns (resolved_path, dataset_obj) or (None, None) if not found.
    """
    if not isinstance(pv, str):
        return None, None

    candidates = []
    if is_position_pv(pv):
        candidates.append(f"{base_group}/motor_positions/{pv}")
    candidates.append(f"{base_group}/{pv}")

    for cand in candidates:
        if cand in h5_file and isinstance(h5_file[cand], h5py.Dataset):
            return cand, h5_file[cand]

    found = _find_dataset_path_by_name(h5_file, base_group, pv)
    if found and found in h5_file and isinstance(h5_file[found], h5py.Dataset):
        return found, h5_file[found]

    return None, None


def copy_dataset_like(h5_file: h5py.File, source: h5py.Dataset, target_path: str):
    """
    Copy real array data to target_path, duplicating shape/dtype and attempting to carry storage settings.
    If target exists, delete it first.
    """
    if target_path in h5_file:
        del h5_file[target_path]

    _ensure_parent_group(h5_file, target_path)

    try:
        dset = h5_file.create_dataset_like(target_path, source, shape=source.shape, dtype=source.dtype)
        dset[...] = source[...]
        return True
    except Exception:
        try:
            h5_file.create_dataset(target_path, data=source[()])
            return True
        except Exception:
            return False


def _build_axis_lookup(mapping: dict):
    axis_lookup = {}
    if 'METADATA' in mapping and 'CA' in mapping['METADATA']:
        for axis_label, pv_string in mapping['METADATA']['CA'].items():
            if isinstance(pv_string, str) and ':' in pv_string:
                motor_id = pv_string.split(':')[1].split('.')[0].split('_')[0]
                axis_lookup[motor_id] = axis_label.upper()
    return axis_lookup


def _process_structure(h5_file: h5py.File, current_path: str, mapping_dict: dict, axis_lookup: dict, stats: dict, base_group: str, include: bool):
    for key, value in mapping_dict.items():
        new_path = f"{current_path}/{key}"

        if isinstance(value, dict):
            h5_file.require_group(new_path)
            _process_structure(h5_file, new_path, value, axis_lookup, stats, base_group, include)
            continue

        try:
            # Write NAME labels when possible
            if isinstance(value, str) and ':' in value:
                parts = value.split(':')
                if len(parts) > 1:
                    motor_id = parts[1].split('_')[0].split('.')[0]
                    axis_name = axis_lookup.get(motor_id)
                    parent_path = "/".join(new_path.split('/')[:-1])
                    name_path = f"{parent_path}/NAME"
                    if axis_name and name_path not in h5_file:
                        h5_file.create_dataset(name_path, data=axis_name)
                        stats.setdefault("names", 0)
                        stats["names"] += 1

            if not include:
                # Only build hierarchy and NAME labels
                continue

            if isinstance(value, str):
                resolved_path, source_node = resolve_pv_dataset(h5_file, value, base_group)
                if source_node is not None:
                    # Special handling for UB matrix: store first 9 values as a flat array
                    if new_path.endswith("HKL/SPEC/UB_MATRIX_VALUE"):
                        try:
                            raw = np.asarray(source_node[...]).ravel()
                            if raw.size < 9:
                                stats["warnings"] += 1
                                logger.error(f"UB source at '{resolved_path}' has only {raw.size} elements; writing as-is to '{new_path}'. Downstream may fail.")
                                ub9 = raw
                            else:
                                ub9 = raw[:9]
                            if new_path in h5_file:
                                del h5_file[new_path]
                            _ensure_parent_group(h5_file, new_path)
                            h5_file.create_dataset(new_path, data=ub9)
                            stats["created"] += 1
                            continue
                        except Exception:
                            stats["warnings"] += 1
                            logger.exception(f"Failed to write UB matrix to '{new_path}' from '{resolved_path}'")
                    # Generic copy for other datasets
                    ok = copy_dataset_like(h5_file, source_node, new_path)
                    if ok:
                        stats["created"] += 1
                    else:
                        stats["warnings"] += 1
                    continue
                else:
                    stats["warnings"] += 1
                    continue

            if is_numeric(value):
                if new_path in h5_file:
                    del h5_file[new_path]
                _ensure_parent_group(h5_file, new_path)
                h5_file.create_dataset(new_path, data=value)
                stats["constants"] += 1
                continue

            stats["warnings"] += 1

        except Exception:
            stats["warnings"] += 1
            logger.exception(f"Error processing key '{key}' at path '{new_path}' with value '{value}'")


def _convert_single_file(src_file: Path, toml_path: Path, base_group: str, include: bool, in_place: bool, output_dir: Path, dry_run: bool) -> str:
    mapping = toml.load(str(toml_path))
    axis_lookup = _build_axis_lookup(mapping)

    # Determine destination file path
    if in_place:
        dst = src_file
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir.joinpath(f"{src_file.stem}_meta_update.h5")

    if dry_run:
        return str(dst)

    # If writing to a copy, duplicate file bytes first
    if not in_place:
        shutil.copy2(src_file, dst)

    stats = {"created": 0, "constants": 0, "warnings": 0}

    with h5py.File(str(dst), 'r+') as h5_file:
        # Ensure base group exists
        h5_file.require_group(base_group)
        _process_structure(h5_file, base_group + "/HKL", mapping.get('HKL', mapping), axis_lookup, stats, base_group, include)

    return str(dst)


def convert_files_or_dir(
    toml_path: str,
    hdf5_path: str,
    base_group: str = "entry/data/metadata",
    include: bool = False,
    in_place: bool = False,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    pattern: str = "*.h5",
    dry_run: bool = False,
) -> List[str]:
    """
    Convert metadata structure in HDF5 file(s) per TOML mapping.

    Parameters:
    - toml_path: Path to TOML mapping file containing HKL hierarchy and PV/constant mappings.
    - hdf5_path: Path to a single HDF5 file or a directory containing HDF5 files.
    - base_group: Base group under which metadata resides (default 'entry/data/metadata').
    - include: When False, only build the HKL group hierarchy and NAME labels; do not copy datasets or constants.
               When True, also copy datasets from PVs and write constants according to mapping.
    - in_place: If True, modify files in place. If False, write converted copies to output_dir
                as '<original_stem>_meta_update.h5'. Originals remain untouched.
    - output_dir: Directory to write converted copies when in_place=False (default 'outputs/conversions').
    - recursive: When hdf5_path is a directory, recurse into subdirectories to find files.
    - pattern: Glob pattern for selecting files within a directory (default '*.h5').
    - dry_run: If True, do not perform writes; return the list of planned output file paths.

    Returns:
    - List of output file paths (planned paths in dry_run mode).

    Notes:
    - NAME labels are derived using 'METADATA.CA' section of the TOML mapping, matching motor IDs.
    - UB_MATRIX_VALUE is truncated to the first 9 elements if present when include=True.
    """
    src = Path(hdf5_path)
    toml_p = Path(toml_path)
    # Determine output directory default from settings.OUTPUT_PATH when not provided
    if output_dir is None:
        base_out = Path(getattr(settings, 'OUTPUT_PATH', './outputs')).expanduser()
        out_dir = base_out.joinpath('conversions')
    else:
        out_dir = Path(output_dir)

    files: List[Path] = []
    if src.is_file():
        files = [src]
    elif src.is_dir():
        if recursive:
            files = [p for p in src.rglob(pattern)]
        else:
            files = [p for p in src.glob(pattern)]
    else:
        raise FileNotFoundError(f"Path not found: {hdf5_path}")

    outputs: List[str] = []
    for f in files:
        try:
            out = _convert_single_file(f, toml_p, base_group, include, in_place, out_dir, dry_run)
            outputs.append(out)
        except Exception:
            logger.exception(f"Failed to convert: {f}")
    return outputs
