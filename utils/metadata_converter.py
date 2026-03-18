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


def _derive_axis_from_pv(pv_string: str, axis_lookup: dict) -> "str | None":
    """
    Derive an axis label from a PV string by extracting the motor ID and looking it up in axis_lookup.

    Uses the same motor-ID extraction as _build_axis_lookup so that, e.g.,
    '6idb1:m17_RBV:Position' → motor_id 'm17' → axis label 'ETA'.
    """
    try:
        if not isinstance(pv_string, str) or ':' not in pv_string:
            return None
        parts = pv_string.split(':')
        if len(parts) < 2:
            return None
        motor_id = parts[1].split('_')[0].split('.')[0]
        return axis_lookup.get(motor_id)
    except Exception:
        return None


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


def _rename_motor_positions_and_link_hkl(h5_file: h5py.File, mapping: dict, base_group: str, axis_lookup: dict = None, force: bool = False):
    """
    Rename PV-named datasets under motor_positions to axis labels and link HKL POSITION to them.

    Axis labels are resolved in priority order:
      1) If {base_group}/HKL/<GROUP>/NAME exists in the HDF5 file, use its string value (uppercased).
      2) Else, if mapping['HKL'][<GROUP>]['NAME'] exists in TOML, use it (uppercased).
      3) Else, derive from the POSITION PV string via motor-ID lookup built from METADATA.CA
         (e.g. '6idb1:m17_RBV:Position' → motor_id 'm17' → 'ETA').
      If none of the three sources yields a label, log a warning and skip that group.

    For each HKL group discovered (union of groups in the TOML and those present in the file):
      - Read its POSITION string from the TOML section for that group.
      - Resolve the dataset path via resolve_pv_dataset (prefers base_group/motor_positions for position PVs).
      - Copy data to {base_group}/motor_positions/{AXIS} using copy_dataset_like.
      - Set units attribute to "deg" on the axis dataset.
      - Delete the original PV-named dataset if it lives under base_group/motor_positions and differs from the axis path.
      - Replace HKL/<GROUP>/POSITION with a SoftLink to the axis dataset. Ensure NAME exists with axis label if missing.
    """


    def _read_name_from_file(group_key: str) -> str | None:
        try:
            name_path = f"{base_group}/HKL/{group_key}/NAME"
            if name_path in h5_file and isinstance(h5_file[name_path], h5py.Dataset):
                ds = h5_file[name_path]
                try:
                    val = ds.asstr()[()] if hasattr(ds, "asstr") else ds[()]
                except Exception:
                    val = None
                if isinstance(val, (bytes, np.bytes_)):
                    try:
                        val = val.decode('utf-8', errors='ignore')
                    except Exception:
                        val = str(val)
                if isinstance(val, (str,)):
                    v = val.strip()
                    return v.upper() if v else None
        except Exception:
            pass
        return None

    def _read_name_from_mapping(group_key: str) -> str | None:
        try:
            if isinstance(mapping, dict):
                hk = mapping.get("HKL", {})
                grp = hk.get(group_key, {}) if isinstance(hk, dict) else {}
                nm = grp.get("NAME")
                if isinstance(nm, str) and nm.strip():
                    return nm.strip().upper()
        except Exception:
            pass
        return None

    # Build the set of HKL group keys from both mapping and the file content
    group_keys = set()
    try:
        if isinstance(mapping, dict):
            hk = mapping.get("HKL", {})
            if isinstance(hk, dict):
                group_keys.update(list(hk.keys()))
    except Exception:
        pass
    try:
        hkl_root = f"{base_group}/HKL"
        if hkl_root in h5_file and isinstance(h5_file[hkl_root], h5py.Group):
            for k in h5_file[hkl_root].keys():
                group_keys.add(k)
    except Exception:
        pass
    if not group_keys:
        return

    hkl_section = mapping.get("HKL", {}) if isinstance(mapping, dict) else {}

    # Derive axis label per group using priority order:
    #   1. NAME dataset in HDF5 file
    #   2. NAME key in TOML mapping
    #   3. Derive from POSITION PV string via motor-ID lookup (handles commented-out NAME fields)
    axis_map = {}
    for group_key in sorted(group_keys):
        label = _read_name_from_file(group_key) or _read_name_from_mapping(group_key)
        if not label and isinstance(axis_lookup, dict) and axis_lookup:
            try:
                pos_pv = hkl_section.get(group_key, {}).get("POSITION") if isinstance(hkl_section, dict) else None
                if isinstance(pos_pv, str):
                    label = _derive_axis_from_pv(pos_pv, axis_lookup)
            except Exception:
                pass
        if label:
            axis_map[group_key] = label
            print(f"  [axis_map] {group_key} -> {label}")
        else:
            print(f"  [axis_map] {group_key} -> SKIPPED (no label resolved)")
            logger.warning(
                f"Skipping HKL group '{group_key}': no NAME found in HDF5, TOML, or derivable from POSITION PV."
            )

    # Build reverse lookup: axis_label -> CA PV string from METADATA.CA
    # e.g. 'MU' -> '6idb1:m28.RBV' so we can find motor_positions/6idb1:m28.RBV
    axis_to_ca_pv = {}
    try:
        ca_section = mapping.get("METADATA", {}).get("CA", {}) if isinstance(mapping, dict) else {}
        for label, pv_string in ca_section.items():
            if isinstance(pv_string, str) and ':' in pv_string:
                axis_to_ca_pv[label.upper()] = pv_string
    except Exception:
        pass

    for group_key, axis_label in axis_map.items():
        print(f"\n[rename] Processing group '{group_key}' -> axis '{axis_label}'")
        group_map = hkl_section.get(group_key)
        if not isinstance(group_map, dict):
            print(f"  SKIP: group not found in TOML mapping")
            logger.warning(f"HKL group '{group_key}' not found in mapping; skipping axis '{axis_label}'.")
            continue

        pv = group_map.get("POSITION")
        if not isinstance(pv, str):
            print(f"  SKIP: no POSITION key in TOML for this group")
            logger.warning(f"No POSITION PV string for HKL group '{group_key}'; skipping axis '{axis_label}'.")
            continue

        print(f"  POSITION PV from TOML: {pv}")
        try:
            resolved_path, source_node = resolve_pv_dataset(h5_file, pv, base_group)
        except Exception:
            logger.exception(f"Error resolving PV '{pv}' for HKL group '{group_key}'.")
            resolved_path, source_node = (None, None)
        print(f"  resolve_pv_dataset -> path={resolved_path}  found={source_node is not None}")

        if source_node is None:
            # Fallback 1: axis-named dataset already exists from a prior conversion run
            fallback_path = f"{base_group}/motor_positions/{axis_label}"
            try:
                if fallback_path in h5_file and isinstance(h5_file[fallback_path], h5py.Dataset):
                    resolved_path = fallback_path
                    source_node = h5_file[fallback_path]
            except Exception:
                pass

        if source_node is None:
            # Fallback 2: find the CA PV-named dataset in motor_positions.
            # The recording system stores datasets by METADATA.CA PV name (e.g. '6idb1:m28.RBV'),
            # which differs from the HKL POSITION PV string (e.g. '6idb1:m28_RBV:Position').
            ca_pv = axis_to_ca_pv.get(axis_label)
            if ca_pv:
                ca_path = f"{base_group}/motor_positions/{ca_pv}"
                try:
                    if ca_path in h5_file and isinstance(h5_file[ca_path], h5py.Dataset):
                        resolved_path = ca_path
                        source_node = h5_file[ca_path]
                except Exception:
                    pass

        if source_node is None:
            logger.warning(f"POSITION PV dataset for '{group_key}' not found (PV='{pv}'); leaving HKL POSITION as-is.")
            continue

        target_axis_path = f"{base_group}/motor_positions/{axis_label}"

        # Copy source data to axis-named dataset (skip if source is already the target)
        if resolved_path != target_axis_path:
            ok = False
            try:
                ok = copy_dataset_like(h5_file, source_node, target_axis_path)
            except Exception:
                ok = False
            if not ok:
                logger.warning(f"Failed to copy POSITION data from '{resolved_path}' to '{target_axis_path}'.")
                continue

        # Set units attribute
        try:
            dset = h5_file[target_axis_path]
            dset.attrs["units"] = "deg"
        except Exception:
            logger.exception(f"Failed to set units attribute on '{target_axis_path}'.")

        # Delete original PV-named dataset under motor_positions if appropriate
        try:
            mp_prefix = f"{base_group}/motor_positions/"
            if isinstance(resolved_path, str) and resolved_path.startswith(mp_prefix) and resolved_path != target_axis_path:
                if resolved_path in h5_file:
                    del h5_file[resolved_path]
        except Exception:
            logger.exception(f"Failed to delete original PV-named dataset '{resolved_path}'.")

        # Link HKL/<GROUP>/POSITION to the axis dataset and ensure NAME
        hk_group_path = f"{base_group}/HKL/{group_key}"
        pos_path = f"{hk_group_path}/POSITION"
        name_path = f"{hk_group_path}/NAME"
        target_link_path = f"/{target_axis_path}"

        try:
            # Ensure HKL group exists
            h5_file.require_group(hk_group_path)

            # Replace POSITION with soft link — skip if already pointing to the correct target (idempotent)
            # When force=True the guard is bypassed and the link is always rewritten
            needs_link_update = True
            if not force and pos_path in h5_file:
                try:
                    existing_link = h5_file.get(pos_path, getlink=True)
                    if isinstance(existing_link, h5py.SoftLink) and existing_link.path == target_link_path:
                        needs_link_update = False
                except Exception:
                    pass
            if needs_link_update:
                if pos_path in h5_file:
                    del h5_file[pos_path]
                h5_file[pos_path] = h5py.SoftLink(target_link_path)

            # Ensure NAME exists with axis label; if present but differs only by case/whitespace, keep existing
            if name_path not in h5_file:
                _ensure_parent_group(h5_file, name_path)
                h5_file.create_dataset(name_path, data=axis_label)
        except Exception:
            logger.exception(f"Failed to link HKL POSITION or set NAME for group '{group_key}'.")

def _print_structure(h5_file: h5py.File, base_group: str, label: str):
    """Print motor_positions contents and HKL POSITION entries for debugging."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    mp_path = f"{base_group}/motor_positions"
    if mp_path in h5_file:
        print(f"\n[motor_positions]  ({mp_path})")
        for name in h5_file[mp_path].keys():
            node = h5_file[mp_path][name]
            if isinstance(node, h5py.Dataset):
                print(f"  Dataset : {name}  shape={node.shape}  dtype={node.dtype}")
            else:
                print(f"  Group   : {name}/")
    else:
        print(f"\n[motor_positions]  NOT FOUND at {mp_path}")

    hkl_path = f"{base_group}/HKL"
    if hkl_path in h5_file:
        print(f"\n[HKL]  ({hkl_path})")
        hkl = h5_file[hkl_path]
        for grp_name in hkl.keys():
            grp = hkl[grp_name]
            if not isinstance(grp, h5py.Group):
                continue
            print(f"  {grp_name}/")
            for key in grp.keys():
                try:
                    link = grp.get(key, getlink=True)
                except Exception:
                    link = None
                if isinstance(link, h5py.SoftLink):
                    print(f"    {key}  ->  SoftLink({link.path})")
                else:
                    node = grp[key]
                    if isinstance(node, h5py.Dataset):
                        print(f"    {key}  Dataset  shape={node.shape}")
                    else:
                        print(f"    {key}/  Group")
    else:
        print(f"\n[HKL]  NOT FOUND at {hkl_path}")

    print(f"\n[axis_lookup from METADATA.CA]")
    print(f"  (shown as motor_id -> axis_label)")

    print(f"{'='*60}\n")


def _convert_single_file(src_file: Path, toml_path: Path, base_group: str, include: bool, in_place: bool, output_dir: Path, dry_run: bool, force: bool = False) -> str:
    mapping = toml.load(str(toml_path))
    axis_lookup = _build_axis_lookup(mapping)

    print(f"\n[metadata_converter] axis_lookup built from METADATA.CA: {axis_lookup}")

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
        _print_structure(h5_file, base_group, f"BEFORE conversion: {dst.name}")

        # Ensure base group exists
        h5_file.require_group(base_group)
        _process_structure(h5_file, base_group + "/HKL", mapping.get('HKL', mapping), axis_lookup, stats, base_group, include)
        # Post processing: rename motor position datasets and link HKL/POSITION to axis datasets
        try:
            _rename_motor_positions_and_link_hkl(h5_file, mapping, base_group, axis_lookup=axis_lookup, force=force)
        except Exception:
            logger.exception("Failed post-processing to rename motor positions and link HKL POSITION.")

        _print_structure(h5_file, base_group, f"AFTER conversion: {dst.name}")

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
    force: bool = False,
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
    - force: If True, bypass the already-formatted guard and rewrite HKL SoftLinks even if they
             already point to the correct target.

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
            out = _convert_single_file(f, toml_p, base_group, include, in_place, out_dir, dry_run, force=force)
            outputs.append(out)
        except Exception:
            logger.exception(f"Failed to convert: {f}")
    return outputs
