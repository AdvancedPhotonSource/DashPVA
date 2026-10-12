"""Transpose /entry/data/data in DashPVA scan files. Nothing else in the file is touched.

    python scripts/transpose_h5.py outputs/              # every .h5 in the folder
    python scripts/transpose_h5.py a.h5 b.h5             # named files
    python scripts/transpose_h5.py 'outputs/pva**.h5'    # a pattern (quote it)
    python scripts/transpose_h5.py outputs/ --copy       # write <name>_T.h5 instead
    python scripts/transpose_h5.py outputs/ --dry-run    # report, change nothing
    python scripts/transpose_h5.py outputs/ --exclude '*_T.h5' hello*

Swaps the last two axes, so (frames, x, y) becomes (frames, y, x) and the frame
axis stays put -- plain .T on a 3D array would reverse the frames too. The
operation is its own inverse: run it twice and the file is back as it was.
"""

import argparse
import fnmatch
import glob
import os
import shutil
import sys

import h5py
import hdf5plugin
import numpy as np

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('paths', nargs='+', help='.h5 files, or folders to scan for them')
parser.add_argument('--copy', action='store_true',
                    help='write <name>_T.h5 beside each file instead of rewriting it')
parser.add_argument('--dry-run', action='store_true', help='report, change nothing')
parser.add_argument('--exclude', nargs='+', default=[], metavar='PATTERN',
                    help="skip files matching these, e.g. --exclude '*_T.h5' hello*")
args = parser.parse_args()

targets = []
for path in args.paths:
    if os.path.isdir(path):
        targets.extend(sorted(glob.glob(os.path.join(path, '*.h5'))))
    elif os.path.isfile(path):
        targets.append(path)
    elif any(ch in path for ch in '*?['):
        # unquoted the shell expands it first; quoted it arrives here, and
        # recursive=True gives ** its usual "any depth" meaning
        found = [m for m in sorted(glob.glob(path, recursive=True)) if os.path.isfile(m)]
        if found:
            targets.extend(found)
        else:
            print(f'skip   {path}: matched nothing')
    else:
        print(f'skip   {path}: not a file or folder')
kept = []
for path in targets:
    hit = next((p for p in args.exclude
                if fnmatch.fnmatch(os.path.basename(path), p) or fnmatch.fnmatch(path, p)), None)
    if hit:
        print(f'skip   {os.path.basename(path)}: excluded by {hit!r}')
    else:
        kept.append(path)
targets = kept
if not targets:
    sys.exit('nothing to do')

blosc = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
done = skipped = failed = planned = 0

for source in targets:
    name = os.path.basename(source)
    try:
        with h5py.File(source, 'r') as probe:
            dataset = probe.get('entry/data/data')
            if not isinstance(dataset, h5py.Dataset):
                print(f'skip   {name}: no /entry/data/data')
                skipped += 1
                continue
            shape = dataset.shape
        if len(shape) < 2:
            print(f'skip   {name}: {shape} is not an image')
            skipped += 1
            continue

        turned = shape[:-2] + (shape[-1], shape[-2])
        if args.dry_run:
            print(f'would  {name}: {shape} -> {turned}')
            planned += 1
            continue

        target = source
        if args.copy:
            root, ext = os.path.splitext(source)
            target = f'{root}_T{ext or ".h5"}'
            shutil.copy2(source, target)

        with h5py.File(target, 'a') as h5f:
            group = h5f['entry/data']
            old = group['data']
            attrs = dict(old.attrs)
            new = group.create_dataset('__data_T__', shape=turned, dtype=old.dtype,
                                       chunks=(1,) + turned[-2:] if len(turned) == 3 else None,
                                       **blosc)
            if len(turned) == 3:
                for index in range(turned[0]):      # frame at a time: these files are large
                    new[index] = np.swapaxes(np.asarray(old[index]), -2, -1)
            else:
                new[...] = np.swapaxes(np.asarray(old[()]), -2, -1)
            for key, value in attrs.items():
                new.attrs[key] = value
            del group['data']
            group.move('__data_T__', 'data')

        print(f'ok     {os.path.basename(target)}: {shape} -> {turned}')
        done += 1
    except Exception as error:
        print(f'FAILED {name}: {type(error).__name__}: {error}')
        failed += 1

if args.dry_run:
    print(f'\n{planned} would be transposed, {skipped} skipped, {failed} failed')
else:
    print(f'\n{done} transposed, {skipped} skipped, {failed} failed')
if done and not args.copy:
    print('rewritten in place; HDF5 does not reclaim the old blocks, '
          'so run h5repack if the files need to shrink')
