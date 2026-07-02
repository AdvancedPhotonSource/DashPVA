"""Unit tests for HpcMcaAssociator (MCA-to-image time association)."""

import time

import numpy as np
import pvaccess as pva
from pvapy.utility.adImageUtility import AdImageUtility

from dashpva.consumers.hpc.meta.hpc_mca_associator import HpcMcaAssociator


def make_assoc(**cfg):
    # startMonitors=False => no live Channel Access; readings are injected.
    base = {'startMonitors': False}
    base.update(cfg)
    return HpcMcaAssociator(base)


def test_default_pv_list_and_short_names():
    a = make_assoc()
    assert a.mcaPvs == [f'12idc:3820:mca{i}.VAL' for i in range(1, 9)]
    assert a._short_name('12idc:3820:mca1.VAL') == 'mca1'


def test_match_within_and_stale():
    a = make_assoc(mcaWindow=0.5, mcaElements=8)
    now = 1000.0
    a._mca_latest = {
        'mca1': (np.arange(20, dtype=float), now),         # coincident -> within
        'mca2': (np.arange(20, dtype=float), now - 2.0),   # 2 s old -> stale
    }
    m = a._match_mca(now)
    assert m['mca1'][2] is True
    assert m['mca2'][2] is False
    # truncated to first 8 elements
    assert len(m['mca1'][0]) == 8
    assert list(m['mca1'][0]) == list(range(8))


def test_match_short_array_and_all_elements():
    a = make_assoc(mcaElements=8)
    a._mca_latest = {'mca1': (np.array([1.0, 2.0, 3.0]), 5.0)}
    assert list(a._match_mca(5.0)['mca1'][0]) == [1.0, 2.0, 3.0]  # shorter -> as-is

    a_all = make_assoc(mcaElements=0)  # 0 => keep all
    a_all._mca_latest = {'mca1': (np.arange(30, dtype=float), 5.0)}
    assert len(a_all._match_mca(5.0)['mca1'][0]) == 30


def test_hold_last_uses_latest_reading():
    a = make_assoc()
    a._mca_cb(pvname='12idc:3820:mca1.VAL', value=[1, 2, 3], timestamp=10.0)
    a._mca_cb(pvname='12idc:3820:mca1.VAL', value=[9, 9, 9], timestamp=11.0)
    m = a._match_mca(11.0)
    assert list(m['mca1'][0]) == [9.0, 9.0, 9.0]
    assert m['mca1'][1] == 11.0


def _synthetic_image(frame_id, ts, nx=16, ny=16):
    img = (np.arange(nx * ny, dtype=np.uint16) % 100).reshape(ny, nx)
    nt = AdImageUtility.generateNtNdArray2D(frame_id, img, nx, ny, np.dtype('uint16'))
    nt['uniqueId'] = frame_id
    nt['timeStamp'] = pva.PvTimeStamp(ts)
    return nt


def test_process_attaches_all_mca_attributes():
    a = make_assoc(mcaWindow=1.0, mcaElements=8, compress=False)
    now = time.time()
    a._mca_latest = {f'mca{i}': (np.arange(20, dtype=float) + i, now) for i in range(1, 9)}

    captured = []
    a.updateOutputChannel = lambda pvObj: captured.append(pvObj)  # avoid framework publish
    out = a.process(_synthetic_image(1, now))

    names = [attr['name'] for attr in out['attribute']]
    for i in range(1, 9):
        assert f'mca{i}' in names
    assert a.nFramesProcessed == 1
    assert a.nMcaWithin == 8 and a.nMcaStale == 0
    assert len(captured) == 1


def test_process_counts_stale_but_still_attaches():
    a = make_assoc(mcaWindow=0.1, mcaElements=8, compress=False, mcaDropStale=False)
    now = time.time()
    a._mca_latest = {'mca1': (np.arange(10, dtype=float), now - 5.0)}  # very stale
    a.updateOutputChannel = lambda pvObj: None
    out = a.process(_synthetic_image(1, now))
    assert 'mca1' in [attr['name'] for attr in out['attribute']]  # attached
    assert a.nMcaStale == 1 and a.nMcaWithin == 0


def test_process_drops_stale_when_configured():
    a = make_assoc(mcaWindow=0.1, mcaElements=8, compress=False, mcaDropStale=True)
    now = time.time()
    a._mca_latest = {'mca1': (np.arange(10, dtype=float), now - 5.0)}
    a.updateOutputChannel = lambda pvObj: None
    out = a.process(_synthetic_image(1, now))
    assert 'mca1' not in [attr['name'] for attr in out['attribute']]  # dropped
    assert a.nMcaStale == 1
