# Third-party
from astropy.table import QTable
import astropy.units as u
from astropy.io.misc.hdf5 import meta_path
import numpy as np
import tables as tb

# Package
from ...samples import JokerSamples
from ..utils import chunk_tasks, table_header_to_units, read_chunk


def test_chunk_tasks():
    N = 10000
    start_idx = 1103
    tasks = chunk_tasks(N, n_batches=16, start_idx=start_idx)
    assert tasks[0][0][0] == start_idx
    assert tasks[-1][0][1] == N+start_idx

    # try with an array:
    tasks = chunk_tasks(N, n_batches=16, start_idx=start_idx,
                        arr=np.random.random(size=8*N))
    n_tasks = sum([tasks[i][0].size for i in range(len(tasks))])
    assert n_tasks == N


def test_table_header_to_units(tmpdir):
    filename = str(tmpdir / 'test.hdf5')

    tbl = QTable()
    tbl['a'] = np.arange(10) * u.kpc
    tbl['b'] = np.arange(10) * u.km/u.s
    tbl['c'] = np.arange(10) * u.day
    tbl.write(filename, path='test', serialize_meta=True)

    with tb.open_file(filename, mode='r') as f:
        units = table_header_to_units(f.root[meta_path('test')])

    for col in tbl.colnames:
        assert tbl[col].unit == units[col]


def test_read_chunk(tmpdir):
    filename = str(tmpdir / 'test.hdf5')

    tbl = QTable()
    tbl['a'] = np.arange(100) * u.kpc
    tbl['b'] = np.arange(100) * u.km/u.s
    tbl['c'] = np.arange(100) * u.day
    tbl.write(filename, path=JokerSamples._hdf5_path, serialize_meta=True)

    chunk = read_chunk(filename, ['a', 'b'], 10, 20, units=None)
    assert chunk.shape == (10, 2)
    assert np.allclose(chunk[:, 0], tbl['a'].value[10:20])
    assert np.allclose(chunk[:, 1], tbl['b'].value[10:20])

    chunk = read_chunk(filename, ['b', 'c'], 0, 100,
                       units={'b': u.kpc/u.Myr})
    assert chunk.shape == (100, 2)
    assert np.allclose(chunk[:, 0], tbl['b'].to_value(u.kpc/u.Myr))
    assert np.allclose(chunk[:, 1], tbl['c'].value)
