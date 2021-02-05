# Third-party
from astropy.table import QTable
import astropy.units as u
from astropy.io.misc.hdf5 import meta_path
import h5py
import numpy as np
import tables as tb

# Package
from ..samples import JokerSamples
from ..utils import (batch_tasks, table_header_to_units,
                     table_contains_column,
                     read_batch, read_batch_slice, read_batch_idx,
                     read_random_batch)


def test_batch_tasks():
    N = 10000
    start_idx = 1103
    tasks = batch_tasks(N, n_batches=16, start_idx=start_idx)
    assert tasks[0][0][0] == start_idx
    assert tasks[-1][0][1] == N+start_idx

    # try with an array:
    tasks = batch_tasks(N, n_batches=16, start_idx=start_idx,
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

    # TODO: pytables doesn't support variable length strings
    # with tb.open_file(filename, mode='r') as f:
    #     units = table_header_to_units(f.root[meta_path('test')])
    with h5py.File(filename, mode='r') as f:
        units = table_header_to_units(f[meta_path('test')])

    for col in tbl.colnames:
        assert tbl[col].unit == units[col]


def test_table_contains_column(tmpdir):
    filename = str(tmpdir / 'test.hdf5')

    tbl = QTable()
    tbl['a'] = np.arange(10) * u.kpc
    tbl['b'] = np.arange(10) * u.km/u.s
    tbl['c'] = np.arange(10) * u.day
    tbl.write(filename, path=JokerSamples._hdf5_path, serialize_meta=True)

    # TODO: pytables doesn't support variable length strings
    # with tb.open_file(filename, mode='r') as f:
    with h5py.File(filename, mode='r') as f:
        assert table_contains_column(f, 'a')
        assert table_contains_column(f, 'b')
        assert table_contains_column(f, 'c')
        assert not table_contains_column(f, 'd')


def test_read_batch_slice(tmpdir):
    filename = str(tmpdir / 'test.hdf5')

    tbl = QTable()
    tbl['a'] = np.arange(100) * u.kpc
    tbl['b'] = np.arange(100) * u.km/u.s
    tbl['c'] = np.arange(100) * u.day
    tbl.write(filename, path=JokerSamples._hdf5_path, serialize_meta=True)

    for read_func in [read_batch_slice, read_batch]:
        batch = read_func(filename, ['a', 'b'], slice(10, 20))
        assert batch.shape == (10, 2)
        assert np.allclose(batch[:, 0], tbl['a'].value[10:20])
        assert np.allclose(batch[:, 1], tbl['b'].value[10:20])

        batch = read_func(filename, ['b', 'c'], slice(0, 100),
                          units={'b': u.kpc/u.Myr})
        assert batch.shape == (100, 2)
        assert np.allclose(batch[:, 0], tbl['b'].to_value(u.kpc/u.Myr))
        assert np.allclose(batch[:, 1], tbl['c'].value)

        batch = read_func(filename, ['b', 'c'], slice(0, 100, 2))
        assert batch.shape == (50, 2)


def test_read_batch_idx(tmpdir):
    filename = str(tmpdir / 'test.hdf5')

    tbl = QTable()
    tbl['a'] = np.arange(100) * u.kpc
    tbl['b'] = np.arange(100) * u.km/u.s
    tbl['c'] = np.arange(100) * u.day
    tbl.write(filename, path=JokerSamples._hdf5_path, serialize_meta=True)

    for read_func in [read_batch_idx, read_batch]:
        idx = np.arange(10, 20, 1)
        batch = read_func(filename, ['a', 'b'], idx, units=None)
        assert batch.shape == (len(idx), 2)
        assert np.allclose(batch[:, 0], tbl['a'].value[idx])
        assert np.allclose(batch[:, 1], tbl['b'].value[idx])

        batch = read_func(filename, ['b', 'c'], idx,
                          units={'b': u.kpc/u.Myr})
        assert batch.shape == (len(idx), 2)
        assert np.allclose(batch[:, 0], tbl['b'].to_value(u.kpc/u.Myr)[idx])
        assert np.allclose(batch[:, 1], tbl['c'].value[idx])


def test_read_random_batch(tmpdir):
    filename = str(tmpdir / 'test.hdf5')

    tbl = QTable()
    tbl['a'] = np.arange(100) * u.kpc
    tbl['b'] = np.arange(100) * u.km/u.s
    tbl['c'] = np.arange(100) * u.day
    tbl.write(filename, path=JokerSamples._hdf5_path, serialize_meta=True)

    for read_func in [read_random_batch, read_batch]:
        batch = read_func(filename, ['a', 'b'], 10, units=None)
        assert batch.shape == (10, 2)

        batch = read_func(filename, ['b', 'c'], 100,
                          units={'b': u.kpc/u.Myr})
        assert batch.shape == (100, 2)
