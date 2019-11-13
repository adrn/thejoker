"""Miscellaneous utilities"""

# Third-party
import astropy.units as u
from astropy.table.meta import get_header_from_yaml
from astropy.io.misc.hdf5 import meta_path
import numpy as np
import tables as tb

# Package
from ..samples import JokerSamples

__all__ = ['batch_tasks', 'table_header_to_units', 'read_batch']


def batch_tasks(n_tasks, n_batches, arr=None, args=None, start_idx=0):
    """Split the tasks into some number of batches to sent out to MPI workers.

    Parameters
    ----------
    n_tasks : int
        The total number of tasks to divide.
    n_batches : int
        The number of batches to split the tasks into. Often, you may want to do
        ``n_batches=pool.size`` for equal sharing amongst MPI workers.
    arr : iterable (optional)
        Instead of returning indices that specify the batches, you can also
        directly split an array into batches.
    args : iterable (optional)
        Other arguments to add to each task.
    start_idx : int (optional)
        What index in the tasks to start from?

    """
    if args is None:
        args = []
    args = list(args)

    tasks = []
    if n_batches > 0 and n_tasks > n_batches:
        # chunk by the number of batches, often the pool size
        base_batch_size = n_tasks // n_batches
        rmdr = n_tasks % n_batches

        i1 = start_idx
        for i in range(n_batches):
            i2 = i1 + base_batch_size
            if i < rmdr:
                i2 += 1

            if arr is None:  # store indices
                tasks.append([(i1, i2), i1] + args)

            else:  # store sliced array
                tasks.append([arr[i1:i2], i1] + args)

            i1 = i2

    else:
        if arr is None:  # store indices
            tasks.append([(start_idx, n_tasks+start_idx), start_idx] + args)

        else:  # store sliced array
            tasks.append([arr[start_idx:n_tasks+start_idx], start_idx] + args)

    return tasks


def table_header_to_units(header_dataset):
    """
    Convert a YAML-ized astropy.table header into a dictionary that maps from
    column name to unit.
    """

    header = get_header_from_yaml(h.decode('utf-8')
                                  for h in header_dataset)

    units = dict()
    for row in header['datatype']:
        units[row['name']] = u.Unit(row.get('unit', u.one))

    return units


def read_batch(prior_samples_file, columns, slice_or_idx, units=None, **kwargs):
    """
    Single-point interface to all read_batch functions below that infers the
    type of read from the ``slice_or_idx`` argument.

    Parameters
    ----------
    prior_samples_file : `str`
        Path to an HDF5 file containing prior samples from The Joker.
    columns : `list`, iterable
        A list of string column names to read in to the batch.
    slice_or_idx : `tuple`, `int`, `numpy.ndarray`
        This determines what rows are read in to the batch from the prior samples file. If a tuple, this is interpreted as a
    """
    if isinstance(slice_or_idx, tuple):
        # read a contiguous batch of prior samples
        batch = read_batch(prior_samples_file, columns,
                           slice(*slice_or_idx), units=units)

    elif isinstance(slice_or_idx, slice):
        # read a contiguous batch of prior samples
        batch = read_batch_slice(prior_samples_file, columns,
                                 slice_or_idx, units=units)

    elif isinstance(slice_or_idx, int):
        # read a random batch of samples of size "slice_or_idx"
        batch = read_random_batch(prior_samples_file, columns,
                                  slice_or_idx, units=units,
                                  **kwargs)

    elif isinstance(slice_or_idx, np.ndarray):
        # read a random batch of samples of size "slice_or_idx"
        batch = read_batch_idx(prior_samples_file, columns,
                               slice_or_idx, units=units)

    else:
        raise ValueError("TODO: DOH")

    return batch


def read_batch_slice(prior_samples_file, columns, slice, units=None):
    """
    Read a batch (row block) of prior samples into a plain numpy array,
    converting units where necessary.
    """

    path = JokerSamples._hdf5_path

    batch = None
    with tb.open_file(prior_samples_file, mode='r') as f:

        for i, name in enumerate(columns):
            arr = f.root[path].read(slice.start, slice.stop, slice.step,
                                    field=name)
            if batch is None:
                batch = np.zeros((len(arr), len(columns)))
            batch[:, i] = arr

        if units is not None:
            table_units = table_header_to_units(f.root[meta_path(path)])
            for i, name in enumerate(columns):
                if name in units:
                    batch[:, i] *= table_units[name].to(units[name])

    return batch


def read_batch_idx(prior_samples_file, columns, idx, units=None):
    """
    Read a batch (row block) of prior samples specified by the input index
    array, ``idx``, into a plain numpy array, converting units where necessary.
    """
    path = JokerSamples._hdf5_path

    batch = np.zeros((len(idx), len(columns)))
    with tb.open_file(prior_samples_file, mode='r') as f:
        for i, name in enumerate(columns):
            batch[:, i] = f.root[path].read_coordinates(idx, field=name)

        if units is not None:
            table_units = table_header_to_units(f.root[meta_path(path)])
            for i, name in enumerate(columns):
                if name in units:
                    batch[:, i] *= table_units[name].to(units[name])

    return batch


def read_random_batch(prior_samples_file, columns, size, units=None,
                      random_state=None):
    """
    Read a random batch (row block) of prior samples into a plain numpy array,
    converting units where necessary.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    path = JokerSamples._hdf5_path
    with tb.open_file(prior_samples_file, mode='r') as f:
        idx = random_state.randint(0, f.root[path].shape[0], size=size)

    return read_batch_idx(prior_samples_file, columns, idx=idx, units=units)
