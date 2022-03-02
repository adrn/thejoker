"""Miscellaneous utilities"""

# Standard library
import inspect
import contextlib
from functools import wraps
import os
from distutils.version import LooseVersion
from tempfile import NamedTemporaryFile

# Third-party
import astropy.units as u
from astropy.table.meta import get_header_from_yaml
from astropy.io.misc.hdf5 import meta_path
import h5py
import numpy as np
import tables as tb

# Package
from .samples import JokerSamples

# TODO: remove this when we drop support for numpy 1.16
# Numpy version:
NUMPY_LT_1_17 = LooseVersion(np.__version__) < '1.17'

if NUMPY_LT_1_17:
    DEFAULT_RNG = np.random.RandomState
    integers = lambda obj, *args, **kwargs: obj.randint(*args, **kwargs)
else:
    DEFAULT_RNG = np.random.default_rng
    integers = lambda obj, *args, **kwargs: obj.integers(*args, **kwargs)


__all__ = ['batch_tasks', 'table_header_to_units', 'read_batch',
           'tempfile_decorator']


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
    if n_batches > 0 and n_tasks >= n_batches:
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


def table_contains_column(root, column):
    from .samples import JokerSamples

    path = meta_path(JokerSamples._hdf5_path)
    header = get_header_from_yaml(h.decode('utf-8') for h in root[path])

    columns = []
    for row in header['datatype']:
        columns.append(row['name'])

    return column in columns


def read_batch(prior_samples_file, columns, slice_or_idx, units=None,
               random_state=None):
    """
    Single-point interface to all read_batch functions below that infers the
    type of read from the ``slice_or_idx`` argument.

    Parameters
    ----------
    prior_samples_file : `str`
        Path to an HDF5 file containing prior samples from The Joker.
    columns : `list`, iterable
        A list of string column names to read in to the batch.
    slice_or_idx : `slice`, `int`, `numpy.ndarray`
        This determines what rows are read in to the batch from the prior
        samples file. If a slice or tuple, this is interpreted as a range of
        contiguous row indices to read. If an integer, this is interpreted as
        the size used to generate random row indices. If a numpy array, this is
        interpreted as the row indices to read.
    units : dict (optional)
        The desired output units to convert the prior samples to.
    random_state : `numpy.random.RandomState`
        Used to generate random row indices if passing an integer (size) in to
        ``slice_or_idx``. This argument is ignored for other read methods.

    Returns
    -------
    batch : `numpy.ndarray`
        The batch of prior samples, loaded into a numpy array and stripped of
        units (converted to the specified ``units``). This array will have
        shape: ``(n_samples, len(columns))``.
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
                                  random_state=random_state)

    elif isinstance(slice_or_idx, np.ndarray):
        # read a random batch of samples of size "slice_or_idx"
        batch = read_batch_idx(prior_samples_file, columns,
                               slice_or_idx, units=units)

    else:
        raise ValueError("Invalid input for slice_or_idx: must be a slice, "
                         "int, or numpy array.")

    return batch


def read_batch_slice(prior_samples_file, columns, slice, units=None):
    """
    Read a batch (row block) of prior samples into a plain numpy array,
    converting units where necessary.
    """

    path = JokerSamples._hdf5_path

    # We have to do this with h5py because current (2021-02-05) versions of
    # pytables don't support variable length strings, which h5py is using to
    # serialize units in the astropy table metadata
    with h5py.File(prior_samples_file, mode='r') as f:
        table_units = table_header_to_units(f[meta_path(path)])

    batch = None
    with tb.open_file(prior_samples_file, mode='r') as f:

        for i, name in enumerate(columns):
            arr = f.root[path].read(slice.start, slice.stop, slice.step,
                                    field=name)
            if batch is None:
                batch = np.zeros((len(arr), len(columns)), dtype=arr.dtype)
            batch[:, i] = arr

        if units is not None:
            # See comment above
            # table_units = table_header_to_units(f.root[meta_path(path)])
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

    # We have to do this with h5py because current (2021-02-05) versions of
    # pytables don't support variable length strings, which h5py is using to
    # serialize units in the astropy table metadata
    with h5py.File(prior_samples_file, mode='r') as f:
        table_units = table_header_to_units(f[meta_path(path)])

    batch = np.zeros((len(idx), len(columns)))
    with tb.open_file(prior_samples_file, mode='r') as f:
        for i, name in enumerate(columns):
            batch[:, i] = f.root[path].read_coordinates(idx, field=name)

        if units is not None:
            # See comment above
            # table_units = table_header_to_units(f.root[meta_path(path)])
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
        random_state = DEFAULT_RNG()

    path = JokerSamples._hdf5_path
    with tb.open_file(prior_samples_file, mode='r') as f:
        idx = integers(random_state, 0, f.root[path].shape[0], size=size)

    return read_batch_idx(prior_samples_file, columns, idx=idx, units=units)


def tempfile_decorator(func):
    wrapped_signature = inspect.signature(func)
    func_args = list(wrapped_signature.parameters.keys())
    if 'prior_samples_file' not in func_args:
        raise ValueError("Cant decorate function because it doesn't contain an "
                         "argument called 'prior_samples_file'")

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        if 'prior_samples_file' in kwargs:
            prior_samples = kwargs['prior_samples_file']
        else:
            prior_samples = args.pop(func_args.index('prior_samples_file'))

        in_memory = kwargs.get('in_memory', False)

        if not isinstance(prior_samples, str) and not in_memory:
            if not isinstance(prior_samples, JokerSamples):
                raise TypeError("prior_samples_file must either be a string "
                                "filename specifying a cache file contining "
                                "prior samples, or must be a JokerSamples "
                                f"instance, not: {type(prior_samples)}")

            # This is required (instead of a context) because the named file
            # can't be opened a second time on Windows...see, e.g.,
            # https://github.com/Kotaimen/awscfncli/issues/93
            f = NamedTemporaryFile(mode='r+', suffix='.hdf5', delete=False)
            f.close()

            try:
                # write samples to tempfile and recursively call this method
                prior_samples.write(f.name, overwrite=True)
                kwargs['prior_samples_file'] = f.name
                func_return = func(*args, **kwargs)
            except Exception as e:
                raise e
            finally:
                os.unlink(f.name)


        else:
            # FIXME: it's a string, so it's probably a filename, but we should
            # validate the contents of the file!
            kwargs['prior_samples_file'] = prior_samples
            func_return = func(*args, **kwargs)

        return func_return

    return wrapper


@contextlib.contextmanager
def random_state_context(random_state):
    state = np.random.get_state()
    if random_state is not None:
        np.random.seed(integers(random_state, 2**32-1))  # HACK
    try:
        yield
    finally:
        if random_state is not None:
            np.random.set_state(state)
