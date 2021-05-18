# Standard library
import os
import warnings

# Third-party
from astropy.table.meta import get_header_from_yaml, get_yaml_from_table
from astropy.io.misc.hdf5 import _encode_mixins, meta_path
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils import metadata

from thejoker.thejoker import validate_prepare_data


def _custom_tbl_dtype_compare(dtype1, dtype2):
    """This is a custom equality operator for comparing table data types that
    is less strict about units when unit is missing in one and dimensionless in
    the other.
    """

    for d1, d2 in zip(dtype1, dtype2):
        for k in set(list(d1.keys()) + list(d2.keys())):
            if k == 'unit':
                if d1.get(k, '') != '' and k not in d2:
                    return False
                if d2.get(k, '') != '' and k not in d1:
                    return False
                if d1.get(k, '') != d2.get(k, ''):
                    return False
            else:
                if d1.get(k, '1') != d2.get(k, '2'):
                    return False

    return True


def write_table_hdf5(table, output, path=None, compression=False,
                     append=False, overwrite=False, serialize_meta=False,
                     metadata_conflicts='error', **create_dataset_kwargs):
    """
    Write a Table object to an HDF5 file

    This requires `h5py <http://www.h5py.org/>`_ to be installed.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Data table that is to be written to file.
    output : str or :class:`h5py:File` or :class:`h5py:Group`
        If a string, the filename to write the table to. If an h5py object,
        either the file or the group object to write the table to.
    path : str
        The path to which to write the table inside the HDF5 file.
        This should be relative to the input file or group.
        If not specified, defaults to ``__astropy_table__``.
    compression : bool or str or int
        Whether to compress the table inside the HDF5 file. If set to `True`,
        ``'gzip'`` compression is used. If a string is specified, it should be
        one of ``'gzip'``, ``'szip'``, or ``'lzf'``. If an integer is
        specified (in the range 0-9), ``'gzip'`` compression is used, and the
        integer denotes the compression level.
    append : bool
        Whether to append the table to an existing HDF5 file.
    overwrite : bool
        Whether to overwrite any existing file without warning.
        If ``append=True`` and ``overwrite=True`` then only the dataset will be
        replaced; the file/group will not be overwritten.
    metadata_conflicts : str
        How to proceed with metadata conflicts. This should be one of:
            * ``'silent'``: silently pick the last conflicting meta-data value
            * ``'warn'``: pick the last conflicting meta-data value, but emit a
              warning (default)
            * ``'error'``: raise an exception.
    **create_dataset_kwargs
        Additional keyword arguments are passed to `h5py.File.create_dataset`.
    """

    from astropy.table import meta
    try:
        import h5py
    except ImportError:
        raise Exception("h5py is required to read and write HDF5 files")

    if path is None:
        # table is just an arbitrary, hardcoded string here.
        path = '__astropy_table__'
    elif path.endswith('/'):
        raise ValueError("table path should end with table name, not /")

    if '/' in path:
        group, name = path.rsplit('/', 1)
    else:
        group, name = None, path

    if isinstance(output, (h5py.File, h5py.Group)):
        if len(list(output.keys())) > 0 and name == '__astropy_table__':
            raise ValueError("table path should always be set via the "
                             "path= argument when writing to existing "
                             "files")
        elif name == '__astropy_table__':
            warnings.warn("table path was not set via the path= argument; "
                          "using default path {}".format(path))

        if group:
            try:
                output_group = output[group]
            except (KeyError, ValueError):
                output_group = output.create_group(group)
        else:
            output_group = output

    elif isinstance(output, str):

        if os.path.exists(output) and not append:
            if overwrite and not append:
                os.remove(output)
            else:
                raise OSError(f"File exists: {output}")

        # Open the file for appending or writing
        f = h5py.File(output, 'a' if append else 'w')

        # Recursively call the write function
        try:
            return write_table_hdf5(table, f, path=path,
                                    compression=compression, append=append,
                                    overwrite=overwrite,
                                    serialize_meta=serialize_meta,
                                    **create_dataset_kwargs)
        finally:
            f.close()

    else:

        raise TypeError('output should be a string or an h5py File or '
                        'Group object')

    # Check whether table already exists
    existing_header = None
    if name in output_group:
        if append and overwrite:
            # Delete only the dataset itself
            del output_group[name]
        elif append:
            # Data table exists, so we interpret "append" to mean "extend
            # existing table with the table passed in". However, this requires
            # the table to have been written by this function in the past, so it
            # should have a metadata header
            if meta_path(name) not in output_group:
                raise ValueError("No metadata exists for existing table. We "
                                 "can only append tables if metadata "
                                 "is consistent for all tables")

            # Load existing table header:
            existing_header = get_header_from_yaml(
                h.decode('utf-8') for h in output_group[meta_path(name)])
        else:
            raise OSError(f"Table {path} already exists")

    # Encode any mixin columns as plain columns + appropriate metadata
    table = _encode_mixins(table)

    # Table with numpy unicode strings can't be written in HDF5 so
    # to write such a table a copy of table is made containing columns as
    # bytestrings.  Now this copy of the table can be written in HDF5.
    if any(col.info.dtype.kind == 'U' for col in table.itercols()):
        table = table.copy(copy_data=False)
        table.convert_unicode_to_bytestring()

    # Warn if information will be lost when serialize_meta=False.  This is
    # hardcoded to the set difference between column info attributes and what
    # HDF5 can store natively (name, dtype) with no meta.
    if serialize_meta is False:
        for col in table.itercols():
            for attr in ('unit', 'format', 'description', 'meta'):
                if getattr(col.info, attr, None) not in (None, {}):
                    warnings.warn("table contains column(s) with defined 'unit', 'format',"
                                  " 'description', or 'meta' info attributes. These will"
                                  " be dropped since serialize_meta=False.",
                                  AstropyUserWarning)

    if existing_header is None:  # Just write the table and metadata
        # Write the table to the file
        if compression:
            if compression is True:
                compression = 'gzip'
            dset = output_group.create_dataset(name, data=table.as_array(),
                                               compression=compression,
                                               **create_dataset_kwargs)
        else:
            dset = output_group.create_dataset(name, data=table.as_array(),
                                               **create_dataset_kwargs)

        if serialize_meta:
            header_yaml = meta.get_yaml_from_table(table)

            header_encoded = [h.encode('utf-8') for h in header_yaml]
            output_group.create_dataset(meta_path(name),
                                        data=header_encoded)

        else:
            # Write the Table meta dict key:value pairs to the file as HDF5
            # attributes.  This works only for a limited set of scalar data types
            # like numbers, strings, etc., but not any complex types.  This path
            # also ignores column meta like unit or format.
            for key in table.meta:
                val = table.meta[key]
                try:
                    dset.attrs[key] = val
                except TypeError:
                    warnings.warn("Attribute `{}` of type {} cannot be written to "
                                  "HDF5 files - skipping. (Consider specifying "
                                  "serialize_meta=True to write all meta data)"
                                  .format(key, type(val)), AstropyUserWarning)

    else:  # We need to append the tables!
        try:
            # FIXME: do something with the merged metadata!
            metadata.merge(existing_header['meta'],
                           table.meta,
                           metadata_conflicts=metadata_conflicts)
        except metadata.MergeConflictError:
            raise metadata.MergeConflictError(
                "Cannot append table to existing file because "
                "the existing file table metadata and this "
                "table object's metadata do not match. If you "
                "want to ignore this issue, or change to a "
                "warning, set metadata_conflicts='silent' or 'warn'.")

        # Now compare datatype of this object and on disk
        this_header = get_header_from_yaml(get_yaml_from_table(table))

        if not _custom_tbl_dtype_compare(existing_header['datatype'],
                                         this_header['datatype']):
            raise ValueError(
                "Cannot append table to existing file because "
                "the existing file table datatype and this "
                "object's table datatype do not match. "
                f"{existing_header['datatype']} vs. {this_header['datatype']}")

        # If we got here, we can now try to append:
        current_size = len(output_group[name])
        output_group[name].resize((current_size + len(table), ))
        output_group[name][current_size:] = table.as_array()


def inferencedata_to_samples(joker_prior, inferencedata, data,
                             prune_divergences=True):
    """
    Create a ``JokerSamples`` instance from an arviz object.

    Parameters
    ----------
    joker_prior : `thejoker.JokerPrior`
    inferencedata : `arviz.InferenceData`
    data : `thejoker.RVData`
    prune_divergences : bool (optional)

    """
    from thejoker.samples import JokerSamples
    import exoplanet.units as xu

    if hasattr(inferencedata, 'posterior'):
        posterior = inferencedata.posterior

    else:
        posterior = inferencedata
        inferencedata = None

    data, *_ = validate_prepare_data(data,
                                     joker_prior.poly_trend,
                                     joker_prior.n_offsets)

    samples = JokerSamples(poly_trend=joker_prior.poly_trend,
                           n_offsets=joker_prior.n_offsets,
                           t_ref=data.t_ref)

    names = joker_prior.par_names

    for name in names:
        if name in joker_prior.pars:
            par = joker_prior.pars[name]
            unit = getattr(par, xu.UNIT_ATTR_NAME)
            samples[name] = posterior[name].values.ravel() * unit
        else:
            samples[name] = posterior[name].values.ravel()

    if hasattr(posterior, 'logp'):
        samples['ln_posterior'] = posterior.logp.values.ravel()

    for name in ['ln_likelihood', 'ln_prior']:
        if hasattr(posterior, name):
            samples[name] = getattr(posterior, name).values.ravel()

    if prune_divergences:
        if inferencedata is None:
            raise ValueError(
                "If you want to remove divergences, you must pass in the root "
                "level inferencedata object (instead of, e.g., inferencedata. "
                "posterior")

        divergences = inferencedata.sample_stats.diverging.values.ravel()
        samples = samples[~divergences]

    return samples


def trace_to_samples(self, trace, data, names=None):
    """
    Create a ``JokerSamples`` instance from a pymc3 trace object.

    Parameters
    ----------
    trace : `~pymc3.backends.base.MultiTrace`
    """
    import pymc3 as pm
    import exoplanet.units as xu
    from thejoker.samples import JokerSamples

    df = pm.trace_to_dataframe(trace)

    data, *_ = validate_prepare_data(data,
                                     self.prior.poly_trend,
                                     self.prior.n_offsets)

    samples = JokerSamples(poly_trend=self.prior.poly_trend,
                           n_offsets=self.prior.n_offsets,
                           t_ref=data.t_ref)

    if names is None:
        names = self.prior.par_names

    for name in names:
        par = self.prior.pars[name]
        unit = getattr(par, xu.UNIT_ATTR_NAME)
        samples[name] = df[name].values * unit

    return samples
