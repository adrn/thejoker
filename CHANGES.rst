1.2 (unreleased)
----------------

- ``TheJoker`` now expects a ``numpy.random.Generator`` instance instead of a
  ``RandomState`` for controlling random number generation.

- Renamed ``t0`` to ``t_ref`` in places where it actually represents a reference
  time.

- Added functionality to compute ``t0``, i.e. the time of phase=0, for samples.

- Added functions to compute time-sampling statistics for the samples returned
  from The Joker (see: ``thejoker.samples_analysis``).

1.1 (2020-04-19)
----------------

- Removed ``astropy-helpers`` from the package infrastructure.
- Switched to using ``tox`` for testing and documentation builds.
- Fixed bug in ``JokerSamples.read()`` when reading from an HDF5 file.
