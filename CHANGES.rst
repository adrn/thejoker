1.2 (unreleased)
================

- ``TheJoker`` now expects a ``numpy.random.Generator`` instance instead of a
  ``RandomState`` for controlling random number generation.

1.1 (2020-04-19)
================

- Removed ``astropy-helpers`` from the package infrastructure.
- Switched to using ``tox`` for testing and documentation builds.
- Fixed bug in ``JokerSamples.read()`` when reading from an HDF5 file.
