1.2.2 (2022-03-03)
------------------

- Fix bug with wheel builds.


1.2.1 (2022-03-03)
------------------

- Compatibility with astropy v5.0 and python 3.10.


1.2 (2021-11-09)
----------------

- ``TheJoker`` now expects a ``numpy.random.Generator`` instance instead of a
  ``RandomState`` for controlling random number generation.

- Renamed ``t0`` to ``t_ref`` in places where it actually represents a reference
  time.

- Added functionality to compute ``t0``, i.e. the time of phase=0, for samples.

- Added functions to compute time-sampling statistics for the samples returned
  from The Joker (see: ``thejoker.samples_analysis``).

- Fixed a bug in the ``UniformLog`` distribution that affected running MCMC.

1.1 (2020-04-19)
----------------

- Removed ``astropy-helpers`` from the package infrastructure.
- Switched to using ``tox`` for testing and documentation builds.
- Fixed bug in ``JokerSamples.read()`` when reading from an HDF5 file.
