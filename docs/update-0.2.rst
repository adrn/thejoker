.. _update-0.2-guide:

***************
Updating to 0.2
***************

Starting with ``twobody`` version >=0.1, and ``thejoker`` version >=0.2, the
following user-facing backwards-incompatible changes were made:

- In the `JokerSamples` object, and in saved/cached versions, eccentricity was
  renamed from ``ecc`` to ``e``, and the phase or mean anomaly at a reference
  time was renamed from ``phi0`` to ``M0``.
- The Joker now supports specifying a reference epoch, and the phase ``M0`` is
  assumed to be at this epoch. Previously, the reference time was always the
  earliest time in the data, and the phase was assumed to be at MJD=0.

If you updated The Joker and are now having errors reading posterior sample
files generated from previous versions of The Joker, or are getting errors
related to the ``twobody`` package, you have two options:

1. Update both packages and use compatibility functions to convert your old
   samples file.
2. Install old versions of both packages.

See below for more information.


Option 1: upgrade ``twobody`` and ``thejoker``
----------------------------------------------

Starting with version 0.2, ``twobody`` and ``thejoker`` should remain
API-compatible. To install the latest versions of both packages, the easiest way
is to use ``pip``::

    pip install twobody thejoker

If you have done this, and still have cached or saved posterior samples from a
previous run of The Joker (i.e. from an older version)


Option 2: downgrade ``twobody`` and ``thejoker``
------------------------------------------------

To install versions of twobody and thejoker from before API-breaking changes
were made, use::

    pip install git+https://github.com/adrn/twobody@fb6578a6ff86cae3ea64338282939e4d1d76afa9
    pip install git+https://github.com/adrn/thejoker@275b79e616d148044f6339c01a38d38a66e863df
