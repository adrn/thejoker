********************************
Frequently Asked Questions (FAQ)
********************************

Q: I keep getting a warning about eccentric_anomaly_from_mean_anomaly reaching
a maximum number of iterations
==============================================================================

If you see a ``RuntimeWarning`` about ``eccentric_anomaly_from_mean_anomaly()``,
this likely just means the iterative method we use to solve for the eccentric
anomaly is getting stuck in a limit cycle. This can happen when your data span
a huge range in days. But don't worry too much about this -- for most extreme
cases (e.g., when the data span years) the variations are below
:math:`10^{-11}`. If you see this warning, you can just increase the default
tolerance from :math:`10^{-13}` to :math:`10^{-11}` to remove the warning. From
the high-level interface (e.g., from `~thejoker.sampler.params.JokerParams`),
pass the keyword argument ``anomaly_tol=1E-11`` to the class on initialization
to change the tolerance.
