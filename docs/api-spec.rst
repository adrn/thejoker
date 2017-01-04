# TODO: API example for adding more linear parameters to the model

For quick tests or for cases with very uninformative data, you can generate
posterior samples like this::

    thejoker = TheJoker(data, P_min=8*u.day, P_max=8192*u.day)
    # or, to fix the jitter to a specific value:
    # thejoker = TheJoker(data, P_min=8*u.day, P_max=8192*u.day, jitter=5.*u.m/u.s)

    thejoker.sample(n_samples=)

If the data are informative or if you are going to run for many objects, you can
generate the prior samples up front and cache them::

    prior_samples = thejoker.sample_prior(n_samples=2**16)

    # or, if MANY samples:
    thejoker.sample_prior(n_samples=2**16, cache=filename)
