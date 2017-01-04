# TODO: API example for adding more linear parameters to the model

For quick tests or for cases with very uninformative data, you can generate
posterior samples like this::

    thejoker = TheJoker(data, P_min=8*u.day, P_max=8192*u.day)
    # or, to fix the jitter to a specific value:
    # thejoker = TheJoker(data, P_min=8*u.day, P_max=8192*u.day, jitter=5.*u.m/u.s)
    # TODO: how to configure prior on jitter?

    samples = thejoker.sample(n_samples=1024)

The object returned is a `JokerParams` object, TODO: keeps track of units.

If the data are informative or if you are going to run for many objects, you can
generate the prior samples up front and cache them::

    prior_samples = thejoker.sample_prior(n_samples=2**16)

    # or, if MANY samples:
    thejoker.sample_prior(n_samples=2**16, cache=filename)

Generate posterior samples from a set of cached prior samples::

    samples = thejoker.sample(n_samples=1024, prior_cache=filename)

Control the parent random state object used to generate random numbers::

    thejoker.random_state = np.random.RandomState(seed=42)

See `random numbers <random.rst>`_ for more information TODO: write page on
random numbers and MPI.

--

To control the size of the prior batch, the number of prior samples generated
or pulled from the cache file and send to the likelihood worker

    samples = thejoker.sample(n_samples=1024, prior_batch_size=2**8)

--

To continue with MCMC

    emcee_sampler = thejoker.run_mcmc(w0, n_walkers=32, n_burn=XX, n_steps=YY)

--

To use an MPI pool

    thejoker = TheJoker(data, pool=pool)
    samples = thejoker.sample(n_samples=1024)
    thejoker.pool.close()

--

TODO: 2 cases to show are (1) floating calibration offsets, (2) long-term
      velocity trend. Also, how to add priors?

To add more linear parameters to the model

    thejoker.set_linear_parameters() # TODO: pass what? dict?

    def func(pars, data):
        return pars[XX]*data._t_bmjd + pars[XX]*data._t_bmjd**2 # quadratic

    def func(pars, data):
        idx = data._t_bmjd < (XX-data.t_offset)
        return pars[XX]*data._t_bmjd + pars[XX]*data._t_bmjd**2 # quadratic

    thejoker.set_model_rv_terms(func) # TODO: name

TODO: default adds v0 as linear param
