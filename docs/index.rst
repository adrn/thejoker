****************************
The Joker [YO-ker] /'joʊkər/
****************************

Introduction
============

|thejoker| [#f1]_ is a custom Monte Carlo sampler for the `two-body problem
<https://en.wikipedia.org/wiki/Two-body_problem>`_ that generates posterior
samplings in Keplerian orbital parameters given radial velocity observations of
stars. It is designed to deliver converged posterior samplings even when the
radial velocity measurements are sparse or very noisy. It is therefore useful
for constraining the orbital properties of binary star or star-planet systems.
Though it fundamentally assumes that any system has two massive bodies (and only
the primary is observed), |thejoker| can also be used for hierarchical systems
in which the velocity perturbations from a third or other bodies are much longer
than the dominant companion. See the paper [#f2]_ for more details about the
method and applications.

.. toctree::
    :maxdepth: 1

    install
    changes


Tutorials
=========

.. toctree::
    :maxdepth: 1

    examples/1-Getting-started.ipynb
    examples/2-Customize-prior.ipynb
    examples/3-Polynomial-velocity-trend.ipynb
    examples/4-Continue-sampling-mcmc.ipynb
    examples/5-Calibration-offsets.ipynb


Science demonstrations
======================

.. toctree::
    :maxdepth: 1

    examples/Thompson-black-hole.ipynb
    examples/Strader-circular-only.ipynb


Getting started
===============

Generating samples with |thejoker| requires three things:

    #. **The data**, `~thejoker.RVData`: radial velocity measurements,
       uncertainties, and observation times
    #. **The priors**, `~thejoker.JokerPrior`:
       the prior distributions over the parameters in *The Joker*
    #. **The sampler**, `~thejoker.TheJoker`: the work horse that runs the
       rejection sampler

Here, we'll work through a simple example to generate poserior samples for
orbital parameters given some sparse, simulated radial velocity data (shown
below). We'll first use these plain arrays to construct a `~thejoker.RVData`
object:

    >>> import astropy.units as u
    >>> import thejoker as tj
    >>> t = [0., 49.452, 95.393, 127.587, 190.408]
    >>> rv = [38.77, 39.70, 37.45, 38.31, 38.31] * u.km/u.s
    >>> err = [0.184, 0.261, 0.112, 0.155, 0.223] * u.km/u.s
    >>> data = tj.RVData(t=t, rv=rv, rv_err=err)
    >>> ax = data.plot() # doctest: +SKIP
    >>> ax.set_xlim(-10, 200) # doctest: +SKIP

.. plot::
    :align: center
    :width: 512

    from thejoker.data import RVData
    import astropy.units as u

    t = [0., 49.452, 95.393, 127.587, 190.408]
    rv = [38.77, 39.70, 37.45, 38.31, 38.31] * u.km/u.s
    err = [0.184, 0.261, 0.112, 0.155, 0.223] * u.km/u.s

    data = RVData(t=t, rv=rv, rv_err=err)
    ax = data.plot() # doctest: +SKIP
    ax.set_xlim(-10, 200)

We next need to specify the prior distributions for the parameters of
|thejoker|. The default prior, explained in the docstring of
`~thejoker.JokerPrior.default()`, assumes some reasonable defaults where
possible, but requires specifying the minimum and maximum period to sample over,
along with parameters that specify the prior over the linear parameters in *The
Joker* (the velocity semi-amplitude, ``K``, and the systemic velocity, ``v0``):

    >>> import numpy as np
    >>> prior = tj.JokerPrior.default(P_min=2*u.day, P_max=256*u.day,
    ...                               sigma_K0=30*u.km/u.s,
    ...                               sigma_v=100*u.km/u.s)

With the data and prior created, we can now instantiate the sampler object and
run the rejection sampler:

    >>> joker = tj.TheJoker(prior)
    >>> prior_samples = prior.sample(size=100_000)
    >>> samples = joker.rejection_sample(data, prior_samples) # doctest: +SKIP

Of the 100_000 prior samples we generated, only a handful pass the rejection
sampling step of |thejoker|. Let's visualize the surviving samples in the
subspace of the period :math:`P` and velocity semi-amplitude :math:`K`. We'll
also plot the true values as a green marker. As a separate plot, we'll also
visualize orbits computed from these posterior samples (check the source code
below to see how these were made):

.. plot::
    :align: center
    :width: 512

    from thejoker import JokerPrior, TheJoker, RVData
    from thejoker.plot import plot_rv_curves
    import astropy.units as u

    t = [0., 49.452, 95.393, 127.587, 190.408]
    rv = [38.77, 39.70, 37.45, 38.31, 38.31] * u.km/u.s
    err = [0.184, 0.261, 0.112, 0.155, 0.223] * u.km/u.s

    data = RVData(t=t, rv=rv, rv_err=err)
    prior = JokerPrior.default(P_min=2*u.day, P_max=256*u.day,
                               sigma_K0=30*u.km/u.s, sigma_v=100*u.km/u.s)
    joker = TheJoker(prior)

    prior_samples = prior.sample(size=100_000)
    samples = joker.rejection_sample(data, prior_samples)

    fig, ax = plt.subplots(1, 1, figsize=(6,6)) # doctest: +SKIP
    ax.scatter(samples['P'].value, samples['K'].to(u.km/u.s).value,
               marker='.', color='k', alpha=0.45) # doctest: +SKIP
    ax.set_xlabel("$P$ [day]")
    ax.set_ylabel("$K$ [km/s]")
    ax.set_xlim(2, 256)
    ax.set_ylim(0.75, 3.)

    ax.scatter(61.942, 1.3959, marker='o', color='#31a354', zorder=-100)

    fig, ax = plt.subplots(1, 1, figsize=(8,5)) # doctest: +SKIP
    t_grid = np.linspace(-10, 210, 1024)
    plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
                   plot_kwargs=dict(color='#888888'))
    ax.set_xlim(-5, 205)


API
===

.. automodapi:: thejoker
    :no-inheritance-diagram:


.. rubric:: Footnotes

.. [#f1] Short for Johannes Kepler.
.. [#f2] `<https://arxiv.org/abs/1610.07602>`_
