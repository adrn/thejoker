****************************
The Joker [YO-ker] /'joʊkər/
****************************

Introduction
============

|thejoker| [#f1]_ is a custom Monte Carlo sampler for the two-body problem and
is therefore useful for constraining star-star or star-planet systems. It is
designed to generate posterior samples over the orbital parameters of the
primary (observed) star given time-domain radial velocity measurements of this
star. Though it is assumed the system has only a primary and a companion,
|thejoker| can also be used for hierarchical systems in which the velocity
perturbations from a third or other bodies are much longer than the dominant
companion. See the paper [#f2]_ for more details about the method and
applications.

|thejoker| is particularly useful for sparse or low-quality radial velocity
data.

.. toctree::
    :maxdepth: 1

    install
    random-numbers
    faq
    update-0.2

.. note::

    The Joker was recently updated with some API-breaking changes. To see a list
    of these changes, and a guide on how to migrate to the new version, please
    see :ref:`update-0.2-guide`.

Getting started
===============

Generating samples with |thejoker| requires specifying three things:

    #. **The data**, `~thejoker.data.RVData`: radial velocity measurements,
       uncertainties, and observation times
    #. **The model parameters**, `~thejoker.sampler.params.JokerParams`:
       hyper-parameters, what parameters to sample over
    #. **The sampler parameters**, `~thejoker.sampler.sampler.TheJoker`: how
       many prior samples to generate, etc.

Here we'll work through a simple example to generate samples for orbital
parameters for some sparse, pre-generated data (shown below). We'll first turn
these data into a `~thejoker.data.RVData` object:

    >>> from thejoker.data import RVData
    >>> import astropy.units as u
    >>> t = [0., 49.452, 95.393, 127.587, 190.408]
    >>> rv = [38.77, 39.70, 37.45, 38.31, 38.31] * u.km/u.s
    >>> err = [0.184, 0.261, 0.112, 0.155, 0.223] * u.km/u.s
    >>> data = RVData(t=t, rv=rv, stddev=err)
    >>> ax = data.plot() # doctest: +SKIP
    >>> ax.set_xlim(-10, 200) # doctest: +SKIP
    >>> ax.set_xlabel("Time [day]") # doctest: +SKIP
    >>> ax.set_ylabel("RV [km/s]") # doctest: +SKIP

.. plot::
    :align: center
    :width: 512

    from thejoker.data import RVData
    import astropy.units as u

    t = [0., 49.452, 95.393, 127.587, 190.408]
    rv = [38.77, 39.70, 37.45, 38.31, 38.31] * u.km/u.s
    err = [0.184, 0.261, 0.112, 0.155, 0.223] * u.km/u.s

    data = RVData(t=t, rv=rv, stddev=err)
    ax = data.plot() # doctest: +SKIP
    ax.set_xlim(-10, 200)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("RV [km/s]")

We next need to specify some hyper-parameters for |thejoker|. At minimum, we
have to specify the minimum and maximum period to consider:

    >>> from thejoker.sampler import JokerParams
    >>> params = JokerParams(P_min=8*u.day, P_max=512*u.day)

Finally we can create the sampler object and run:

    >>> from thejoker.sampler import TheJoker
    >>> joker = TheJoker(params)
    >>> samples = joker.rejection_sample(data, n_prior_samples=65536) # doctest: +SKIP

Of the 65536 prior samples we considered, only a handful pass the rejection
sampling step of |thejoker|. Let's visualize the surviving samples in the
subspace of the period :math:`P` and velocity semi-amplitude :math:`K`. We'll
also plot the true values as a green marker. As a separate plot, we'll also
visualize orbits computed from these posterior samples (check the source code
below to see how these were made):

.. plot::
    :align: center
    :width: 512

    from thejoker.data import RVData
    from thejoker.sampler import JokerParams, TheJoker
    from thejoker.plot import plot_rv_curves
    import astropy.units as u
    import schwimmbad

    t = [0., 49.452, 95.393, 127.587, 190.408]
    rv = [38.77, 39.70, 37.45, 38.31, 38.31] * u.km/u.s
    err = [0.184, 0.261, 0.112, 0.155, 0.223] * u.km/u.s

    data = RVData(t=t, rv=rv, stddev=err)
    params = JokerParams(P_min=8*u.day, P_max=512*u.day)
    pool = schwimmbad.MultiPool()
    joker = TheJoker(params, pool=pool)

    samples = joker.rejection_sample(data, n_prior_samples=65536)

    fig, ax = plt.subplots(1, 1, figsize=(6,6)) # doctest: +SKIP
    ax.scatter(samples['P'].value, samples['K'].to(u.km/u.s).value,
               marker='.', color='k', alpha=0.45) # doctest: +SKIP
    ax.set_xlabel("$P$ [day]")
    ax.set_ylabel("$K$ [km/s]")
    ax.set_xlim(-5, 128)
    ax.set_ylim(0.75, 3.)

    ax.scatter(61.942, 1.3959, marker='o', color='#31a354', zorder=-100)

    fig, ax = plt.subplots(1, 1, figsize=(8,5)) # doctest: +SKIP
    t_grid = np.linspace(-10, 210, 1024)
    plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
                   plot_kwargs=dict(color='#888888'))
    ax.set_xlim(-5, 205)

More examples / tutorials
=========================

.. toctree::
    :maxdepth: 1

    examples/multiproc-example.ipynb
..    examples/velocity-trend.ipynb

.. examples/continue-mcmc.ipynb
.. examples/mpi.rst
.. examples/caching-samples.ipynb

API
===

.. toctree::
    :maxdepth: 1

    data
    sampler

.. rubric:: Footnotes

.. [#f1] Short for Johannes Kepler.
.. [#f2] `<https://arxiv.org/abs/1610.07602>`_
