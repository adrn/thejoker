import pymc3 as pm
from pymc3.distributions import draw_values

def _get_P(pars, unpars, P_lim=None):
    """Note: this is an internal function.

    This returns the period prior as a pymc3 variable.
    """

    if 'P' in pars and P_lim is not None:
        raise ValueError("Period appears in the input parameters, but period "
                         "limits are also specified (via P_lim). Specify one "
                         "or the other, but on both.")

    elif 'P' not in pars and P_lim is None:
        raise ValueError("The period prior requires some specification: "
                         "Either pass in an explicit pymc3 variable with a "
                         "defined distribution, or pass in limits, P_lim, for "
                         "an assumed (default) prior proportional to 1/P")

    # Short-circuit if P is already defined
    if 'P' in pars:
        return pars['P'], unpars.get('P', None)

    # At this point, P is not in pars but P_lim has been specified:
    with pm.Model() as model:
        logP_kw = dict()
        if P_lim is not None:
            logP_kw['lower'] = np.log10(P_lim[0])
            logP_kw['upper'] = np.log10(P_lim[1])

        logP = pm.Uniform('logP', **logP_kw)
        default_P = pm.Deterministic('P', 10**logP)

    return default_P, logP


def _get_prior_pars(pars=None, unpars=None, P_lim=None):
    """Note: these are internal functions. Use ``joker.sample_prior()`` to
    generate samples from the prior.
    """

    # Parse and clean up the input
    # - pars can be a dict or list
    if pars is None:
        pars = dict()
    elif isinstance(pars, list):
        pars = {p.name: p for p in pars}

    # unpars must be a dict
    if unpars is None:
        unpars = dict()

    # Initialize the default prior:
    # - P is special because we allow passing in a range, assuming 1/P period:
    pars['P'], unpars['P'] = _get_P(pars, unpars, P_lim)

    # Now set up the default priors on
    with pm.Model() as model:
        default_pars = {
            'e': xo.distributions.eccentricity.kipping13('e'),
            'omega': pm.Uniform('omega', lower=0, upper=2*np.pi),
            'M0': pm.Uniform('M0', lower=0, upper=2*np.pi),
            'jitter': pm.Constant('jitter', 0.)
        }

    # Fill the parameter dictionary with defaults if not specified
    for k in default_pars.keys():
        pars[k] = pars.get(k, default_pars.get(k))
        unpars[k] = unpars.get(k, None)

    return pars, unpars


def _sample_prior(pars, unpars, size=1, return_logprobs=False):
    """Note: these are internal functions. Use ``joker.sample_prior()`` to
    generate samples from the prior.
    """
    pars_list = list(pars.values())
    npars = len(pars_list)

    log_prior = []
    if return_logprobs:
        # Add deterministic variables to track the value of the prior at each
        # sample generated:
        with pm.Model() as model:
            for par in pars_list:
                if par.name in unpars.keys() and unpars[par.name] is not None:
                    upar = unpars[par.name]
                    logp_var = pm.Deterministic(f'{upar.name}_log_prior',
                                                upar.distribution.logp(upar))
                else:
                    logp_var = pm.Deterministic(f'{par.name}_log_prior',
                                                par.distribution.logp(par))
                log_prior.append(logp_var)

    samples_values = draw_values(pars_list + log_prior,
                                 size=size)
    prior_samples = {p.name: samples
                     for p, samples in zip(pars_list,
                                           samples_values[:npars])}
    if not return_logprobs:
        return prior_samples

    log_prior_vals = {p.name: samples
                      for p, samples in zip(pars_list,
                                            samples_values[npars:])}

    return prior_samples, log_prior_vals
