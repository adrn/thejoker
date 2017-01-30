# Third-party
import astropy.units as u
import numpy as np

# Package
from ..mcmc import (pack_samples, pack_samples_mcmc, to_mcmc_params, from_mcmc_params,
                    ln_likelihood, ln_prior, ln_posterior)

from .helpers import FakeData

def test_roundtrip():
    # jitter = 0, no v_terms
    p = np.array([63.12, 1.952, 0.1, 0.249, 0., 1.5])
    mcmc_p = to_mcmc_params(p)
    p2 = from_mcmc_params(mcmc_p)
    assert np.allclose(p, p2.reshape(p.shape))

    # with v_terms
    p = np.array([63.12, 1.952, 0.1, 0.249, 0., 1.5, -31.5, 1E-4])
    mcmc_p = to_mcmc_params(p)
    p2 = from_mcmc_params(mcmc_p)
    assert np.allclose(p, p2.reshape(p.shape))

# def test_emcee_run():
#     d = FakeData()
#     data = d.data['binary']
#     params = d.joker_params['binary']
#     samples = d.truths['binary'].copy()
#     samples['v0'] = d.v0

#     for k in samples:
#         samples[k] = u.Quantity([samples[k], samples[k]])

#     mcmc_p = pack_samples_mcmc(samples, params, data)[0]
#     lnpost = ln_posterior(mcmc_p, params, data)

#     import emcee

#     n_walkers = 128
#     p0 = emcee.utils.sample_ball(mcmc_p, 1E-6*np.abs(mcmc_p), size=n_walkers)

#     sampler = emcee.EnsembleSampler(n_walkers, p0.shape[1],
#                                     lnpostfn=ln_posterior, args=(params,data))
#     pos,prob,state = sampler.run_mcmc(p0, 1024) # MAGIC NUMBER

#     import matplotlib.pyplot as plt
#     nwalkers, nlinks, dim = sampler.chain.shape
#     for k in range(dim):
#         plt.figure()
#         for n in range(nwalkers):
#             plt.plot(sampler.chain[n,:,k], marker='', drawstyle='steps', alpha=0.1)

#     plt.show()

#     return

#     # -----

#     idx = 2
#     vals = np.linspace(0.95, 1.05, 128) * mcmc_p[idx]
#     probs = np.zeros_like(vals)
#     for i,val in enumerate(vals):
#         _p = mcmc_p.copy()
#         _p[idx] = val
#         probs[i] = ln_posterior(_p, params, data)
#     plt.plot(vals, probs)
#     plt.axvline(mcmc_p[idx], color='r', zorder=-100)
#     plt.show()

class TestMCMC(object):

    # TODO: repeated code!
    def truths_to_nlp(self, truths):
        # P, phi0, ecc, omega
        P = truths['P'].to(u.day).value
        phi0 = truths['phi0'].to(u.radian).value
        ecc = truths['ecc']
        omega = truths['omega'].to(u.radian).value
        return np.array([P, phi0, ecc, omega, 0.])

    def setup(self):
        d = FakeData()
        self.fd = d
        self.data = d.data
        self.joker_params = d.joker_params
        self.truths = d.truths

    def test_funcs(self):
        data = self.data['binary']
        truth = self.truths['binary']
        nlp = self.truths_to_nlp(truth)
        params = self.joker_params['binary']

        p = np.concatenate((nlp, [truth['K'].value], [self.fd.v0.value]))
        mcmc_p = to_mcmc_params(p)
        p2 = from_mcmc_params(mcmc_p)
        assert np.allclose(p, p2.reshape(p.shape)) # test roundtrip

        lp = ln_prior(p, params)
        assert np.isfinite(lp)

        ll = ln_likelihood(p, params, data)
        assert np.isfinite(ll).all()

        # remove jitter from params passed in to mcmc_p
        mcmc_p = list(mcmc_p)
        mcmc_p.pop(5) # log-jitter is 5th index in mcmc packed
        lnpost = ln_posterior(mcmc_p, params, data)

        assert np.isfinite(lnpost)
        assert np.allclose(lnpost, lp+ll.sum())
