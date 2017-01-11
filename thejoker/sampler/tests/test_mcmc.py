# Third-party
import astropy.units as u
import numpy as np

# Package
from ..mcmc import (to_mcmc_params, from_mcmc_params, ln_likelihood, ln_prior,
                    ln_posterior)

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
