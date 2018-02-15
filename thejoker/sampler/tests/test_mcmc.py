# Third-party
import astropy.units as u
import numpy as np

# Package
from ..mcmc import TheJokerMCMCModel
from ..params import JokerParams
from .helpers import FakeData


class TestMCMC(object):

    def truths_to_nlp(self, truths):
        # P, M0, ecc, omega
        P = truths['P'].to(u.day).value
        M0 = truths['M0'].to(u.radian).value
        ecc = truths['e']
        omega = truths['omega'].to(u.radian).value
        return np.array([P, M0, ecc, omega, 0.])

    def setup(self):
        d = FakeData()
        self.data = d.datasets
        self.joker_params = d.params
        self.truths = d.truths

    def test_funcs(self):
        data = self.data['binary']
        truth = self.truths['binary']
        nlp = self.truths_to_nlp(truth)
        params = self.joker_params['binary']
        model = TheJokerMCMCModel(params, data)

        p = np.concatenate((nlp, [truth['K'].value], [truth['v0'].value]))
        mcmc_p = model.to_mcmc_params(p)
        p2 = model.from_mcmc_params(mcmc_p)
        assert np.allclose(p, p2.reshape(p.shape)) # test roundtrip

        lp = model.ln_prior(p)
        assert np.isfinite(lp)

        ll = model.ln_likelihood(p)
        assert np.isfinite(ll).all()

        # remove jitter from params passed in to mcmc_p
        mcmc_p = list(mcmc_p)
        mcmc_p.pop(5) # log-jitter is 5th index in mcmc packed
        lnpost = model.ln_posterior(mcmc_p)

        assert np.isfinite(lnpost)
        assert np.allclose(lnpost, lp+ll.sum())
