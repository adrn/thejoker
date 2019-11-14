# Third-party
from astropy.tests.helper import quantity_allclose
import astropy.units as u
import numpy as np
import pytest

# Package
from ..mcmc import TheJokerMCMCModel


@pytest.mark.skip(reason="TODO: reimplement this")
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
        params = self.joker_params['binary']
        model = TheJokerMCMCModel(params, data)

        # test roundtrip
        mcmc_p = model.pack_samples(truth)
        p2 = model.unpack_samples(mcmc_p)
        for k in truth:
            assert quantity_allclose(p2[k], truth[k])

        assert 'jitter' in p2 and quantity_allclose(p2['jitter'],
                                                    params.jitter)

        lp = model.ln_prior(model._strip_units(p2))
        assert np.isfinite(lp)

        ll = model.ln_likelihood(model._strip_units(p2))
        assert np.isfinite(ll).all()

        lnpost = model(mcmc_p)
        assert np.isfinite(lnpost)
        assert np.allclose(lnpost, lp+ll.sum())
