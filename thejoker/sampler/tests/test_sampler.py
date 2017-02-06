# Third-party
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest

# Package
from ..params import JokerParams
from ..sampler import TheJoker
from .helpers import FakeData

class TestSampler(object):

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

    def test_init(self):
        TheJoker(self.joker_params['binary'])
        TheJoker(self.joker_params['triple'])

        # invalid pool
        class DummyPool(object):
            pass

        pool = DummyPool()
        with pytest.raises(TypeError):
            TheJoker(self.joker_params['binary'], pool=pool)

        # invalid random state
        rnd = "aisdfijs"
        with pytest.raises(TypeError):
            TheJoker(self.joker_params['triple'], random_state=rnd)

        # invalid params
        pars = "aisdfijs"
        with pytest.raises(TypeError):
            TheJoker(pars)

    def test_sample_prior(self):

        rnd1 = np.random.RandomState(42)
        joker1 = TheJoker(self.joker_params['binary'], random_state=rnd1)

        rnd2 = np.random.RandomState(42)
        joker2 = TheJoker(self.joker_params['triple'], random_state=rnd2)

        samples1 = joker1.sample_prior(8)
        samples2 = joker2.sample_prior(8)

        for key in samples1.keys():
            assert quantity_allclose(samples1[key], samples2[key])

        samples, ln_vals = joker2.sample_prior(8, return_logvals=True)
        assert np.isfinite(ln_vals).all()

    def test_rejection_sample(self):

        rnd = np.random.RandomState(42)

        data = self.data['binary']
        joker = TheJoker(self.joker_params['binary'], random_state=rnd)

        with pytest.raises(ValueError):
            joker.rejection_sample(data)

        joker.rejection_sample(data, n_prior_samples=128)

        # check that jitter is always set to the fixed value
        jitter = 5.*u.m/u.s
        params = JokerParams(P_min=8*u.day, P_max=1024*u.day, jitter=jitter)
        joker = TheJoker(params)

        prior_samples = joker.sample_prior(128)
        assert quantity_allclose(prior_samples['jitter'], jitter)

        full_samples = joker.rejection_sample(data, n_prior_samples=128)
        assert quantity_allclose(full_samples['jitter'], jitter)
