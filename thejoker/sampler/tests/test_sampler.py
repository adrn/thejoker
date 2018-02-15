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

    def test_init(self):
        TheJoker(self.joker_params['binary'])
        # TheJoker(self.joker_params['triple']) # TODO: disabled

        # invalid pool
        class DummyPool(object):
            pass

        pool = DummyPool()
        with pytest.raises(TypeError):
            TheJoker(self.joker_params['binary'], pool=pool)

        # invalid random state
        rnd = "aisdfijs"
        with pytest.raises(TypeError):
            TheJoker(self.joker_params['binary'], random_state=rnd)

        # invalid params
        pars = "aisdfijs"
        with pytest.raises(TypeError):
            TheJoker(pars)

    def test_sample_prior(self):
        rnd = np.random.RandomState(42)
        joker = TheJoker(self.joker_params['binary'], random_state=rnd)
        samples = joker.sample_prior(8)
        samples, ln_vals = joker.sample_prior(8, return_logprobs=True)
        assert np.isfinite(ln_vals).all()

    def test_rejection_sample(self):
        rnd = np.random.RandomState(42)

        # First, try just running rejection_sample()
        data = self.data['binary']
        joker = TheJoker(self.joker_params['binary'], random_state=rnd)

        with pytest.raises(ValueError):
            joker.rejection_sample(data)

        joker.rejection_sample(data, n_prior_samples=128)

        # Now re-run with jitter set, check that it's always the fixed value
        jitter = 5.*u.m/u.s
        params = JokerParams(P_min=8*u.day, P_max=128*u.day, jitter=jitter)
        joker = TheJoker(params)

        prior_samples = joker.sample_prior(128)
        assert quantity_allclose(prior_samples['jitter'], jitter)

        full_samples = joker.rejection_sample(data, n_prior_samples=128)
        assert quantity_allclose(full_samples['jitter'], jitter)

    def test_iterative_rejection_sample(self):

        # First, try just running rejection_sample()
        data = self.data['binary']
        jitter = 100*u.m/u.s
        params = JokerParams(P_min=8*u.day, P_max=128*u.day, jitter=jitter)
        joker = TheJoker(params)

        samples = joker.iterative_rejection_sample(data, n_prior_samples=100000,
                                                   n_requested_samples=2)

        assert quantity_allclose(samples['jitter'], jitter)

    def test_mcmc_continue(self):
        rnd = np.random.RandomState(42)

        # First, try just running rejection_sample()
        data = self.data['binary']
        joker = TheJoker(self.joker_params['binary'], random_state=rnd)

        samples = joker.rejection_sample(data, n_prior_samples=16384)
        samples = joker.mcmc_sample(data, samples, n_steps=8, n_burn=8,
                                    n_walkers=128, return_sampler=False)
        samples, sampler = joker.mcmc_sample(data, samples, n_steps=8, n_burn=8,
                                             n_walkers=128, return_sampler=True)
