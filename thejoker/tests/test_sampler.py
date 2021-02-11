# Standard library
import os

# Third-party
import astropy.units as u
from astropy.time import Time
import numpy as np
import pytest
from schwimmbad import SerialPool, MultiPool
from twobody import KeplerOrbit

# Package
from ..prior import JokerPrior
from ..data import RVData
from ..thejoker import TheJoker
from .test_prior import get_prior
from ..utils import DEFAULT_RNG, NUMPY_LT_1_17


def make_data(n_times=8, random_state=None, v1=None, K=None):
    if random_state is None:
        random_state = DEFAULT_RNG()
    rnd = random_state

    P = 51.8239 * u.day
    if K is None:
        K = 54.2473 * u.km/u.s
    v0 = 31.48502 * u.km/u.s
    EPOCH = Time('J2000')
    t = Time('J2000') + P * np.sort(rnd.uniform(0, 3., n_times))

    # binary system - random parameters
    orbit = KeplerOrbit(P=P, K=K, e=0.3, omega=0.283*u.radian,
                        M0=2.592*u.radian, t0=EPOCH,
                        i=90*u.deg, Omega=0*u.deg)  # these don't matter

    rv = orbit.radial_velocity(t) + v0
    if v1 is not None:
        rv = rv + v1 * (t - EPOCH).jd * u.day

    err = np.full_like(rv.value, 0.5) * u.km/u.s
    data = RVData(t, rv, rv_err=err)

    return data, orbit


@pytest.mark.parametrize('case', range(get_prior()))
def test_init(case):
    prior, _ = get_prior(case)

    # Try various initializations
    TheJoker(prior)

    with pytest.raises(TypeError):
        TheJoker('jsdfkj')

    # Pools:
    with SerialPool() as pool:
        TheJoker(prior, pool=pool)

    # fail when pool is invalid:
    with pytest.raises(TypeError):
        TheJoker(prior, pool='sdfks')

    # Random state:
    rnd = DEFAULT_RNG(42)
    TheJoker(prior, random_state=rnd)

    # fail when random state is invalid:
    with pytest.raises(TypeError):
        TheJoker(prior, random_state='sdfks')

    if not NUMPY_LT_1_17:
        with pytest.warns(DeprecationWarning):
            rnd = np.random.RandomState(42)
            TheJoker(prior, random_state=rnd)

    # tempfile location:
    joker = TheJoker(prior, tempfile_path='/tmp/joker')
    assert os.path.exists(joker.tempfile_path)


@pytest.mark.parametrize('case', range(get_prior()))
def test_marginal_ln_likelihood(tmpdir, case):
    prior, _ = get_prior(case)

    data, _ = make_data()
    prior_samples = prior.sample(size=100)
    joker = TheJoker(prior)

    # pass JokerSamples instance
    ll = joker.marginal_ln_likelihood(data, prior_samples)
    assert len(ll) == len(prior_samples)

    # save prior samples to a file and pass that instead
    filename = str(tmpdir / 'samples.hdf5')
    prior_samples.write(filename, overwrite=True)

    ll = joker.marginal_ln_likelihood(data, filename)
    assert len(ll) == len(prior_samples)

    # make sure batches work:
    ll = joker.marginal_ln_likelihood(data, filename,
                                      n_batches=10)
    assert len(ll) == len(prior_samples)

    # NOTE: this makes it so I can't parallelize tests, I think
    with MultiPool(processes=2) as pool:
        joker = TheJoker(prior, pool=pool)
        ll = joker.marginal_ln_likelihood(data, filename)
    assert len(ll) == len(prior_samples)


@pytest.mark.parametrize('prior', [
    JokerPrior.default(P_min=5*u.day, P_max=500*u.day,
                       sigma_K0=25*u.km/u.s, sigma_v=100*u.km/u.s),
    JokerPrior.default(P_min=5*u.day, P_max=500*u.day,
                       sigma_K0=25*u.km/u.s, poly_trend=2,
                       sigma_v=[100*u.km/u.s, 0.5*u.km/u.s/u.day])
])
def test_rejection_sample(tmpdir, prior):
    data, orbit = make_data()
    flat_data, orbit = make_data(K=0.1*u.m/u.s)

    prior_samples = prior.sample(size=16384, return_logprobs=True)
    filename = str(tmpdir / 'samples.hdf5')
    prior_samples.write(filename, overwrite=True)

    joker = TheJoker(prior)

    for _samples in [prior_samples, filename]:
        # pass JokerSamples instance, process all samples:
        samples = joker.rejection_sample(data, _samples)
        assert len(samples) > 0
        assert len(samples) < 10  # HACK: this should generally be true...

        # Check that return_logprobs works
        samples = joker.rejection_sample(data, _samples, return_logprobs=True)
        assert len(samples) > 0

        samples = joker.rejection_sample(flat_data, _samples)
        assert len(samples) > 10  # HACK: this should generally be true...

        # Probably the wrong place to do this, but check that computing the log
        # unmarginalized likelihood works:
        lls = samples.ln_unmarginalized_likelihood(flat_data)
        assert np.isfinite(lls).all()

    samples, lls = joker.rejection_sample(flat_data, prior_samples,
                                          return_all_logprobs=True)
    assert len(lls) == len(prior_samples)

    # Check that setting the random state makes it deterministic
    all_Ps = []
    all_Ks = []
    for i in range(10):
        joker = TheJoker(prior, random_state=DEFAULT_RNG(42))
        samples = joker.rejection_sample(flat_data, prior_samples)
        all_Ps.append(samples['P'])
        all_Ks.append(samples['K'])

    for i in range(1, len(all_Ps)):
        assert u.allclose(all_Ps[0], all_Ps[i])
        assert u.allclose(all_Ks[0], all_Ks[i])


@pytest.mark.parametrize('prior', [
    JokerPrior.default(P_min=5*u.day, P_max=500*u.day,
                       sigma_K0=25*u.km/u.s, sigma_v=100*u.km/u.s),
    JokerPrior.default(P_min=5*u.day, P_max=500*u.day,
                       sigma_K0=25*u.km/u.s, poly_trend=2,
                       sigma_v=[100*u.km/u.s, 0.5*u.km/u.s/u.day])
])
def test_iterative_rejection_sample(tmpdir, prior):
    data, orbit = make_data(n_times=3)

    prior_samples = prior.sample(size=10_000, return_logprobs=True)
    filename = str(tmpdir / 'samples.hdf5')
    prior_samples.write(filename, overwrite=True)

    joker = TheJoker(prior, random_state=DEFAULT_RNG(42))

    for _samples in [prior_samples, filename]:
        # pass JokerSamples instance, process all samples:
        samples = joker.iterative_rejection_sample(data, _samples,
                                                   n_requested_samples=4)
        assert len(samples) > 1

        samples = joker.iterative_rejection_sample(data, _samples,
                                                   n_requested_samples=4,
                                                   return_logprobs=True)
        assert len(samples) > 1

    # Check that setting the random state makes it deterministic
    all_Ps = []
    all_Ks = []
    for i in range(10):
        joker = TheJoker(prior, random_state=DEFAULT_RNG(42))
        samples = joker.iterative_rejection_sample(data, prior_samples,
                                                   n_requested_samples=4,
                                                   randomize_prior_order=True)
        all_Ps.append(samples['P'])
        all_Ks.append(samples['K'])

    for i in range(1, len(all_Ps)):
        assert u.allclose(all_Ps[0], all_Ps[i])
        assert u.allclose(all_Ks[0], all_Ks[i])
