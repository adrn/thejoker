# Third-party
import astropy.units as u
import numpy as np
import time

# Package
from ...data import RVData
from ..likelihood import marginal_ln_likelihood
from ..fast_likelihood import batch_marginal_ln_likelihood
from .. import JokerParams, TheJoker
from .helpers import FakeData


def test_against_py():
    joker_params = JokerParams(P_min=8*u.day, P_max=32768*u.day,
                               jitter=0*u.m/u.s,
                               linear_par_Vinv=np.diag(np.zeros(2)))
    joker = TheJoker(joker_params)

    t = np.random.uniform(0, 250, 16) + 56831.324
    t.sort()

    rv = np.cos(t)
    rv_err = np.random.uniform(0.1, 0.2, t.size)

    data = RVData(t=t, rv=rv*u.km/u.s, stddev=rv_err*u.km/u.s)

    samples = joker.sample_prior(size=16384)

    chunk = []
    for k in samples:
        chunk.append(np.array(samples[k]))

    chunk = np.ascontiguousarray(np.vstack(chunk).T)

    t0 = time.time()
    cy_ll = batch_marginal_ln_likelihood(chunk, data, joker_params)
    print("Cython:", time.time() - t0)

    t0 = time.time()
    n_chunk = len(chunk)
    py_ll = np.zeros(n_chunk)
    for i in range(n_chunk):
        try:
            py_ll[i] = marginal_ln_likelihood(chunk[i], data, joker_params)
        except Exception as e:
            py_ll[i] = np.nan
    print("Python:", time.time() - t0)

    assert np.allclose(np.array(cy_ll), py_ll)
