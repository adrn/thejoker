# Third-party
import time

import astropy.units as u
import numpy as np
import pymc as pm

import thejoker.units as xu

# Package
from ...data import RVData
from ...likelihood_helpers import get_constant_term_design_matrix
from ...prior import JokerPrior
from ..fast_likelihood import CJokerHelper
from .py_likelihood import get_aAbB, marginal_ln_likelihood

# TODO: horrible copy-pasta test code below


def test_against_py():
    rng = np.random.default_rng(seed=42)

    with pm.Model():
        K = xu.with_unit(pm.Normal("K", 0, 10.0), u.km / u.s)

        prior = JokerPrior.default(
            P_min=8 * u.day,
            P_max=32768 * u.day,
            s=0 * u.m / u.s,
            sigma_v=100 * u.km / u.s,
            pars={"K": K},
        )

    # t = np.random.uniform(0, 250, 16) + 56831.324
    t = np.sort(rng.uniform(0, 250, 3)) + 56831.324
    rv = np.cos(t)
    rv_err = rng.uniform(0.1, 0.2, t.size)
    data = RVData(t=t, rv=rv * u.km / u.s, rv_err=rv_err * u.km / u.s)
    trend_M = get_constant_term_design_matrix(data)

    samples = prior.sample(size=8192)
    chunk, _ = samples.pack()
    chunk = np.ascontiguousarray(chunk)

    helper = CJokerHelper(data, prior, trend_M)

    t0 = time.time()
    cy_ll = helper.batch_marginal_ln_likelihood(chunk)
    print("Cython:", time.time() - t0)

    t0 = time.time()
    py_ll = marginal_ln_likelihood(samples, prior, data)
    print("Python:", time.time() - t0)

    assert np.allclose(np.array(cy_ll), py_ll)


def test_scale_varK_against_py():
    rng = np.random.default_rng(seed=42)

    prior = JokerPrior.default(
        P_min=8 * u.day,
        P_max=32768 * u.day,
        s=0 * u.m / u.s,
        sigma_K0=25 * u.km / u.s,
        sigma_v=100 * u.km / u.s,
    )

    # t = np.random.uniform(0, 250, 16) + 56831.324
    t = np.sort(rng.uniform(0, 250, 3)) + 56831.324
    rv = np.cos(t)
    rv_err = rng.uniform(0.1, 0.2, t.size)
    data = RVData(t=t, rv=rv * u.km / u.s, rv_err=rv_err * u.km / u.s)
    trend_M = get_constant_term_design_matrix(data)

    samples = prior.sample(size=8192)
    chunk, _ = samples.pack()
    chunk = np.ascontiguousarray(chunk)

    helper = CJokerHelper(data, prior, trend_M)

    t0 = time.time()
    cy_ll = helper.batch_marginal_ln_likelihood(chunk)
    print("Cython:", time.time() - t0)

    t0 = time.time()
    py_ll = marginal_ln_likelihood(samples, prior, data)
    print("Python:", time.time() - t0)

    assert np.allclose(np.array(cy_ll), py_ll)


def test_likelihood_helpers():
    rng = np.random.default_rng(seed=42)

    with pm.Model():
        K = xu.with_unit(pm.Normal("K", 0, 1.0), u.km / u.s)

        prior = JokerPrior.default(
            P_min=8 * u.day,
            P_max=32768 * u.day,
            s=0 * u.m / u.s,
            sigma_v=1 * u.km / u.s,
            pars={"K": K},
        )

    # t = rng.uniform(0, 250, 16) + 56831.324
    t = np.sort(rng.uniform(0, 250, 3)) + 56831.324
    rv = np.cos(t)
    rv_err = rng.uniform(0.1, 0.2, t.size)
    data = RVData(t=t, rv=rv * u.km / u.s, rv_err=rv_err * u.km / u.s)
    trend_M = get_constant_term_design_matrix(data)

    samples = prior.sample(size=16)  # HACK: MAGIC NUMBER 16!
    chunk, _ = samples.pack()

    helper = CJokerHelper(data, prior, trend_M)

    py_vals = get_aAbB(samples, prior, data)

    for i in range(len(samples)):
        ll = helper.test_likelihood_worker(np.ascontiguousarray(chunk[i]))
        assert np.abs(ll) < 1e8

        cy_vals = {}
        for k in py_vals.keys():
            cy_vals[k] = np.array(getattr(helper, k))

        for k in py_vals:
            # print(i, k)
            # print('cy', cy_vals[k])
            # print('py', py_vals[k][i])
            # print(np.allclose(cy_vals[k], py_vals[k][i]))
            # print()
            assert np.allclose(cy_vals[k], py_vals[k][i])
