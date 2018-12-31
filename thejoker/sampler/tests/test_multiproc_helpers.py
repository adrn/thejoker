# Third-party
import astropy.units as u
import h5py
import numpy as np
import schwimmbad

# Package
from ..multiproc_helpers import (get_good_sample_indices, compute_likelihoods,
                                 sample_indices_to_full_samples, chunk_tasks)
from .helpers import FakeData
from ..io import save_prior_samples
from ..samples import JokerSamples


def test_chunk_tasks():
    N = 10000
    start_idx = 1103
    tasks = chunk_tasks(N, n_batches=16, start_idx=start_idx)
    assert tasks[0][0][0] == start_idx
    assert tasks[-1][0][1] == N+start_idx

    # try with an array:
    tasks = chunk_tasks(N, n_batches=16, start_idx=start_idx,
                        arr=np.random.random(size=8*N))
    n_tasks = sum([tasks[i][0].size for i in range(len(tasks))])
    assert n_tasks == N


class TestMultiproc(object):

    # TODO: this is bad to copy pasta from test_likelihood.py
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

    def test_multiproc_helpers(self, tmpdir):
        prior_samples_file = str(tmpdir.join('prior-samples.h5'))
        pool = schwimmbad.SerialPool()

        data = self.data['circ_binary']
        joker_params = self.joker_params['circ_binary']
        truth = self.truths['circ_binary']
        nlp = self.truths_to_nlp(truth)

        # write some nonsense out to the prior file
        n = 8192
        samples = JokerSamples()
        samples['P'] = np.random.uniform(nlp[0]-2., nlp[0]+2., n) * u.day
        samples['M0'] = np.random.uniform(0, 2*np.pi, n) * u.radian
        samples['e'] = np.zeros(n) * u.one
        samples['omega'] = np.zeros(n) * u.radian
        samples['jitter'] = np.zeros(n) * u.km/u.s
        save_prior_samples(prior_samples_file, samples, u.km/u.s)

        lls = compute_likelihoods(n, prior_samples_file, 0, data,
                                  joker_params, pool)
        idx = get_good_sample_indices(lls)
        assert len(idx) >= 1

        lls = compute_likelihoods(n, prior_samples_file, 0, data,
                                  joker_params, pool, n_batches=13)
        idx = get_good_sample_indices(lls)
        assert len(idx) >= 1

        full_samples = sample_indices_to_full_samples(idx, prior_samples_file,
                                                      data, joker_params, pool)
        print(full_samples)
