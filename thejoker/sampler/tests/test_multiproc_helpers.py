# Third-party
import astropy.units as u
import h5py
import numpy as np
import schwimmbad

# Package
from ..multiproc_helpers import (get_good_sample_indices, compute_likelihoods,
                                 sample_indices_to_full_samples, chunk_tasks)
from .helpers import FakeData


def test_chunk_tasks():
    N = 10000
    tasks = chunk_tasks(N, n_batches=16, start_idx=1000)
    assert tasks[0][0][0] == 1000
    assert tasks[-1][0][1] == N

    # try with an array:
    start_idx = 1103
    tasks = chunk_tasks(N, n_batches=16, start_idx=start_idx,
                        arr=np.random.random(size=N))
    n_tasks = sum([tasks[i][0].size for i in range(len(tasks))])
    assert n_tasks == (N-start_idx)


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
        P = np.random.uniform(nlp[0]-2., nlp[0]+2., n)
        M0 = np.random.uniform(0, 2*np.pi, n)
        ecc = np.zeros(n)
        omega = np.zeros(n)
        jitter = np.zeros(n)
        samples = np.vstack((P,M0,ecc,omega,jitter)).T

        # TODO: use save_prior_samples here

        with h5py.File(prior_samples_file) as f:
            f['samples'] = samples

        lls = compute_likelihoods(n, prior_samples_file, 0, data,
                                  joker_params, pool)
        idx = get_good_sample_indices(lls)
        assert len(idx) >= 1

        full_samples = sample_indices_to_full_samples(idx, prior_samples_file,
                                                      data, joker_params, pool)
        print(full_samples)
