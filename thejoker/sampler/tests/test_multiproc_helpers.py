# Third-party
import astropy.units as u
import h5py
import numpy as np
import schwimmbad

# Package
from ..multiproc_helpers import get_good_sample_indices, sample_indices_to_full_samples
from .helpers import FakeData

class TestMultiproc(object):

    # TODO: this is bad to copy pasta from test_likelihood.py
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
        phi0 = np.random.uniform(0, 2*np.pi, n)
        ecc = np.zeros(n)
        omega = np.zeros(n)
        jitter = np.zeros(n)
        samples = np.vstack((P,phi0,ecc,omega,jitter)).T

        with h5py.File(prior_samples_file) as f:
            f['samples'] = samples

        idx = get_good_sample_indices(n, prior_samples_file, data, joker_params, pool)
        assert len(idx) > 1

        full_samples = sample_indices_to_full_samples(idx, prior_samples_file, data,
                                                      joker_params, pool)
        print(full_samples)

