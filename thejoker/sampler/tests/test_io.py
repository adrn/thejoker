# Third-party
import astropy.units as u
import h5py
import numpy as np

# Package
from ...data import RVData
from ..io import pack_prior_samples, save_prior_samples

class TestIO(object):

    def setup(self):
        mjd = np.linspace(55555., 56555., 256)
        self.data = RVData(mjd, np.sin(mjd)*u.km/u.s,
                           stddev=0.1*np.random.uniform(size=mjd.size)*u.km/u.s)

        n = 128
        samples = dict()
        samples['P'] = np.random.uniform(size=n) * u.day
        samples['phi0'] = np.random.uniform(size=n) * u.radian
        samples['omega'] = np.random.uniform(size=n) * u.radian
        samples['ecc'] = np.random.uniform(size=n) * u.one
        samples['jitter'] = np.random.uniform(size=n) * u.m/u.s
        self.samples = samples
        self.n = n

    def test_pack_prior_samples(self):
        samples = self.samples.copy()

        M,units = pack_prior_samples(samples, self.data.rv.unit)
        assert units[-1] == u.km/u.s
        assert M.shape == (self.n, 5)

        samples.pop('jitter')
        assert 'jitter' not in samples
        M,units = pack_prior_samples(samples, self.data.rv.unit)
        assert units[-1] == u.km/u.s
        assert M.shape == (self.n, 5)

    def test_save_prior_samples(self, tmpdir):

        path = str(tmpdir.join('io-test1.hdf5'))
        save_prior_samples(path, self.samples, self.data.rv.unit)
        with h5py.File(path, 'r') as f:
            assert f['samples'][:].shape == (self.n, 5)

        path = str(tmpdir.join('io-test2.hdf5'))
        with h5py.File(path, 'w') as f:
            save_prior_samples(f, self.samples, self.data.rv.unit)

        with h5py.File(path, 'r') as f:
            assert f['samples'][:].shape == (self.n, 5)
