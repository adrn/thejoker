# Third-party
import astropy.units as u
import h5py
import numpy as np

# Package
from ...data import RVData
from ..io import save_prior_samples


class TestIO(object):

    def setup(self):
        mjd = np.linspace(55555., 56555., 256)
        self.data = RVData(mjd, np.sin(mjd)*u.km/u.s,
                           stddev=0.1*np.random.uniform(size=mjd.size)*u.km/u.s)

        n = 128
        samples = dict()
        samples['P'] = np.random.uniform(size=n) * u.day
        samples['M0'] = np.random.uniform(size=n) * u.radian
        samples['omega'] = np.random.uniform(size=n) * u.radian
        samples['e'] = np.random.uniform(size=n) * u.one
        samples['jitter'] = np.random.uniform(size=n) * u.m/u.s
        self.samples = samples
        self.n = n

    def test_save_prior_samples(self, tmpdir):

        path = str(tmpdir.join('io-test1.hdf5'))
        save_prior_samples(path, self.samples, self.data.rv.unit)
        with h5py.File(path, 'r') as f:
            for k in self.samples.keys():
                assert f['samples/'+k][:].shape == (self.n, )

        path = str(tmpdir.join('io-test2.hdf5'))
        with h5py.File(path, 'w') as f:
            save_prior_samples(f, self.samples, self.data.rv.unit)

        with h5py.File(path, 'r') as f:
            for k in self.samples.keys():
                assert f['samples/'+k][:].shape == (self.n, )
