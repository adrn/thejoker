# Third-party
import astropy.units as u
import numpy as np

# Package
from ...data import RVData
from ..params import JokerParams, PolynomialVelocityTrend
from ..likelihood import design_matrix, tensor_vector_scalar

class TestLikelihood(object):

    def setup(self):
        self.data = dict()

        mjd = np.linspace(0, 1000., 128) + 55555.
        P = 53.2352 # u.day
        v0 = 100. # u.km/u.s
        rv = 10. * np.cos(2*np.pi*mjd/P) + v0
        self.data['binary'] = RVData(mjd, rv*u.km/u.s, stddev=np.ones_like(rv)*0.1*u.km/u.s)
        self.P = P
        self.v0 = v0

        mjd = np.linspace(0, 1000., 128) + 55555.
        v1 = 1E-3 # u.km/u.s/u.day
        rv = 10. * np.cos(2*np.pi*mjd/P) + v0 + v1*mjd
        self.data['triple'] = RVData(mjd, rv*u.km/u.s, stddev=np.ones_like(rv)*0.1*u.km/u.s)
        self.v1 = v1

    def test_design_matrix(self):
        # P, phi0, ecc, omega
        nlp = np.array([self.P, 0, 0, 0])

        data = self.data['binary']
        params = JokerParams(P_min=8*u.day, P_max=1024*u.day)
        A = design_matrix(nlp, data, params)
        assert A.shape == (2, len(data)) # K, v0
        assert np.allclose(A[1], 1)

        trend = PolynomialVelocityTrend(n_terms=2)
        params = JokerParams(P_min=8*u.day, P_max=1024*u.day, trends=trend)
        A = design_matrix(nlp, data, params)
        assert A.shape == (3, len(data)) # K, v0, v1
        assert np.allclose(A[1], 1)
        assert np.allclose(A[2], data._t_bmjd)
