# Third-party
import astropy.units as u
import numpy as np

# Package
from ...data import RVData
from ..params import JokerParams, PolynomialVelocityTrend
from ..likelihood import design_matrix, tensor_vector_scalar, marginal_ln_likelihood
from ...celestialmechanics import rv_from_elements

class TestLikelihood(object):

    def setup(self):
        self.data = dict()
        self.params = dict()
        self.truths = dict()

        EPOCH = 56512.1492
        mjd = np.linspace(0, 300., 128) + EPOCH
        P = 53.2352 # u.day
        v0 = 100. # u.km/u.s
        K = 10. # u.km/u.s
        zdot = rv_from_elements(times=mjd, P=P, K=K, e=0, omega=0, phi0=0)
        err = np.random.uniform(0.01, 0.02, size=mjd.shape) # u.km/u.s
        eps = np.random.normal(0, err)
        rv = K*zdot + v0 + eps
        self.data['binary'] = RVData(mjd, rv*u.km/u.s, stddev=err*u.km/u.s, t_offset=0.)
        self.params['binary'] = JokerParams(P_min=8*u.day, P_max=1024*u.day)
        self.truths['binary'] = np.array([K, v0])
        self.P = P

        v1 = 1E-3 # u.km/u.s/u.day
        rv = K*zdot + v0 + v1*mjd + eps
        self.data['triple'] = RVData(mjd, rv*u.km/u.s, stddev=err*u.km/u.s, t_offset=0.)
        trend = PolynomialVelocityTrend(n_terms=2)
        self.params['triple'] = JokerParams(P_min=8*u.day, P_max=1024*u.day, trends=trend)
        self.truths['triple'] = np.array([K, v0, v1])
        self.v1 = v1

    def test_design_matrix(self):
        # P, phi0, ecc, omega
        nlp = np.array([self.P, 0, 0, 0])

        data = self.data['binary']
        A = design_matrix(nlp, data, self.params['binary'])
        assert A.shape == (len(data), 2) # K, v0
        assert np.allclose(A[:,1], 1)

        A = design_matrix(nlp, data, self.params['triple'])
        assert A.shape == (len(data), 3) # K, v0, v1
        assert np.allclose(A[:,1], 1)
        assert np.allclose(A[:,2], data._t_bmjd)

    def test_tensor_vector_scalar(self):
        # P, phi0, ecc, omega
        nlp = np.array([self.P, 0, 0, 0])

        data = self.data['binary']
        A = design_matrix(nlp, data, self.params['binary'])
        ATCinvA, p, chi2 = tensor_vector_scalar(A, data.ivar.value,
                                                data.rv.value)
        assert np.allclose(p, self.truths['binary'], rtol=1e-2)

        data = self.data['triple']
        A = design_matrix(nlp, data, self.params['triple'])
        ATCinvA, p, chi2 = tensor_vector_scalar(A, data.ivar.value,
                                                data.rv.value)
        assert np.allclose(p, self.truths['triple'], rtol=1e-2)

    def test_marginal_ln_likelihood_period(self):
        # make a grid over periods with other pars fixed at truth and make sure
        #   highest likelihood period is close to truth
        # TODO: should grid over all of the parameters!

        # P, phi0, ecc, omega, jitter
        nlp = np.array([self.P, 0., 0, 0, 0.])

        data = self.data['binary']
        ll = marginal_ln_likelihood(nlp, data, self.params['binary'])

        vals = np.linspace(52, 54.5, 256)
        lls = np.zeros_like(vals)
        for i,val in enumerate(vals):
            nlp = np.array([val, 0., 0, 0, 0.])
            lls[i] = marginal_ln_likelihood(nlp, data, self.params['binary'])
        best_P = vals[np.argmax(lls)]
        assert abs(best_P - self.P) < abs(vals[1]-vals[0])

        # --
        data = self.data['triple']
        ll = marginal_ln_likelihood(nlp, data, self.params['triple'])

        vals = np.linspace(52, 54.5, 256)
        lls = np.zeros_like(vals)
        for i,val in enumerate(vals):
            nlp = np.array([val, 0., 0, 0, 0.])
            lls[i] = marginal_ln_likelihood(nlp, data, self.params['triple'])
        best_P = vals[np.argmax(lls)]
        assert abs(best_P - self.P) < abs(vals[1]-vals[0])


