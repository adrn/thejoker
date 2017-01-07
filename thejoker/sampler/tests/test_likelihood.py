# Third-party
import astropy.units as u
import numpy as np

# Package
from ...data import RVData
from ..params import JokerParams, PolynomialVelocityTrend
from ..likelihood import design_matrix, tensor_vector_scalar, marginal_ln_likelihood
from ...celestialmechanics import rv_from_elements

from .helpers import FakeData

class TestLikelihood(object):

    def truths_to_nlp(self, truths):
        # P, phi0, ecc, omega
        P = truths['P'].to(u.day).value
        phi0 = truths['phi0'].to(u.radian).value
        ecc = truths['ecc']
        omega = truths['omega'].to(u.radian).value
        return np.array([P, phi0, ecc, omega, 0.])

    def setup(self):
        d = FakeData()
        self.data = d.data
        self.joker_params = d.joker_params
        self.truths = d.truths

    def test_design_matrix(self):

        data = self.data['binary']
        nlp = self.truths_to_nlp(self.truths['binary'])
        A = design_matrix(nlp, data, self.joker_params['binary'])
        assert A.shape == (len(data), 2) # K, v0
        assert np.allclose(A[:,1], 1)

        nlp = self.truths_to_nlp(self.truths['triple'])
        A = design_matrix(nlp, data, self.joker_params['triple'])
        assert A.shape == (len(data), 3) # K, v0, v1
        assert np.allclose(A[:,1], 1)
        assert np.allclose(A[:,2], data._t_bmjd)

    def test_tensor_vector_scalar(self):

        data = self.data['binary']
        nlp = self.truths_to_nlp(self.truths['binary'])
        A = design_matrix(nlp, data, self.joker_params['binary'])
        ATCinvA, p, chi2 = tensor_vector_scalar(A, data.ivar.value,
                                                data.rv.value)
        print(p)
        print(self.truths['binary']['K'])
        print(self.truths['binary']['trends'].coeffs)
        return
        # assert np.allclose(p, self.truths['binary'], rtol=1e-2)

        # --

        data = self.data['triple']
        nlp = self.truths_to_nlp(self.truths['triple'])
        A = design_matrix(nlp, data, self.joker_params['triple'])
        ATCinvA, p, chi2 = tensor_vector_scalar(A, data.ivar.value,
                                                data.rv.value)
        assert np.allclose(p, self.truths['triple'], rtol=1e-2)

    def test_marginal_ln_likelihood_period(self):
        # make a grid over periods with other pars fixed at truth and make sure
        #   highest likelihood period is close to truth
        # TODO: should grid over all of the parameters!

        data = self.data['binary']
        nlp = self.truths_to_nlp(self.truths['binary'])
        true_P = nlp[0]
        ll = marginal_ln_likelihood(nlp, data, self.joker_params['binary'])

        vals = np.linspace(52, 54.5, 256)
        lls = np.zeros_like(vals)
        for i,val in enumerate(vals):
            _nlp = nlp.copy()
            _nlp[0] = val
            lls[i] = marginal_ln_likelihood(_nlp, data,
                                            self.joker_params['binary'])
        best_P = vals[np.argmax(lls)]
        assert abs(best_P - true_P) < abs(vals[1]-vals[0])

        # --
        data = self.data['triple']
        nlp = self.truths_to_nlp(self.truths['triple'])
        true_P = nlp[0]
        ll = marginal_ln_likelihood(nlp, data, self.joker_params['triple'])

        vals = np.linspace(52, 54.5, 256)
        lls = np.zeros_like(vals)
        for i,val in enumerate(vals):
            _nlp = nlp.copy()
            _nlp[0] = val
            lls[i] = marginal_ln_likelihood(_nlp, data,
                                            self.joker_params['triple'])
        best_P = vals[np.argmax(lls)]
        assert abs(best_P - true_P) < abs(vals[1]-vals[0])


