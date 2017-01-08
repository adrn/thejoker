# Third-party
import astropy.units as u
import numpy as np

# Package
from ..likelihood import design_matrix, tensor_vector_scalar, marginal_ln_likelihood

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
        self.fd = d
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
        true_p = [self.truths['binary']['K'].value, self.fd.v0.value]
        assert np.allclose(p, true_p, rtol=1e-2)

        # --

        data = self.data['triple']
        nlp = self.truths_to_nlp(self.truths['triple'])
        A = design_matrix(nlp, data, self.joker_params['triple'])
        ATCinvA, p, chi2 = tensor_vector_scalar(A, data.ivar.value,
                                                data.rv.value)

        true_p = [self.truths['binary']['K'].value, self.fd.v0.value, self.fd.v1.value]
        assert np.allclose(p, true_p, rtol=1e-2)

