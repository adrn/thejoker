# Third-party
import astropy.units as u
import numpy as np

# Package
from ..likelihood import design_matrix, tensor_vector_scalar, marginal_ln_likelihood

from .helpers import FakeData

class TestLikelihood(object):

    # TODO: repeated code
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

        true_p = [self.truths['triple']['K'].value, self.fd.v0.value, self.fd.v1.value]
        assert np.allclose(p, true_p, rtol=1e-2)

    def test_marginal_ln_likelihood_P(self):
        """
        Check that the true period is the maximum likelihood period
        """
        data = self.data['circ_binary']
        joker_params = self.joker_params['circ_binary']
        true_nlp = self.truths_to_nlp(self.truths['circ_binary'])
        # print('ln_like', marginal_ln_likelihood(true_nlp, data, joker_params))

        vals = np.linspace(true_nlp[0]-1., true_nlp[0]+1, 4096)
        lls = np.zeros_like(vals)
        for i,val in enumerate(vals):
            nlp = true_nlp.copy()
            nlp[0] = val
            lls[i] = marginal_ln_likelihood(nlp, data, joker_params)

        assert np.allclose(true_nlp[0], vals[lls.argmax()], rtol=1E-4)

    def test_marginal_ln_likelihood_P_samples(self):
        """
        This test is a little crazy. We generate samples in P, phi0
        """

        data = self.data['circ_binary']
        joker_params = self.joker_params['circ_binary']
        true_nlp = self.truths_to_nlp(self.truths['circ_binary'])

        n_samples = 16384
        P = np.random.uniform(true_nlp[0] - 2., true_nlp[0] + 2., size=n_samples)
        phi0 = np.random.uniform(0, 2*np.pi, size=n_samples)

        lls = np.zeros_like(P)
        n_neg = 0
        for i in range(n_samples):
            nlp = true_nlp.copy()
            nlp[0] = P[i]
            nlp[1] = phi0[i]
            lls[i] = marginal_ln_likelihood(nlp, data, joker_params)

            A = design_matrix(nlp, data, joker_params)
            _,p,_ = tensor_vector_scalar(A, data.ivar.value, data.rv.value)
            if p[0] < 0:
                n_neg += 1

        # rejection sample using the marginal likelihood
        uu = np.random.uniform(size=n_samples)
        idx = uu < np.exp(lls - lls.max())
        print("{} good samples".format(idx.sum()))

        assert idx.sum() > 1
        assert np.std(P[idx]) < 1.

        # import matplotlib.pyplot as plt

        # plt.figure()
        # bins = np.linspace(true_nlp[0] - 2., true_nlp[0] + 2., 32)
        # plt.hist(P, bins=bins, alpha=0.25, normed=True)
        # plt.hist(P[idx], bins=bins, alpha=0.4, normed=True)
        # plt.axvline(true_nlp[0], color='r')

        # plt.figure()
        # bins = np.linspace(0, 2*np.pi, 32)
        # plt.hist(phi0, bins=bins, alpha=0.25, normed=True)
        # plt.hist(phi0[idx], bins=bins, alpha=0.4, normed=True)
        # plt.axvline(true_nlp[1], color='r')

        # plt.show()
