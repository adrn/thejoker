# Third-party
import astropy.units as u
import numpy as np

# Package
from ..likelihood import (design_matrix, likelihood_worker,
                          marginal_ln_likelihood)

from .helpers import FakeData, Lambda


class TestLikelihood(object):

    # TODO: repeated code
    def truths_to_nlp(self, truths):
        # P, M0, ecc, omega
        P = truths['P'].to(u.day).value
        M0 = truths['M0'].to(u.radian).value
        ecc = truths['e']
        omega = truths['omega'].to(u.radian).value
        return np.array([P, M0, ecc, omega, 0.])

    def setup(self):
        d = FakeData()
        self.datasets = d.datasets
        self.params = d.params
        self.truths = d.truths

    def test_design_matrix(self):

        data = self.datasets['binary']
        nlp = self.truths_to_nlp(self.truths['binary'])
        A = design_matrix(nlp, data, self.params['binary'])
        assert A.shape == (len(data), 2)  # K, v0
        assert np.allclose(A[:, 1], 1)

        # TODO: triple disabled
        # nlp = self.truths_to_nlp(self.truths['triple'])
        # A = design_matrix(nlp, data, self.params['triple'])
        # assert A.shape == (len(data), 3) # K, v0, v1
        # assert np.allclose(A[:,1], 1)
        # assert np.allclose(A[:,2], data._t_bmjd)

    def test_likelihood_worker(self):

        data = self.datasets['binary']
        nlp = self.truths_to_nlp(self.truths['binary'])
        M = design_matrix(nlp, data, self.params['binary'])
        ll, b, B, a, Ainv = likelihood_worker(data.rv.value, data.ivar.value, M,
                                              mu=np.zeros(2),
                                              Lambda=Lambda,
                                              make_aAinv=True)

        assert np.array(ll).shape == ()
        assert a.shape == (2, )
        assert Ainv.shape == (2, 2)
        assert b.shape == (len(data), )
        assert B.shape == (len(data), len(data))

        true_p = [self.truths['binary']['K'].value,
                  self.truths['binary']['v0'].value]
        assert np.allclose(a, true_p, rtol=1e-2)

    def test_marginal_ln_likelihood_P(self):
        """
        Check that the true period is the maximum likelihood period
        """
        data = self.datasets['circ_binary']
        params = self.params['circ_binary']
        true_nlp = self.truths_to_nlp(self.truths['circ_binary'])
        # print('ln_like', marginal_ln_likelihood(true_nlp, data, params))

        vals = np.linspace(true_nlp[0]-1., true_nlp[0]+1, 4096)
        lls = np.zeros_like(vals)
        for i, val in enumerate(vals):
            nlp = true_nlp.copy()
            nlp[0] = val
            lls[i] = marginal_ln_likelihood(nlp, data, params)

        assert np.allclose(true_nlp[0], vals[lls.argmax()], rtol=1E-4)

    def test_marginal_ln_likelihood_P_samples(self):
        """
        This test is a little crazy. We generate samples in P, M0
        """

        data = self.datasets['circ_binary']
        params = self.params['circ_binary']
        true_nlp = self.truths_to_nlp(self.truths['circ_binary'])

        n_samples = 16384
        P = np.random.uniform(true_nlp[0] - 2., true_nlp[0] + 2., size=n_samples)
        M0 = np.random.uniform(0, 2*np.pi, size=n_samples)

        lls = np.zeros_like(P)
        n_neg = 0
        for i in range(n_samples):
            nlp = true_nlp.copy()
            nlp[0] = P[i]
            nlp[1] = M0[i]
            lls[i] = marginal_ln_likelihood(nlp, data, params)

            M = design_matrix(nlp, data, params)
            *_, a, Ainv = likelihood_worker(data.rv.value, data.ivar.value,
                                            M, mu=np.zeros(2), Lambda=Lambda,
                                            make_aAinv=True)
            if a[0] < 0:
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
        # plt.hist(M0, bins=bins, alpha=0.25, normed=True)
        # plt.hist(M0[idx], bins=bins, alpha=0.4, normed=True)
        # plt.axvline(true_nlp[1], color='r')

        # plt.show()
