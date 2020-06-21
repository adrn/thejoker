import numpy as np

from ..data import RVData
from ..data_helpers import validate_prepare_data
from .test_data import get_valid_input
from ..utils import DEFAULT_RNG


def test_design_matrix():
    # implicitly tests get_constant_term_design_matrix
    rnd = DEFAULT_RNG(42)

    # Set up mulitple valid data objects:
    ndata1 = 8
    ndata2 = 4
    ndata3 = 3

    _, raw1 = get_valid_input(rnd=rnd, size=ndata1)
    data1 = RVData(raw1['t_obj'], raw1['rv'], raw1['err'])

    _, raw2 = get_valid_input(rnd=rnd, size=ndata2)
    data2 = RVData(raw2['t_obj'], raw2['rv'], raw2['err'])

    _, raw3 = get_valid_input(rnd=rnd, size=ndata3)
    data3 = RVData(raw3['t_obj'], raw3['rv'], raw3['err'])

    data, ids, M = validate_prepare_data([data1, data2, data3], 1, 2)
    assert np.allclose(M[:, 0], 1.)

    idx = np.arange(len(data), dtype=int)
    mask = (idx >= ndata1) & (idx < (ndata1+ndata2))
    assert np.allclose(M[mask, 1], 1.)
    assert np.allclose(M[~mask, 1], 0.)

    mask = (idx >= (ndata1+ndata2)) & (idx < (ndata1+ndata2+ndata3))
    assert np.allclose(M[mask, 2], 1.)
    assert np.allclose(M[~mask, 2], 0.)
