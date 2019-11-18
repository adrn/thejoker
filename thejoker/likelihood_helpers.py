# Third-party
import numpy as np


def get_constant_term_design_matrix(data, ids=None):
    """
    Construct the portion of the design matrix relevant for the linear
    parameters of The Joker beyond the amplitude, ``K``.
    """

    if ids is None:
        ids = np.zeros(len(data), dtype=int)
    ids = np.array(ids)

    unq_ids = np.unique(ids)
    constant_part = np.zeros((len(data), len(unq_ids)))

    constant_part[:, 0] = 1.
    for id_ in unq_ids[1:]:
        constant_part[ids == id_, 1] = 1.

    return constant_part


def get_trend_design_matrix(data, ids, poly_trend):
    """
    TODO
    """
    # Combine design matrix for constant term, which may contain columns for
    # sampling over v0 offsets, with the rest of the long-term trend columns
    const_M = get_constant_term_design_matrix(data, ids)
    dt = data._t_bmjd - data._t0_bmjd
    trend_M = np.vander(dt, N=poly_trend, increasing=True)[:, 1:]
    return np.hstack((const_M, trend_M))
