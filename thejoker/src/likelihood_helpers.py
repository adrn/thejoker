# Third-party
import numpy as np


def get_constant_term_design_matrix(data, ids=None):
    """Construct the portion of the design matrix relevant for the linear
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
