from astropy.time import Time
import astropy.units as u
import numpy as np

from .src.fast_likelihood import CJokerSB2Helper
from .samples import JokerSamples
from .likelihood_helpers import get_trend_design_matrix
from .thejoker import TheJoker

__all__ = ['TheJokerSB2', 'JokerSB2Samples']


def validate_prepare_data_sb2(data, poly_trend, t_ref=None):
    """Internal function.

    Used to take an input ``RVData`` instance, or a list/dict of ``RVData``
    instances, and produce concatenated time, RV, and error arrays, along
    with a consistent t_ref.
    """
    from .data import RVData

    # If we've gotten here, data is dict-like:
    rv_unit = None
    t = []
    rv = []
    err = []
    ids = []
    for k in data.keys():
        d = data[k]

        if not isinstance(d, RVData):
            raise TypeError(f"All data must be specified as RVData instances: "
                            f"Object at key '{k}' is a '{type(d)}' instead.")

        if d._has_cov:
            raise NotImplementedError("We currently don't support "
                                      "multi-survey data when a full "
                                      "covariance matrix is specified. "
                                      "Raise an issue in adrn/thejoker if "
                                      "you want this functionality.")

        if rv_unit is None:
            rv_unit = d.rv.unit

        t.append(d.t.tcb.mjd)
        rv.append(d.rv.to_value(rv_unit))
        err.append(d.rv_err.to_value(rv_unit))
        ids.append([k] * len(d))

    t = np.concatenate(t)
    rv = np.concatenate(rv) * rv_unit
    err = np.concatenate(err) * rv_unit
    ids = np.concatenate(ids)

    all_data = RVData(t=Time(t, format='mjd', scale='tcb'),
                      rv=rv, rv_err=err,
                      t_ref=t_ref, sort=False)
    K_M = np.zeros((len(all_data), 2))
    K_M[ids == '1', 0] = 1.
    K_M[ids == '2', 1] = -1.
    trend_M = get_trend_design_matrix(all_data, ids=None, poly_trend=poly_trend)

    return all_data, ids, np.hstack((K_M, trend_M))


class JokerSB2Samples(JokerSamples):

    def __init__(self, samples=None, t_ref=None, poly_trend=None,
                 **kwargs):
        """
        A dictionary-like object for storing prior or posterior samples from
        The Joker, with some extra functionality.

        Parameters
        ----------
        samples : `~astropy.table.QTable`, table-like (optional)
            The samples data as an Astropy table object, or something
            convertable to an Astropy table (e.g., a dictionary with
            `~astropy.units.Quantity` object values). This is optional because
            the samples data can be added later by setting keys on the resulting
            instance.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. See
            the docstring for `thejoker.JokerPrior` for more details.
        t_ref : `astropy.time.Time`, numeric (optional)
            The reference time for the orbital parameters.
        **kwargs
            Additional keyword arguments are stored internally as metadata.
        """
        super().__init__(t_ref=t_ref, poly_trend=poly_trend, **kwargs)

        self._valid_units['K1'] = self._valid_units.pop('K')
        self._valid_units['K2'] = self._valid_units.get('K1')

        if samples is not None:
            for col in samples.colnames:
                self[col] = samples[col]

    def __repr__(self):
        return (f'<JokerSB2Samples [{", ".join(self.par_names)}] '
                f'({len(self)} samples)>')

    @property
    def primary(self):
        new_samples = JokerSamples(t_ref=self.t_ref, poly_trend=self.poly_trend)
        new_samples['K'] = self['K1']
        for name in new_samples._valid_units.keys():
            if name == 'K' or name not in self.tbl.colnames:
                continue

            else:
                new_samples[name] = self[name]
        return new_samples

    @property
    def secondary(self):
        new_samples = JokerSamples(t_ref=self.t_ref, poly_trend=self.poly_trend)
        new_samples['K'] = self['K2']
        for name in new_samples._valid_units.keys():
            if name == 'K' or name not in self.tbl.colnames:
                continue

            elif name == 'omega':
                new_samples[name] = self[name] - 180*u.deg

            else:
                new_samples[name] = self[name]
        return new_samples

    def get_orbit(self, index=None, which='1', **kwargs):
        pass


class TheJokerSB2(TheJoker):
    _samples_cls = JokerSB2Samples

    def _make_joker_helper(self, data):
        assert len(data) == 2
        assert '1' in data.keys() and '2' in data.keys()
        all_data, ids, M = validate_prepare_data_sb2(
            data, self.prior.poly_trend, t_ref=data['1'].t_ref)

        joker_helper = CJokerSB2Helper(all_data, self.prior, M)
        return joker_helper
