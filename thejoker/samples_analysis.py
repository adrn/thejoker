# Standard library

# Third-party
import astropy.units as u
import numpy as np

__all__ = ['MAP_sample', 'is_P_unimodal', 'is_P_Kmodal',
           'max_phase_gap', 'phase_coverage', 'periods_spanned',
           'phase_coverage_per_period']


def MAP_sample(samples, return_index=False):
    """Return the maximum a posteriori sample.

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`

    """
    if ('ln_prior' not in samples.tbl.colnames
            or 'ln_likelihood' not in samples.tbl.colnames):
        raise ValueError("You must pass in samples that have prior and "
                         "likelihood information stored; use return_logprobs="
                         "True when generating the samples.")

    ln_post = samples['ln_prior'] + samples['ln_likelihood']
    idx = np.argmax(ln_post)

    if return_index:
        return samples[idx], idx
    else:
        return samples[idx]


def is_P_unimodal(samples, data):
    """
    Check whether the samples returned are within one period mode.

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`

    Returns
    -------
    is_unimodal : bool
    """

    P_samples = samples['P'].to(u.day).value
    P_min = np.min(P_samples)
    T = np.ptp(data.t.tcb.mjd)
    delta = 4*P_min**2 / (2*np.pi*T)

    return np.ptp(P_samples) < delta


def is_P_Kmodal(samples, data, n_clusters=2):
    """
    Experimental!

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    n_clusters : int (optional)

    Returns
    -------
    is_Kmodal : bool
    mode_means : array
    n_per_mode : array

    """
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=n_clusters)

    lnP = np.log(samples['P'].value).reshape(-1, 1)
    y = clf.fit_predict(lnP)

    unimodals = []
    means = []
    n_per_mode = []
    for j in np.unique(y):
        sub_samples = samples[y == j]
        if len(sub_samples) == 1:
            unimodals.append(True)
            means.append(sub_samples)
        else:
            unimodals.append(is_P_unimodal(sub_samples, data))
            means.append(MAP_sample(sub_samples)['P'])
        n_per_mode.append((y == j).sum())

    return all(unimodals), u.Quantity(means), np.array(n_per_mode)


def max_phase_gap(sample, data):
    """Based on the MPG statistic defined here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu7var/sssec_cu7var_validation_sos_sl/ssec_cu7var_sos_sl_qa.html

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    phase = np.sort(data.phase(sample['P']))
    phase = np.concatenate((phase, phase))
    return (phase[1:] - phase[:-1]).max()


def phase_coverage(sample, data, n_bins=10):
    """Based on the PC statistic defined here:
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_analysis/chap_cu7var/sssec_cu7var_validation_sos_sl/ssec_cu7var_sos_sl_qa.html

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    P = sample['P']
    H, _ = np.histogram(data.phase(P),
                        bins=np.linspace(0, 1, n_bins+1))
    return (H > 0).sum() / n_bins


def periods_spanned(sample, data):
    """Compute the number of periods spanned by the data

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    P = sample['P']
    T = data.t.jd.max() - data.t.jd.min()
    return T / P.to_value(u.day)


def phase_coverage_per_period(sample, data):
    """The maximum number of data points within a period.

    Parameters
    ----------
    sample : `~thejoker.JokerSamples`
    data : `~thejoker.RVData`
    """
    P = sample['P']
    dt = (data.t - data.t_ref).to(u.day)
    phase = dt / P
    H1, _ = np.histogram(phase, bins=np.arange(0, phase.max()+1, 1))
    H2, _ = np.histogram(phase, bins=np.arange(-0.5, phase.max()+1, 1))
    return max(H1.max(), H2.max())
