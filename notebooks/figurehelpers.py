# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import corner

# Project
from thejoker import Paths
paths = Paths()
from thejoker.data import RVData
from thejoker.celestialmechanics import OrbitalParams
from thejoker.plot import plot_rv_curves, _truth_color

# plot units
samples_units = OrbitalParams._name_to_unit.copy()
samples_units['K'] = u.km/u.s
samples_units['v0'] = u.km/u.s
samples_units['omega'] = u.degree
samples_units['phi0'] = u.degree

def make_rv_curve_figure(all_data, all_pars, truth_pars=None, rv_unit=u.km/u.s,
                         n_plot_curves=128, titles=None,
                         rv_lim=None, units=None):

    if isinstance(all_pars, OrbitalParams):
        all_pars = [all_pars]
    n_pars = len(all_pars)

    if isinstance(all_data, RVData):
        all_data = [all_data]
    data = all_data[0]

    if len(all_data) == 1:
        only_one_data = True

    elif len(all_data) != len(all_pars):
        raise ValueError("dood.")

    else:
        only_one_data = False

    # custom units to plot with
    if units is None:
        units = all_pars[0]._name_to_unit.copy()

    # take t_grid from data with max extent
    min_t,max_t = (np.inf, -np.inf)
    for data in all_data:
        min_t = min(min_t, data.t.mjd.min())
        max_t = max(max_t, data.t.mjd.max())

    dmjd = max_t - min_t
    t_grid = np.linspace(min_t - 0.25*dmjd,
                         max_t + 0.25*dmjd,
                         1024)

    fig,axes = plt.subplots(n_pars, 1, figsize=(6.5, 3*n_pars), sharex=True, sharey=True)

    for i in range(n_pars):
        ax = axes[i]
        pars = all_pars[i]

        if not only_one_data:
            data = all_data[i]

        if titles is not None:
            ax.set_title(titles[i])

        plot_rv_curves(pars[:n_plot_curves], t_grid, rv_unit=rv_unit, add_labels=False,
                       ax=ax, plot_kwargs={'color': '#aaaaaa', 'zorder': -100, 'marker': ''})

        data.plot(ax=ax, rv_unit=rv_unit, ecolor='k', markersize=4,
                  elinewidth=1, alpha=1., zorder=100)
        ax.set_xlim(t_grid.min(), t_grid.max())

        if rv_lim is None:
            rv_lim = (truth_pars.v0.to(rv_unit).value - 20,
                      truth_pars.v0.to(rv_unit).value + 20)
        ax.set_ylim(rv_lim)

        if truth_pars is not None:
            # now plot the truth:
            truth_rv = truth_pars.rv_orbit(0).generate_rv_curve(t_grid)
            ax.plot(t_grid, truth_rv.to(rv_unit).value, linestyle='--',
                    marker='', linewidth=1, alpha=0.8, color=_truth_color)

        ax.set_ylabel('RV [{}]'.format(rv_unit.to_string(format='latex_inline')))

    ax.set_xlabel('BMJD')

    for ax in fig.axes:
        ax.set_rasterization_zorder(-1)

    return fig

def apw_corner(samples, **corner_kwargs):
    """
    A thin wrapper around `corner.corner` to set defaults differently.
    """

    corner_kw = corner_kwargs.copy()
    corner_kw.setdefault("plot_contours", False)
    corner_kw.setdefault("plot_density", False)
    corner_kw.setdefault("plot_datapoints", True)
    corner_kw.setdefault("data_kwargs", dict(alpha=corner_kw.pop('alpha'),
                                             markersize=corner_kw.pop('markersize')))

    # default truth style
    corner_kw.setdefault("truth_color", _truth_color)
    corner_kw.setdefault("truth_alpha", 0.5)
    corner_kw.setdefault("truth_linestyle", 'solid')

    truth_style = dict()
    truth_style['color'] = corner_kw.pop('truth_color')
    truth_style['alpha'] = corner_kw.pop('truth_alpha')
    truth_style['linestyle'] = corner_kw.pop('truth_linestyle')
    truth_style['marker'] = '' # Don't show markers

    if 'truths' in corner_kw: # draw lines only
        truths = corner_kw.pop('truths')
    else:
        truths = None

    fig = corner.corner(samples, **corner_kw)

    if truths is not None: # draw lines only
        axes = np.array(fig.axes).reshape(samples.shape[1],samples.shape[1])

        for i in range(samples.shape[1]):
            for j in range(samples.shape[1]):
                if i == j:
                    axes[i,j].axvline(truths[j], **truth_style)

                elif i > j:
                    axes[i,j].axvline(truths[j], **truth_style)
                    axes[i,j].axhline(truths[i], **truth_style)

    return fig
