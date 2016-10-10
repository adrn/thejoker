# Standard library
from os.path import join

# Third-party
import corner
from astropy import log as logger
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker.data import RVData
from thejoker.util import quantity_from_hdf5
from thejoker.celestialmechanics import OrbitalParams, SimulatedRVOrbit, rv_from_elements
from thejoker.plot import plot_rv_curves, plot_corner, _truth_color

plt.style.use('../thejoker/thejoker.mplstyle')

def make_rv_curve_figure(data, all_pars, truth_pars=None, rv_unit=u.km/u.s,
                         n_plot_curves=128, titles=None,
                         rv_lim=None, units=None):

    if isinstance(all_pars, OrbitalParams):
        all_pars = [all_pars]
    n_pars = len(all_pars)

    # custom units to plot with
    if units is None:
        units = all_pars[0]._name_to_unit.copy()

    # take t_grid from data with max extent
    min_t = data.t.mjd.min()
    max_t = data.t.mjd.max()

    dmjd = max_t - min_t
    t_grid = np.linspace(min_t - 0.25*dmjd,
                         max_t + 0.25*dmjd,
                         1024)

    fig,axes = plt.subplots(n_pars, 1, figsize=(6.5, 3*n_pars), sharex=True, sharey=True)

    for i in range(n_pars):
        ax = axes[i]
        pars = all_pars[i]

        if titles is not None:
            ax.set_title(titles[i])

        plot_rv_curves(pars[:n_plot_curves], t_grid, rv_unit=rv_unit, add_labels=False,
                       ax=ax, plot_kwargs={'color': '#888888', 'zorder': -100, 'marker': ''})

        data.plot(ax=ax, rv_unit=rv_unit, ecolor='k', markersize=3,
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

        unit_label = ' [{}]'.format(rv_unit._repr_latex_())
        ax.set_ylabel('RV{}'.format(unit_label))

    ax.set_xlabel('BMJD')

    for ax in fig.axes:
        ax.set_rasterization_zorder(-1)

    return fig

def main():
    data_filename = "../data/experiment1.h5"

    # plot units
    units = OrbitalParams._name_to_unit.copy()
    units['K'] = u.km/u.s
    units['v0'] = u.km/u.s
    units['omega'] = u.degree
    units['phi0'] = u.degree

    # read the data
    with h5py.File(data_filename, 'r') as f:
        data = RVData.from_hdf5(f)
        truth_pars = OrbitalParams.from_hdf5(f['truth'])
        truth_vec = truth_pars.pack(plot_transform=True, units=units)[0]

    # read the samples from fixing the jitter
    with h5py.File(join(paths.cache, 'experiment1-fixed-jitter.h5'), 'r') as g:
        pars1 = OrbitalParams.from_hdf5(g)
        samples1 = pars1.pack(plot_transform=True, units=units)

    # read the samples from sampling over the jitter
    with h5py.File(join(paths.cache, 'experiment1-sample-jitter.h5'), 'r') as g:
        pars2 = OrbitalParams.from_hdf5(g)
        samples2 = pars2.pack(plot_transform=True, units=units)

    lims = dict(rv_lim=(21, 67), y_lim=(2.55, 6.45),
                x1_lim=(5, 25), x2_lim=(-0.025, 1.025))

    # Make RV curve + 2 projections figures
    fig = make_rv_curve_figure(data, [pars1, pars2], truth_pars=truth_pars,
                               units=units, rv_lim=(21, 67))
    fig.axes[0].set_title("Experiment 1")
    fig.tight_layout()
    fig.savefig(join(paths.figures, 'exp1-rv-curves.pdf'), dpi=128)

    # Make a corner plot of all samples
    _med_v0 = np.median(samples1[:,-1])
    _mad_v0 = np.median(np.abs(samples1[:,-1] - _med_v0))
    ranges = [lims['y_lim'], lims['x2_lim'], (0,360), (0,360), (-3.5, 3.5),
              lims['x1_lim'], (_med_v0 - 5*_mad_v0, _med_v0 + 5*_mad_v0)]
    labels = [r'$\ln \left(\frac{P}{\rm day}\right)$', '$e$', r'$\omega$ [deg]', r'$\phi_0$ [deg]',
              r'$\ln \left(\frac{s}{\rm m\,s^{-1}}\right)$', r'$K$ [km s$^{-1}$]', '$v_0$ [km s$^{-1}$]']

    corner_style = dict(truth_color=_truth_color, data_kwargs=dict(alpha=0.5, markersize=2.),
                        plot_contours=False, plot_density=False, bins=32, color='#555555')

    # remove jitter from top plots
    s_idx = 4
    fig1 = corner.corner(np.delete(samples1, s_idx, axis=1), range=np.delete(ranges, s_idx, axis=0),
                         truths=np.delete(truth_vec, s_idx), labels=np.delete(labels, s_idx),
                         **corner_style)
    fig1.subplots_adjust(left=0.08, bottom=0.08)
    fig1.suptitle("Experiment 1a: fixed jitter", fontsize=26)
    fig1.savefig(join(paths.figures, 'exp1-corner-a.pdf'), dpi=128)

    fig2 = corner.corner(samples2, range=ranges, truths=truth_vec, labels=labels, **corner_style)
    fig2.subplots_adjust(left=0.08, bottom=0.08)
    fig2.suptitle("Experiment 1b: sample jitter", fontsize=26)
    fig2.savefig(join(paths.figures, 'exp1-corner-b.pdf'), dpi=128)

if __name__ == '__main__':
    main()
