# Standard library
from os.path import join

# Third-party
import corner
from astropy import log as logger
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Project
from thejoker import Paths
paths = Paths()
from thejoker.data import RVData
from thejoker.util import quantity_from_hdf5
from thejoker.celestialmechanics import OrbitalParams, SimulatedRVOrbit, rv_from_elements
from thejoker.plot import plot_rv_curves, plot_corner, _truth_color

plt.style.use('../thejoker/thejoker.mplstyle')

def make_the_figure(data, pars, samples, truth_pars=None, rv_unit=u.km/u.s,
                    n_plot_curves=128, title='',
                    y_name='P', x1_name='ecc', x2_name='K',
                    rv_lim=None, y_lim=None, x1_lim=None, x2_lim=None,
                    units=None):

    index_map = dict([(key, i) for i,key in enumerate(pars._name_to_unit.keys())])
    i_y = index_map[y_name]
    i_x1 = index_map[x1_name]
    i_x2 = index_map[x2_name]

    # custom units to plot with
    if units is None:
        units = pars._name_to_unit.copy()

    fig = plt.figure(figsize=(8,6.5))

    gs = gridspec.GridSpec(2, 2)

    # First, plot the RV curves on the top axis
    ax = fig.add_subplot(gs[0, :])
    ax.set_title(title)

    dmjd = data.t.mjd.max() - data.t.mjd.min()
    t_grid = np.linspace(data.t.mjd.min() - 0.25*dmjd,
                         data.t.mjd.max() + 0.25*dmjd,
                         1024)

    plot_rv_curves(pars[:128], t_grid, rv_unit=rv_unit,
                   ax=ax, plot_kwargs={'color': '#888888', 'zorder': -100, 'marker': ''})

    data.plot(ax=ax, rv_unit=rv_unit, ecolor='k', markersize=3,
              elinewidth=1, alpha=1., zorder=100)
    ax.set_xlim(t_grid.min(), t_grid.max())

    if rv_lim is None:
        rv_lim = (truth_pars.v0.to(rv_unit).value - 20, truth_pars.v0.to(rv_unit).value + 20)
    ax.set_ylim(rv_lim)

    # Projections of the posterior samples:
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    style = dict(alpha=0.25, marker='.', linestyle='none', zorder=-100, color='#888888')
    ax2.plot(samples[:,i_x1], samples[:,i_y], **style)
    ax3.plot(samples[:,i_x2], samples[:,i_y], **style)

    # -------------------------------------------------------------
    # now plot the truth:
    truth_rv = truth_pars.rv_orbit(0).generate_rv_curve(t_grid)
    truth_vec = truth_pars.pack(plot_transform=True, units=units)[0]
    ax.plot(t_grid, truth_rv.to(rv_unit).value, linestyle='--',
            marker='', linewidth=1, alpha=0.8, color=_truth_color)
    ax2.scatter(truth_vec[i_x1], truth_vec[i_y], marker='+', color=_truth_color, s=40, alpha=0.8)
    ax3.scatter(truth_vec[i_x2], truth_vec[i_y], marker='+', color=_truth_color, s=40, alpha=0.8)

    if y_lim is None:
        dy = samples[:,i_y].max() - samples[:,i_y].min()
        y_lim = (samples[:,i_y].min() - dy*0.05,
                 samples[:,i_y].max() + dy*0.05)

    if x1_lim is None:
        dx1 = samples[:,i_x1].max() - samples[:,i_x1].min()
        x1_lim = (samples[:,i_x1].min() - dx1*0.05,
                  samples[:,i_x1].max() + dx1*0.05)

    if x2_lim is None:
        dx2 = samples[:,i_x2].max() - samples[:,i_x2].min()
        x2_lim = (samples[:,i_x2].min() - dx2*0.05,
                  samples[:,i_x2].max() + dx2*0.05)

    ax2.set_xlim(x1_lim)
    ax2.set_ylim(y_lim)
    ax3.set_xlim(x2_lim)
    ax3.set_ylim(y_lim)
    ax3.set_yticklabels([])

    ax2.set_xlabel(pars._latex_labels[i_x1])
    ax2.set_ylabel(pars._latex_labels[i_y])
    ax3.set_xlabel(pars._latex_labels[i_x2])

    for ax in fig.axes:
        ax.set_rasterization_zorder(-1)

    fig.tight_layout()

    fig.subplots_adjust(hspace=0.5)

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
                x1_lim=(5, 35), x2_lim=(-0.025, 1.025))

    # Make RV curve + 2 projections figures
    fig1 = make_the_figure(data, pars1, samples1, truth_pars=truth_pars,
                           y_name='P', x1_name='K', x2_name='ecc',
                           title="Experiment1 (a)", units=units, **lims)

    fig2 = make_the_figure(data, pars2[128:], samples2, truth_pars=truth_pars,
                           y_name='P', x1_name='K', x2_name='ecc',
                           title="Experiment1 (b)", units=units, **lims)

    fig1.savefig(join(paths.figures, 'exp1-rv-curves-a.pdf'), dpi=128)
    fig2.savefig(join(paths.figures, 'exp1-rv-curves-b.pdf'), dpi=128)

    # Make a corner plot of all samples
    _med_v0 = np.median(samples1[:,-1])
    _mad_v0 = np.median(np.abs(samples1[:,-1] - _med_v0))
    ranges = [lims['y_lim'], lims['x2_lim'], (0,360), (0,360), (-3., 2.5),
              lims['x1_lim'], (_med_v0 - 5*_mad_v0, _med_v0 + 5*_mad_v0)]
    labels = [r'$\ln \left(\frac{P}{\rm day}\right)$', '$e$', r'$\omega$ [deg]', r'$\phi_0$ [deg]',
              r'$\ln \left(\frac{s}{\rm m\,s^{-1}}\right)$', r'$K$ [km s$^{-1}$]', '$v_0$ [km s$^{-1}$]']

    corner_style = dict(alpha=0.75, truth_color=_truth_color, data_kwargs=dict(markersize=3.),
                        plot_contours=False, plot_density=False, bins=32, color='#666666')

    # remove jitter from top plots
    s_idx = 4
    fig1 = corner.corner(np.delete(samples1, s_idx, axis=1), range=np.delete(ranges, s_idx, axis=0),
                         truths=np.delete(truth_vec, s_idx), labels=np.delete(labels, s_idx),
                         **corner_style)
    fig1.savefig(join(paths.figures, 'exp1-corner-a.pdf'), dpi=128)

    fig2 = corner.corner(samples2, range=ranges, truths=truth_vec, labels=labels, **corner_style)
    fig2.savefig(join(paths.figures, 'exp1-corner-b.pdf'), dpi=128)

if __name__ == '__main__':
    main()
