# Third-party
import astropy.time as atime
from astropy import log as logger
import astropy.units as u
import corner
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['plot_rv_curves', 'plot_corner']

_truth_color = '#006837'
_prev_result_color = '#2166AC'

def plot_rv_curves(orbital_pars, t_grid, rv_unit=None, data=None,
                   ax=None, plot_kwargs=dict(), data_plot_kwargs=dict(),
                   add_labels=True):
    """
    Plot radial velocity curves for the input set of orbital parameters
    over the input grid of times.

    Parameters
    ----------
    orbital_pars : `~thejoker.celestialmechanics.OrbitalParams`
    t_grid : array_like, `~astropy.time.Time`
        Array of times. Either in BMJD or as an Astropy time object.
    rv_unit : `~astropy.units.UnitBase` (optional)
        The units to use when plotting RV's.
    data : `~thejoker.data.RVData` (optional)
        Over-plot the data as well.
    ax : `~matplotlib.Axes` (optional)
        A matplotlib axes object to plot on to. If not specified, will
        create a new figure and plot on that.
    plot_kwargs : dict
        Passed to `matplotlib.pyplot.plot()`.
    data_plot_kwargs : dict
        Passed to `thejoker.data.RVData.plot()`.
    add_labels : bool (optional)
        Add labels to the axes or not.

    Returns
    -------
    fig : `~matplotlib.Figure`

    """

    if ax is None:
        fig,ax = plt.subplots(1,1)
    else:
        fig = ax.figure

    n_samples = len(orbital_pars)

    if n_samples > 128:
        logger.warning("Plotting more than 128 radial velocity curves ({}) -- "
                       "are you sure you want to do this?".format(n_samples))

    if isinstance(t_grid, atime.Time):
        t_grid = t_grid.tcb.mjd

    # scale the transparency of the lines
    Q = 4. # HACK
    line_alpha = 0.05 + Q / (n_samples + Q)

    if rv_unit is None:
        rv_unit = u.km/u.s

    # default plotting style
    style = plot_kwargs.copy()
    style.setdefault('linestyle', '-')
    style.setdefault('alpha', line_alpha)
    style.setdefault('marker', None)
    style.setdefault('color', '#555555')

    # plot orbits over the data
    model_rv = np.zeros((n_samples, len(t_grid)))
    for i in range(n_samples):
        orbit = orbital_pars.rv_orbit(i)
        model_rv[i] = orbit.generate_rv_curve(t_grid).to(rv_unit).value
    ax.plot(t_grid, model_rv.T, **style)

    if data is not None:
        data_style = data_plot_kwargs.copy()
        data_style.setdefault('rv_unit', rv_unit)
        data_style.setdefault('markersize', 5.)

        if data_style['rv_unit'] != rv_unit:
            raise u.UnitsError("Data plot units don't match rv_unit!")

        data.plot(ax=ax, **data_style)

        _rv = data.rv.to(rv_unit).value
        drv = _rv.max()-_rv.min()
        ax.set_ylim(_rv.min() - 0.2*drv, _rv.max() + 0.2*drv)

    ax.set_xlim(t_grid.min(), t_grid.max())
    if add_labels:
        ax.set_xlabel('BMJD')
        ax.set_ylabel('RV [{}]'.format(rv_unit.to_string(format='latex_inline')))

    return fig
