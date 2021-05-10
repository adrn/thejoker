# Standard library
import warnings

# Third-party
from astropy.time import Time
import astropy.units as u
from astropy.utils.decorators import deprecated_renamed_argument
import numpy as np

# Project
from .data import RVData
from .data_helpers import validate_prepare_data
from .prior_helpers import get_v0_offsets_equiv_units

__all__ = ['plot_rv_curves', 'plot_phase_fold']


def get_t_grid(data, P, span_factor=0.1, max_t_grid=None):
    w = np.ptp(data.t.mjd)
    dt = P.to_value(u.day) / 64  # MAGIC NUMBER

    if P.shape != ():
        raise ValueError("Period must be a scalar quantity")

    n_grid = w / dt
    if max_t_grid is not None and n_grid > max_t_grid:
        dt = w / max_t_grid
        n_grid = max_t_grid

    if n_grid > 1e4:
        warnings.warn(
            "Time grid has more than 10,000 grid points, so plotting orbits "
            "could be very slow! Set 't_grid' manually, or set 'max_t_grid' to "
            "decrease the number of time grid points.",
            ResourceWarning)

    t_grid = np.arange(data.t.mjd.min() - w*span_factor/2,
                       data.t.mjd.max() + w*span_factor/2 + dt,
                       dt)

    return t_grid


@deprecated_renamed_argument('relative_to_t0', 'relative_to_t_ref',
                             since='v1.2', warning_type=DeprecationWarning)
def plot_rv_curves(samples, t_grid=None, rv_unit=None, data=None,
                   ax=None, plot_kwargs=dict(), data_plot_kwargs=dict(),
                   add_labels=True, relative_to_t_ref=False,
                   apply_mean_v0_offset=True, max_t_grid=None):
    """
    Plot radial velocity curves for the input set of orbital parameter
    samples over the input grid of times.

    Parameters
    ----------
    samples : :class:`~thejoker.sampler.JokerSamples`
        Posterior samples from The Joker.
    t_grid : array_like, `~astropy.time.Time`, optional
        Array of times. Either in BMJD or as an Astropy Time object. If not
        specified, the time grid will be set to the data range with a small
        buffer.
    rv_unit : `~astropy.units.UnitBase`, optional
        The units to use when plotting RV's.
    data : `~thejoker.data.RVData`, optional
        Over-plot the data as well.
    ax : `~matplotlib.Axes`, optional
        A matplotlib axes object to plot on to. If not specified, will
        create a new figure and plot on that.
    plot_kwargs : dict, optional
        Passed to `matplotlib.pyplot.plot()`.
    data_plot_kwargs : dict, optional
        Passed to `thejoker.data.RVData.plot()`.
    add_labels : bool, optional
        Add labels to the axes or not.
    relative_to_t_ref : bool, optional
        Plot the time axis relative to ``samples.t_ref``.
    max_t_grid : int, optional
        The maximum number of grid points to use when

    Returns
    -------
    fig : `~matplotlib.Figure`

    """

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    fig = ax.figure

    if data is not None:
        data, ids, _ = validate_prepare_data(data, samples.poly_trend,
                                             samples.n_offsets)

    if t_grid is None:
        if data is None:
            raise ValueError('If data is not passed in, you must specify '
                             'the time grid.')

        t_grid = get_t_grid(data, samples['P'].min(), max_t_grid=max_t_grid)

    if not isinstance(t_grid, Time):  # Assume BMJD
        t_grid = Time(t_grid, format='mjd', scale='tcb')

    # scale the transparency of the lines
    n_plot = len(samples)
    Q = 4.  # HACK
    line_alpha = 0.05 + Q / (n_plot + Q)

    if rv_unit is None:
        rv_unit = u.km/u.s

    # default plotting style
    # TODO: move default style to global style config
    style = plot_kwargs.copy()
    style.setdefault('linestyle', '-')
    style.setdefault('linewidth', 0.5)
    style.setdefault('alpha', line_alpha)
    style.setdefault('marker', '')
    style.setdefault('color', '#555555')
    style.setdefault('rasterized', True)

    # plot orbits over the data
    model_rv = np.zeros((n_plot, len(t_grid)))
    for i in range(n_plot):
        orbit = samples.get_orbit(i)

        if t_grid is None:
            t_grid = get_t_grid(data, samples['P'][i], max_t_grid=max_t_grid)

        model_rv[i] = orbit.radial_velocity(t_grid).to(rv_unit).value

    model_ylim = (np.percentile(model_rv.min(axis=1), 5),
                  np.percentile(model_rv.max(axis=1), 95))

    bmjd = t_grid.tcb.mjd
    if relative_to_t_ref:
        if samples.t_ref is None:
            raise ValueError('Input samples object has no epoch .t_ref')
        bmjd = bmjd - samples.t_ref.tcb.mjd

    ax.plot(bmjd, model_rv.T, **style)

    if data is not None:
        if apply_mean_v0_offset:
            data_rv = np.array(data.rv.value)  # copy
            data_err = np.array(data.rv_err.to_value(data.rv.unit))

            unq_ids = np.unique(ids)
            dv0_names = get_v0_offsets_equiv_units(samples.n_offsets).keys()
            for i, name in enumerate(dv0_names):
                mask = ids == unq_ids[i+1]
                offset_samples = samples[name].to_value(data.rv.unit)
                data_rv[mask] -= np.mean(offset_samples)
                data_err[mask] = np.sqrt(data_err[mask]**2 +
                                         np.var(offset_samples))
            data = RVData(t=data.t,
                          rv=data_rv * data.rv.unit,
                          rv_err=data_err * data.rv.unit)

        data_style = data_plot_kwargs.copy()
        data_style.setdefault('rv_unit', rv_unit)
        data_style.setdefault('markersize', 4.)

        if data_style['rv_unit'] != rv_unit:
            raise u.UnitsError("Data plot units don't match rv_unit!")

        data.plot(ax=ax, relative_to_t_ref=relative_to_t_ref, add_labels=False,
                  **data_style)

        _rv = data.rv.to(rv_unit).value
        drv = _rv.max() - _rv.min()
        data_ylim = (_rv.min() - 0.2*drv, _rv.max() + 0.2*drv)
    else:
        data_ylim = None

    ax.set_xlim(bmjd.min(), bmjd.max())
    if add_labels:
        ax.set_xlabel('BMJD')
        ax.set_ylabel('RV [{}]'
                      .format(rv_unit.to_string(format='latex_inline')))

    if data_ylim is not None:
        ylim = (min(data_ylim[0], model_ylim[0]),
                max(data_ylim[1], model_ylim[1]))

    else:
        ylim = model_ylim

    ax.set_ylim(ylim)

    return fig


def plot_phase_fold(sample, data=None, ax=None,
                    with_time_unit=False, n_phase_samples=4096,
                    add_labels=True, show_s_errorbar=True, residual=False,
                    remove_trend=True, plot_kwargs=None, data_plot_kwargs=None):
    """
    Plot phase-folded radial velocity curves for the input orbital parameter
    sample, optionally with data phase-folded to the same period.

    Parameters
    ----------
    samples : :class:`~thejoker.sampler.JokerSamples`
        Posterior samples from The Joker.
    data : `~thejoker.data.RVData`, optional
        Over-plot the data as well.
    ax : `~matplotlib.Axes`, optional
        A matplotlib axes object to plot on to. If not specified, will
        create a new figure and plot on that.
    with_time_unit : bool, `astropy.units.Unit` (optional)
        Plot the phase in time units, not on 0–1 scale (i.e., mod P not mod 1).
    n_phase_samples : int (optional)
        Number of grid points in phase grid.
    add_labels : bool, optional
        Add labels to the axes or not.
    show_s_errorbar : bool, optional
        Plot an additional error bar to show the extra uncertainty ``s`` value
        for this sample.
    residual : bool, optional
        Plot the residual of the data relative to the model.
    remove_trend : bool, optional
        Remove the long-term velocity trend from the data and model before
        plotting.
    plot_kwargs : dict, optional
        Passed to `matplotlib.pyplot.plot()` for plotting the orbits.
    data_plot_kwargs : dict, optional
        Passed to `thejoker.data.RVData.plot()`.

    Returns
    -------
    fig : `~matplotlib.Figure`
    """

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    fig = ax.figure

    # TODO: what do if passing in multiple samples?

    if data is not None:
        data, ids, _ = validate_prepare_data(data, sample.poly_trend,
                                             sample.n_offsets)
        rv_unit = data.rv.unit
    else:
        rv_unit = sample['v0'].unit

    # plotting styles:
    if plot_kwargs is None:
        plot_kwargs = dict()
    if data_plot_kwargs is None:
        data_plot_kwargs = dict()

    # TODO: move default style to global style config
    orbit_style = plot_kwargs.copy()
    orbit_style.setdefault('linestyle', '-')
    orbit_style.setdefault('linewidth', 0.5)
    orbit_style.setdefault('alpha', 1.)
    orbit_style.setdefault('marker', '')
    orbit_style.setdefault('color', '#555555')
    orbit_style.setdefault('rasterized', True)

    data_style = data_plot_kwargs.copy()
    data_style.setdefault('linestyle', 'none')
    data_style.setdefault('marker', 'o')
    data_style.setdefault('markersize', 4.)
    data_style.setdefault('zorder', 10)

    # Get orbit from input sample
    orbit = sample.get_orbit()
    P = sample['P'].item()

    if data is not None:
        rv = data.rv

        if remove_trend:
            # HACK:
            trend = orbit._vtrend
            orbit._vtrend = lambda t: 0.
            rv = rv - trend(data.t)

        v0_offset_names = get_v0_offsets_equiv_units(sample.n_offsets).keys()
        for i, offset_name in zip(range(1, sample.n_offsets+1),
                                  v0_offset_names):
            _tmp = sample[offset_name].item()
            rv[ids == i] -= _tmp

        t0 = sample.get_t0()

        time_unit = u.day
        dt_jd = (data.t - t0).tcb.jd * u.day
        if with_time_unit is False:
            phase = (dt_jd / P) % 1
        else:
            if with_time_unit is not True:
                time_unit = u.Unit(with_time_unit)
            phase = (dt_jd % P).to_value(time_unit)

        if residual:
            rv = rv - orbit.radial_velocity(data.t)

        # plot the phase-folded data and orbit
        ax.errorbar(phase, rv.to(rv_unit).value,
                    data.rv_err.to(rv_unit).value,
                    **data_style)

        if show_s_errorbar and 's' in sample.par_names:
            ax.errorbar(phase, rv.to(rv_unit).value,
                        np.sqrt(data.rv_err**2 +
                                sample['s']**2).to(rv_unit).value,
                        linestyle='none', marker='', elinewidth=0.,
                        color='#aaaaaa', alpha=0.9, capsize=0,
                        zorder=9)

    elif data is None and residual:
        raise ValueError("TODO: not allowed")

    # Set up the phase grid:
    unit_phase_grid = np.linspace(0, 1, n_phase_samples)
    if with_time_unit is not False:
        phase_grid = unit_phase_grid * P.to_value(time_unit)
    else:
        phase_grid = unit_phase_grid

    if not residual:
        ax.plot(phase_grid,
                orbit.radial_velocity(t0 +
                                      P * unit_phase_grid).to_value(rv_unit),
                **orbit_style)

    if add_labels:
        if with_time_unit is not False:
            ax.set_xlabel(r'phase, $(t-t_0)~{\rm mod}~P$ ' +
                          f'[{time_unit:latex_inline}]')
        else:
            ax.set_xlabel(r'phase, $\frac{t-t_0}{P}$')
        ax.set_ylabel(f'RV [{data.rv.unit:latex_inline}]')

    return fig
