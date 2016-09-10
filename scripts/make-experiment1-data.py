"""

Generate simulated data for Experiment 1 that match all of our assumptions:
    - A pair of stars, one of which is observed spectroscopically, with radial velocity variations
      that are described purely through the 2-body problem
    - Gaussian uncertainties on the RV measurements

"""

# Standard library
import os

# Third-party
import astropy.units as u
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker.celestialmechanics import SimulatedRVOrbit
from thejoker.data import RVData
from thejoker.units import usys

plt.style.use("../thejoker/thejoker.mplstyle")

def main():

    # high-eccentricity orbit with reasonable or randomly chosen parameters
    orbit = SimulatedRVOrbit(ecc=0.867, P=53.09*u.day, a_sin_i=4.2*u.R_sun,
                             omega=np.random.uniform(0, 2*np.pi)*u.rad,
                             phi0=np.random.uniform(0, 2*np.pi)*u.rad,
                             v0=np.random.normal(0, 30) * u.km/u.s)
    print(orbit.mf)

    n_obs = 8 # MAGIC NUMBER: number of observations

    obs_t_bounds = (55555., 57000.) # MAGIC NUMBERs

    bmjd = np.random.uniform(*obs_t_bounds, size=n_obs)
    rv = orbit.generate_rv_curve(bmjd)
    rv_err = np.random.uniform(100, 200, size=n_obs) * u.m/u.s # apogee-like
    rv = np.random.normal(rv.decompose(usys).value, rv_err.decompose(usys).value) * usys['speed']

    data = RVData(t=bmjd, rv=rv, stddev=rv_err)
    with h5py.File(os.path.join(paths.root, "data", "experiment1.h5"), "w") as f:

        d = f.create_dataset('mjd', data=bmjd)
        d.attrs['format'] = 'mjd'
        d.attrs['scale'] = 'tcb'

        d = f.create_dataset('rv', data=rv)
        d.attrs['unit'] = str(usys['speed'])

        d = f.create_dataset('rv_err', data=rv_err)
        d.attrs['unit'] = str(usys['speed'])

    # plot!
    plot_path = os.path.join(paths.plots, "experiment1")
    os.makedirs(plot_path, exist_ok=True)

    ax = data.plot(rv_unit=u.km/u.s)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(plot_path, "data.png"))

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    parser.add_argument("-s", "--seed", dest="seed", default=None,
                        type=int, help="Random number seed.")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    main()
