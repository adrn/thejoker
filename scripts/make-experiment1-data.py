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
from thejoker.celestialmechanics import OrbitalParams
from thejoker.data import RVData
from thejoker.units import usys

plt.style.use("../thejoker/thejoker.mplstyle")

def main():

    # high-eccentricity orbit with reasonable or randomly chosen parameters
    opars = OrbitalParams(P=103.71*u.day, asini=8.*u.R_sun, ecc=0.613,
                          omega=np.random.uniform(0, 2*np.pi)*u.rad,
                          phi0=np.random.uniform(0, 2*np.pi)*u.rad,
                          v0=np.random.normal(0, 30) * u.km/u.s)
    orbit = opars.rv_orbit(0)
    print("Mass function:", orbit.mf)
    print("omega:", orbit.omega.to(u.degree))
    print("phi0:", orbit.phi0.to(u.degree))
    print("v0:", orbit.v0.to(u.km/u.s))

    n_obs = 4 # MAGIC NUMBER: number of observations

    # Experiment 1 data
    bmjd = np.random.uniform(0, 3*365, size=n_obs) + 55555. # 3 year survey
    bmjd.sort()
    rv = orbit.generate_rv_curve(bmjd)
    rv_err = np.random.uniform(100, 200, size=n_obs) * u.m/u.s # apogee-like
    rv = np.random.normal(rv.decompose(usys).value, rv_err.decompose(usys).value) * usys['speed']

    data = RVData(t=bmjd, rv=rv, stddev=rv_err)
    with h5py.File(os.path.join(paths.root, "data", "experiment1.h5"), "w") as f:

        d = f.create_dataset('mjd', data=bmjd)
        d.attrs['format'] = 'mjd'
        d.attrs['scale'] = 'tcb'

        d = f.create_dataset('rv', data=rv.decompose(usys).value)
        d.attrs['unit'] = str(usys['speed'])

        d = f.create_dataset('rv_err', data=rv_err.decompose(usys).value)
        d.attrs['unit'] = str(usys['speed'])

        f.create_dataset('truth_vector', data=opars.pack())

    # plot!
    plot_path = os.path.join(paths.plots, "experiment1")
    os.makedirs(plot_path, exist_ok=True)

    ax = data.plot(rv_unit=u.km/u.s)
    orbit.plot(t=np.linspace(55555, 55555+3*365, num=1024), ax=ax, alpha=0.4)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(plot_path, "data.png"))

    # ------------------------------------------------------------------------
    # Experiment 5 data
    bmjd[1] = bmjd[0] + 2. # 2 days later
    rv[1] = orbit.generate_rv_curve(bmjd[1:2])
    rv_err[1] = np.random.uniform(100, 200) * u.m/u.s
    rv[1] = np.random.normal(rv[1].decompose(usys).value, rv_err[1].decompose(usys).value) * usys['speed']

    data = RVData(t=bmjd, rv=rv, stddev=rv_err)
    with h5py.File(os.path.join(paths.root, "data", "experiment5.h5"), "w") as f:

        d = f.create_dataset('mjd', data=bmjd)
        d.attrs['format'] = 'mjd'
        d.attrs['scale'] = 'tcb'

        d = f.create_dataset('rv', data=rv.decompose(usys).value)
        d.attrs['unit'] = str(usys['speed'])

        d = f.create_dataset('rv_err', data=rv_err.decompose(usys).value)
        d.attrs['unit'] = str(usys['speed'])

        f.create_dataset('truth_vector', data=opars.pack())

    # plot!
    plot_path = os.path.join(paths.plots, "Experiment5")
    os.makedirs(plot_path, exist_ok=True)

    ax = data.plot(rv_unit=u.km/u.s)
    orbit.plot(t=np.linspace(55555, 55555+3*365, num=1024), ax=ax, alpha=0.4)
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
