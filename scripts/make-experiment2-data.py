"""

Generate simulated data for Experiment 2 where we inflate the uncertainties successively

"""

# Standard library
import os

# Third-party
import astropy.units as u
import h5py
import matplotlib
matplotlib.use('agg')
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker.celestialmechanics import OrbitalParams
from thejoker.data import RVData
from thejoker.units import default_units

def main():

    # high-eccentricity orbit with reasonable or randomly chosen parameters
    opars = OrbitalParams(P=103.71*u.day, K=8.134*u.km/u.s, ecc=0.313,
                          omega=np.random.uniform(0, 2*np.pi)*u.rad,
                          phi0=np.random.uniform(0, 2*np.pi)*u.rad,
                          v0=np.random.normal(0, 30) * u.km/u.s)
    orbit = opars.rv_orbit(0)
    print("Mass function:", orbit.pars.mf)
    print("omega:", orbit.pars.omega.to(u.degree))
    print("phi0:", orbit.pars.phi0.to(u.degree))
    print("v0:", orbit.pars.v0.to(u.km/u.s))

    n_obs = 8 # MAGIC NUMBER: number of observations

    # Experiment 1 data
    bmjd = np.random.uniform(0, 3*365, size=n_obs) + 55555. # 3 year survey
    bmjd.sort()
    rv = orbit.generate_rv_curve(bmjd)
    rv_err = np.random.uniform(100, 200, size=n_obs) * u.m/u.s # apogee-like
    rv = np.random.normal(rv.to(default_units['v0']).value,
                          rv_err.to(default_units['v0']).value) * default_units['v0']

    with h5py.File(os.path.join(paths.root, "data", "experiment2.h5"), "w") as outf:
        for fac in 2**np.arange(0, 3+1):
            g = outf.create_group(str(fac))

            data = RVData(t=bmjd, rv=rv, stddev=fac*rv_err)
            data.to_hdf5(g)
            g.create_dataset('truth_vector', data=opars.pack())

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
