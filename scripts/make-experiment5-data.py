"""

Generate simulated data for Experiment 5:
    - Read in data from Experiment 1
    - Move 2nd data point through 1 cycle in period to show how the sampling depends on when, in
      phase, it is observed

"""

# Standard library
import os

# Third-party
import astropy.units as u
import h5py
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker.celestialmechanics import OrbitalParams
from thejoker.data import RVData
from thejoker.units import usys

def main():

    # Designer RV curves!

    opars = OrbitalParams(P=127.31*u.day, asini=22.124*u.R_sun, ecc=0.213,
                          omega=np.random.uniform(0, 2*np.pi)*u.rad,
                          phi0=np.random.uniform(0, 2*np.pi)*u.rad,
                          v0=np.random.normal(0, 30) * u.km/u.s)
    orbit = opars.rv_orbit(0)
    print("Mass function:", orbit.mf)
    print("omega:", orbit.omega.to(u.degree))
    print("phi0:", orbit.phi0.to(u.degree))
    print("v0:", orbit.v0.to(u.km/u.s))

    EPOCH = 55555. # arbitrary number
    P = opars.P.to(u.day).value[0]
    _t = np.array([0., 4.9, 4.92, 5.01]) * P
    t1 = np.concatenate((_t, [5.5 * P])) + EPOCH
    t2 = np.concatenate((_t, [5.92 * P])) + EPOCH

    rv_err = np.random.uniform(0.2, 0.3, size=t1.size) * u.km/u.s

    _rnd = np.random.normal(size=t1.size)
    rv1 = orbit.generate_rv_curve(t1) + _rnd*rv_err
    rv2 = orbit.generate_rv_curve(t2) + _rnd*rv_err

    with h5py.File(os.path.join(paths.root, "data", "experiment5.h5"), "w") as f:
        data1 = RVData(t=t1, rv=rv1, stddev=rv_err)
        data2 = RVData(t=t2, rv=rv2, stddev=rv_err)

        g = f.create_group("0")
        data1.to_hdf5(g)
        g.create_dataset('truth_vector', data=opars.pack())

        g = f.create_group("1")
        data2.to_hdf5(g)
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
