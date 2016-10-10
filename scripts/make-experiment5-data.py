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
                          omega=137.234*u.degree,
                          phi0=36.231*u.degree,
                          v0=17.643*u.km/u.s)
    orbit = opars.rv_orbit(0)

    EPOCH = 55555. # arbitrary number
    P = opars.P.to(u.day).value[0]
    f0 = opars._phi0[0]/(2*np.pi)
    _t = (np.array([0.02, 4.08, 4.45, 4.47]) + f0) * P

    ts = [
        np.concatenate((_t, (np.array([6.46, 7.04]) + f0) * P)) + EPOCH,
        np.concatenate((_t, (np.array([6.04, 6.07]) + f0) * P)) + EPOCH,
        np.concatenate((_t, (np.array([6.62, 6.65]) + f0) * P)) + EPOCH
    ]

    rv_err = np.random.uniform(0.2, 0.3, size=ts[0].size) * u.km/u.s

    _rnd = np.random.normal(size=ts[0].size)

    rvs = []
    for t in ts:
        rvs.append(orbit.generate_rv_curve(t) + _rnd*rv_err)

    # import matplotlib.pyplot as plt
    # plt.errorbar(t1, rv1.to(u.km/u.s).value, rv_err.to(u.km/u.s).value, linestyle='none', marker='o', zorder=90)
    # plt.errorbar(t2, rv2.to(u.km/u.s).value, rv_err.to(u.km/u.s).value, linestyle='none', marker='o', zorder=100)
    # t_grid = np.linspace(t1.min()-150, t1.max()+150, 1024)
    # plt.plot(t_grid, orbit.generate_rv_curve(t_grid).to(u.km/u.s), marker=None, linestyle='--', zorder=-1, alpha=0.5)
    # plt.show()
    # return

    with h5py.File(os.path.join(paths.root, "data", "experiment5.h5"), "w") as f:
        _data = RVData(t=ts[0], rv=rvs[0], stddev=rv_err)
        data0 = _data[:-2]

        g = f.create_group("0")
        data0.to_hdf5(g)
        g.create_dataset('truth_vector', data=opars.pack())

        for i,t,rv in zip(range(len(ts)), ts, rvs):
            data = RVData(t=t, rv=rv, stddev=rv_err)
            g = f.create_group(str(i+1))
            data.to_hdf5(g)

            g = f.create_group('truth')
            opars.to_hdf5(g)

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
