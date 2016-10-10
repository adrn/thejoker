# Standard library
import os

# Third-party
import astropy.time as atime
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

samples_filename = "../cache/experiment1.h5"
data_filename = "../data/experiment1.h5"
plot_path = "../paper/figures/"

def main():

with h5py.File(data_filename, 'r') as f:
    data = RVData.from_hdf5(f)
    truth_opars = OrbitalParams.unpack(f['truth_vector'])

if __name__ == '__main__':
    main()
