"""

Create a subset of APOGEE's allVisit file that contains only the stars in Troup's sample.

"""

# Standard library
import os

# Third-party
import astropy.time as atime
import astropy.units as u
from astropy.io import fits
import h5py
import numpy as np

# Project
from thejoker import Paths
paths = Paths()

def main():
    allVisit_path = os.path.join(paths.root, "data", "allVisit-l30e.2.fits")
    troup_csv_path = os.path.join(paths.root, "data", "troup16-dr12.csv")

    if not os.path.exists(allVisit_path):
        download_cmd = ("wget https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allVisit-l30e.2.fits -O {}"
                        .format(allVisit_path))
        raise IOError("Path to main APOGEE DR13 allVisit file does not exist: {}\n"
                      "\t Download file with: {}"
                      .format(allVisit_path, download_cmd))

    troup = np.genfromtxt(troup_csv_path, delimiter=",", names=True, dtype=None)
    allVisit = fits.getdata(allVisit_path, 1)

    with h5py.File(paths.troup_allVisit, 'w') as f:
        for apogee_id in troup['APOGEE_ID'].astype(str):
            idx = allVisit['APOGEE_ID'].astype(str) == apogee_id
            this_data = allVisit[idx]

            good_idx = (np.isfinite(this_data['MJD']) &
                        np.isfinite(this_data['VHELIO']) &
                        np.isfinite(this_data['VRELERR']))
            this_data = this_data[good_idx]

            g = f.create_group(apogee_id)

            # TODO: are the MJD in allVisit definitely UTC and not barycenter?
            bmjd = atime.Time(this_data['MJD'], format='mjd', scale='utc').tcb.mjd

            d = g.create_dataset('mjd', data=bmjd)
            d.attrs['format'] = 'mjd'
            d.attrs['scale'] = 'tcb'

            rv = np.array(this_data['VHELIO']) * u.km/u.s
            rv_err = np.array(this_data['VRELERR']) * u.km/u.s

            d = g.create_dataset('rv', data=rv.value)
            d.attrs['unit'] = str(rv.unit)

            d = g.create_dataset('rv_err', data=rv_err.value)
            d.attrs['unit'] = str(rv_err.unit)

if __name__ == '__main__':
    main()
