"""

Create a subset of APOGEE's allVisit file that contains only the stars in Troup's sample.

"""

# Standard library
import os

# Third-party
from astropy.io import fits
import h5py
import numpy as np

# Project
from thejoker import Paths
paths = Paths(__file__)

def main():
    allVisit_path = os.path.join(paths.root, "data", "allVisit-l30e.2.fits")
    troup_csv_path = os.path.join(paths.root, "data", "troup16-dr12.csv")
    output_path = os.path.join(paths.root, "data", "troup-allVisit.h5")

    if not os.path.exists(allVisit_path):
        download_cmd = ("wget https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allVisit-l30e.2.fits -O {}"
                        .format(allVisit_path))
        raise IOError("Path to main APOGEE DR13 allVisit file does not exist: {}\n"
                      "\t Download file with: {}"
                      .format(allVisit_path, download_cmd))

    troup = np.genfromtxt(troup_csv_path, delimiter=",", names=True, dtype=None)
    allVisit = fits.getdata(allVisit_path, 1)

    with h5py.File(output_path, 'w') as f:
        for apogee_id in troup['APOGEE_ID'].astype(str):
            idx = allVisit['APOGEE_ID'].astype(str) == apogee_id
            f.create_dataset(apogee_id, data=allVisit[idx])

if __name__ == '__main__':
    main()
