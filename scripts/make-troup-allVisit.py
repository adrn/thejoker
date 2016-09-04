"""

Create a subset of APOGEE's allVisit file that contains only the stars in Troup's sample.

"""

# Standard library
import os

# Third-party
from astropy.io import fits
import astropy.time as atime
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

_basepath = os.path.split(os.path.abspath(os.path.join(__file__, "..")))[0]
if not os.path.exists(os.path.join("..", "scripts")):
    raise IOError("You must run this script from inside the scripts directory:\n{}"
                  .format(os.path.join(_basepath, "scripts")))

def main():
    allVisit_path = "../data/allVisit-l30e.2.fits"
    if not os.path.exists(allVisit_path):
        download_cmd = "wget https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allVisit-l30e.2.fits -O ../data/allVisit-l30e.2.fits"
        raise IOError("Path to main APOGEE DR13 allVisit file does not exist: {}\n"
                      "\t Download file with: {}"
                      .format(allVisit_path, download_cmd))

    troup = np.genfromtxt("../data/troup16-dr12.csv", delimiter=",", names=True, dtype=None)
    allVisit = fits.getdata(allVisit_path, 1)

    # get a boolean array to select only stars in troup's sample - I could do this with
    #   a join operation if I made these astropy table objects...
    idx = np.zeros(len(allVisit)).astype(bool)
    for apogee_id in troup['APOGEE_ID'].astype(str):
        idx |= allVisit['APOGEE_ID'].astype(str) == apogee_id

    # Create HDU's for the new file -- primary HDU header should match allVisit
    hdu0 = fits.PrimaryHDU(header=fits.getheader("../data/allVisit-l30e.2.fits", 0))
    hdu1 = fits.BinTableHDU(data=allVisit[idx])
    hdulist = fits.HDUList([hdu0, hdu1])

    hdulist.writeto("../data/troup-allVisit.fits")

if __name__ == '__main__':
    main()
