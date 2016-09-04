from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from os.path import exists, join

# Third-party
import astropy.time as atime
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from .data import RVData

def test_rvdata():

    # test various initializations
    t = np.random.uniform(55555., 56012., size=1024)
    rv = 100 * np.sin(0.5*t) * u.km/u.s
    ivar = 1 / (np.random.normal(0,5,size=1024)*u.km/u.s)**2
    RVData(t=t, rv=rv, ivar=ivar)

    t = atime.Time(t, format='mjd', scale='utc')
    RVData(t=t, rv=rv, ivar=ivar)

    with pytest.raises(TypeError):
        RVData(t=t, rv=rv.value, ivar=ivar)

    with pytest.raises(TypeError):
        RVData(t=t, rv=rv, ivar=ivar.value)

    # check that copy works
    t = atime.Time(t, format='mjd', scale='utc')
    data1 = RVData(t=t, rv=rv, ivar=ivar)
    data2 = data1.copy()

    data1._t *= 1.5
    data1._rv *= 1.5
    data1._ivar *= 1.5
    assert np.all(data2._t != data1._t)
    assert np.all(data2._rv != data1._rv)
    assert np.all(data2._ivar != data1._ivar)

    # check that plotting at least succeeds (TODO: could be better)
    data1.plot()
    data1.plot(color='r')
    data1.plot(ax=plt.gca())

    # try classmethod
    _basepath = '/Users/adrian/projects/ebak'
    if exists(_basepath):
        print("running classmethod test")

        apogee_id = "2M03080601+7950502"
        data = RVData.from_apogee(join(_basepath, 'data', 'allVisit-l30e.2.fits'),
                                  apogee_id=apogee_id)

        from astropy.io import fits
        d = fits.getdata(join(_basepath, 'data', 'allVisit-l30e.2.fits'), 1)
        data = RVData.from_apogee(d[d['APOGEE_ID'].astype(str) == apogee_id])
