"""A port from exoplanet, to support pymc recent versions"""

__all__ = ["KeplerianOrbit"]


import numpy as np
import pytensor.tensor as pt
from astropy import units as u
from astropy.constants import G
from exoplanet_core.pymc import ops

tt = pt
G_grav = G.to(u.R_sun**3 / u.M_sun / u.day**2).value


class KeplerianOrbit:
    """A system of bodies on Keplerian orbits around a common central

    Given the input parameters, the values of all other parameters will be
    computed so a ``KeplerianOrbit`` instance will always have attributes for
    each argument. Note that the units of the computed attributes will all be
    in the standard units of this class (``R_sun``, ``M_sun``, and ``days``)
    except for ``rho_star`` which will be in ``g / cm^3``.

    There are only specific combinations of input parameters that can be used:

    1. First, either ``period`` or ``a`` must be given. If values are given
       for both parameters, then neither ``m_star`` or ``rho_star`` can be
       defined because the stellar density implied by each planet will be
       computed in ``rho_star``.
    2. Only one of ``incl`` and ``b`` can be given.
    3. If a value is given for ``ecc`` then ``omega`` must also be given.
    4. If no stellar parameters are given, the central body is assumed to be
       the sun. If only ``rho_star`` is defined, the radius of the central is
       assumed to be ``1 * R_sun``. Otherwise, at most two of ``m_star``,
       ``r_star``, and ``rho_star`` can be defined.
    5. Either ``t0`` (reference transit) or ``t_periastron`` must be given,
       but not both.


    Args:
        period: The orbital periods of the bodies in days.
        a: The semimajor axes of the orbits in ``R_sun``.
        t0: The time of a reference transit for each orbits in days.
        t_periastron: The epoch of a reference periastron passage in days.
        incl: The inclinations of the orbits in radians.
        b: The impact parameters of the orbits.
        ecc: The eccentricities of the orbits. Must be ``0 <= ecc < 1``.
        omega: The arguments of periastron for the orbits in radians.
        Omega: The position angles of the ascending nodes in radians.
        m_planet: The masses of the planets in units of ``m_planet_units``.
        m_star: The mass of the star in ``M_sun``.
        r_star: The radius of the star in ``R_sun``.
        rho_star: The density of the star in units of ``rho_star_units``.

    """

    def __init__(
        self,
        period=None,
        a=None,
        t0=None,
        t_periastron=None,
        ecc=None,
        omega=None,
        sin_omega=None,
        cos_omega=None,
        Omega=None,
        model=None,
        **kwargs,
    ):
        self.period = period
        self.m_planet = tt.zeros_like(period)
        self.m_star = tt.ones_like(period)
        self.m_total = self.m_star + self.m_planet

        if a is None:
            a = (
                G_grav * (self.m_star + self.m_planet) * self.period**2 / (4 * np.pi**2)
            ) ** (1.0 / 3)
        self.a = a

        self.n = 2 * np.pi / self.period
        self.K0 = self.n * self.a / self.m_total

        if Omega is None:
            self.Omega = None
        else:
            self.Omega = pt.as_tensor_variable(Omega)
            self.cos_Omega = tt.cos(self.Omega)
            self.sin_Omega = tt.sin(self.Omega)

        # Eccentricity
        if ecc is None:
            self.ecc = None
            self.M0 = 0.5 * np.pi + tt.zeros_like(self.n)
        else:
            self.ecc = pt.as_tensor_variable(ecc)
            if omega is not None:
                if sin_omega is not None and cos_omega is not None:
                    raise ValueError(
                        "either 'omega' or 'sin_omega' and 'cos_omega' can be "
                        "provided"
                    )
                self.omega = pt.as_tensor_variable(omega)
                self.cos_omega = tt.cos(self.omega)
                self.sin_omega = tt.sin(self.omega)
            elif sin_omega is not None and cos_omega is not None:
                self.cos_omega = pt.as_tensor_variable(cos_omega)
                self.sin_omega = pt.as_tensor_variable(sin_omega)
                self.omega = tt.arctan2(self.sin_omega, self.cos_omega)

            else:
                raise ValueError("both e and omega must be provided")

            opsw = 1 + self.sin_omega
            E0 = 2 * tt.arctan2(
                tt.sqrt(1 - self.ecc) * self.cos_omega,
                tt.sqrt(1 + self.ecc) * opsw,
            )
            self.M0 = E0 - self.ecc * tt.sin(E0)

            ome2 = 1 - self.ecc**2
            self.K0 /= tt.sqrt(ome2)

        zla = tt.zeros_like(self.period)
        self.incl = 0.5 * np.pi + zla
        self.cos_incl = zla
        self.b = zla

        if t0 is not None and t_periastron is not None:
            raise ValueError("you can't define both t0 and t_periastron")
        if t0 is None and t_periastron is None:
            t0 = tt.zeros_like(self.period)

        if t0 is None:
            self.t_periastron = pt.as_tensor_variable(t_periastron)
            self.t0 = self.t_periastron + self.M0 / self.n
        else:
            self.t0 = pt.as_tensor_variable(t0)
            self.t_periastron = self.t0 - self.M0 / self.n

        self.tref = self.t_periastron - self.t0

        self.sin_incl = tt.sin(self.incl)

    def _rotate_vector(self, x, y):
        """Apply the rotation matrices to go from orbit to observer frame

        In order,
        1. rotate about the z axis by an amount omega -> x1, y1, z1
        2. rotate about the x1 axis by an amount -incl -> x2, y2, z2
        3. rotate about the z2 axis by an amount Omega -> x3, y3, z3

        Args:
            x: A tensor representing the x coodinate in the plane of the
                orbit.
            y: A tensor representing the y coodinate in the plane of the
                orbit.

        Returns:
            Three tensors representing ``(X, Y, Z)`` in the observer frame.

        """

        # 1) rotate about z0 axis by omega
        if self.ecc is None:
            x1 = x
            y1 = y
        else:
            x1 = self.cos_omega * x - self.sin_omega * y
            y1 = self.sin_omega * x + self.cos_omega * y

        # 2) rotate about x1 axis by -incl
        x2 = x1
        y2 = self.cos_incl * y1
        # z3 = z2, subsequent rotation by Omega doesn't affect it
        Z = -self.sin_incl * y1

        # 3) rotate about z2 axis by Omega
        if self.Omega is None:
            return (x2, y2, Z)

        X = self.cos_Omega * x2 - self.sin_Omega * y2
        Y = self.sin_Omega * x2 + self.cos_Omega * y2
        return X, Y, Z

    def _warp_times(self, t, _pad=True):
        if _pad:
            return tt.shape_padright(t) - self.t0
        return t - self.t0

    def _get_true_anomaly(self, t, _pad=True):
        M = (self._warp_times(t, _pad=_pad) - self.tref) * self.n
        if self.ecc is None:
            return tt.sin(M), tt.cos(M)
        sinf, cosf = ops.kepler(M, self.ecc + tt.zeros_like(M))
        return sinf, cosf

    def _get_velocity(self, m, t):
        """Get the velocity vector of a body in the observer frame"""
        sinf, cosf = self._get_true_anomaly(t)
        K = self.K0 * m
        if self.ecc is None:
            return self._rotate_vector(-K * sinf, K * cosf)
        return self._rotate_vector(-K * sinf, K * (cosf + self.ecc))

    def get_star_velocity(self, t):
        """Get the star's velocity vector

        .. note:: For a system with multiple planets, this will return one
            column per planet with the contributions from each planet. The
            total velocity can be found by summing along the last axis.

        Args:
            t: The times where the velocity should be evaluated.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.

        """
        return tuple(tt.squeeze(x) for x in self._get_velocity(self.m_planet, t))

    def get_radial_velocity(self, t, K=None, output_units=None):
        """Get the radial velocity of the star

        .. note:: The convention in exoplanet is that positive `z` points
            *towards* the observer. However, for consistency with radial
            velocity literature this method returns values where positive
            radial velocity corresponds to a redshift as expected.

        Args:
            t: The times where the radial velocity should be evaluated.
            K (Optional): The semi-amplitudes of the orbits. If provided, the
                ``m_planet`` and ``incl`` parameters will be ignored and this
                amplitude will be used instead.
            output_units (Optional): An AstroPy velocity unit. If not given,
                the output will be evaluated in ``m/s``. This is ignored if a
                value is given for ``K``.

        Returns:
            The reflex radial velocity evaluated at ``t`` in units of
            ``output_units``. For multiple planets, this will have one row for
            each planet.

        """

        # Special case for K given: m_planet, incl, etc. is ignored
        if K is not None:
            sinf, cosf = self._get_true_anomaly(t)
            if self.ecc is None:
                return tt.squeeze(K * cosf)
            # cos(w + f) + e * cos(w) from Lovis & Fischer
            return tt.squeeze(
                K
                * (
                    self.cos_omega * cosf
                    - self.sin_omega * sinf
                    + self.ecc * self.cos_omega
                )
            )

        # Compute the velocity using the full orbit solution
        if output_units is None:
            output_units = u.m / u.s
        conv = (1 * u.R_sun / u.day).to(output_units).value
        v = self.get_star_velocity(t)
        return -conv * v[2]


def get_true_anomaly(M, e, **kwargs):
    """Get the true anomaly for a tensor of mean anomalies and eccentricities

    Args:
        M: The mean anomaly.
        e: The eccentricity. This should have the same shape as ``M``.

    Returns:
        The true anomaly of the orbit.

    """
    sinf, cosf = ops.kepler(M, e)
    return tt.arctan2(sinf, cosf)
