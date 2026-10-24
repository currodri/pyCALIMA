"""Grain heating and emission around an AGN (optically thin).

Building blocks for hot-dust models of quasars:

* ``skirtor_sed``: the accretion-disk SED of the SKIRTOR models (Stalevski et
  al. 2012, 2016, after Schartmann et al. 2005), lambda L_lambda propto
  lambda^1.2 (1-10 nm), flat (10-100 nm), lambda^-0.5 (0.1-5 um), lambda^-3
  beyond, normalised to L_bol over 1 nm - 1 mm;
* ``disk_anisotropy``: the angular distribution of the disk emission,
  normalised to a mean of 1 over the sphere, so that the flux at polar angle
  theta is f(theta) L_lambda / (4 pi r^2);
* ``HotGrain``: one grain (material, radius) with its Draine & Lee /
  Laor & Draine absorption cross section, a tabulated emitted power P_em(T)
  for fast equilibrium temperatures up to 5000 K, and its emission spectrum
  4 pi C_abs B_lambda(T);
* ``AGNIllumination``: an SED and anisotropy law; equilibrium temperature of
  a grain at (r, theta) and its inverse, the radius at which a grain reaches
  a given temperature. In the optically thin limit the absorbed power scales
  as f(theta) / r^2, so both are a table look-up.

The equilibrium temperature avoids the 800 K bracket of the
``dust_emission.compute_equilibrium_temperature*`` helpers; it agrees with
``dust_sublimation.compute_bin_temperature``.

Units are cgs throughout (wavelengths in cm, cross sections in cm^2,
L_lambda in erg s^-1 cm^-1, radii in cm) unless a name says otherwise.
"""

import weakref

import numpy as np

from pycalima.models.dust_radiation.dust_emission import interpolate_cross_sections

_H = 6.62607015e-27       # [erg s]
_C = 2.99792458e10        # [cm s^-1]
_KB = 1.380649e-16        # [erg K^-1]
PC_CM = 3.0856775814913673e18


def planck_lambda(lam_cm, T):
    """B_lambda(T) [erg s^-1 cm^-2 cm^-1 sr^-1]; broadcasts lam_cm against T."""
    lam = np.asarray(lam_cm, dtype=float)
    x = np.clip(_H * _C / (lam * _KB * np.asarray(T, dtype=float)), None, 700.0)
    return 2.0 * _H * _C ** 2 / lam ** 5 / np.expm1(x)


def skirtor_shape(lam_um):
    """SKIRTOR lambda L_lambda shape, continuous, arbitrary normalisation."""
    lam = np.asarray(lam_um, dtype=float)
    return np.where(lam < 0.01, (lam / 0.01) ** 1.2,
           np.where(lam < 0.1, 1.0,
           np.where(lam < 5.0, (lam / 0.1) ** -0.5,
                    50.0 ** -0.5 * (lam / 5.0) ** -3)))


def skirtor_sed(lam_cm, L_bol):
    """SKIRTOR disk L_lambda [erg s^-1 cm^-1] with integral L_bol over 1 nm - 1 mm."""
    lam_n = np.logspace(-3, 3, 6001)                      # [um]
    norm = L_bol / np.trapezoid(skirtor_shape(lam_n) / lam_n, lam_n)
    lam_um = np.asarray(lam_cm, dtype=float) * 1e4
    inside = (lam_um >= 1e-3) & (lam_um <= 1e3)
    return np.where(inside, norm * skirtor_shape(lam_um) / lam_um * 1e4, 0.0)


ANISOTROPY_LAWS = ('isotropic', 'cos', 'skirtor')


def disk_anisotropy(cos_theta, law='skirtor'):
    """Angular distribution f(theta) of the disk emission, <f> = 1 over 4 pi.

    ``theta`` is the polar angle from the disk axis; the disk emits into both
    hemispheres.

    * ``'isotropic'``: f = 1;
    * ``'cos'``: f = 2 ``|cos theta|`` (a flat, optically thick disk);
    * ``'skirtor'``: f = (6/7) ``|cos theta|`` (1 + 2 ``|cos theta|``) (Netzer 1987
      limb darkening as used by SKIRTOR, Stalevski et al. 2016).
    """
    mu = np.abs(np.asarray(cos_theta, dtype=float))
    if law == 'isotropic':
        return np.ones_like(mu)
    if law == 'cos':
        return 2.0 * mu
    if law == 'skirtor':
        return 6.0 / 7.0 * mu * (1.0 + 2.0 * mu)
    raise ValueError(f"law must be one of {ANISOTROPY_LAWS}, not {law!r}")


class HotGrain:
    """A grain of one material and radius, with its absorption cross section
    and a tabulated emitted power for fast equilibrium temperatures.

    Parameters
    ----------
    material : {'graphite', 'silicate'}
    a_um : float
        Grain radius [um], within the 0.001-10 um of the optical tables.
    T_table : array, optional
        Temperatures [K] of the P_em(T) table (default 3-5000 K).
    """

    def __init__(self, material, a_um, T_table=None):
        self.material = material
        self.a_um = float(a_um)
        self.a_cm = self.a_um * 1e-4
        _, wav, _, C_abs, _ = interpolate_cross_sections(material, self.a_um)
        order = np.argsort(wav)
        self._build(np.asarray(wav, dtype=float)[order], np.asarray(C_abs, dtype=float)[order],
                    T_table)

    @classmethod
    def from_optical_constants(cls, name, a_um, nk_file, T_table=None, lam_um=None):
        """A grain of radius a_um [um] from tabulated optical constants
        (a ``stx3141_supp`` file of Kitzmann & Heng 2018, e.g. "Fe.dat"):
        Mie absorption (``collapse.optics``: log-log n, k, power-law far-IR
        tail, small-particle expansion where ``|m| x`` < 0.1) on a wavelength grid
        (default 300 points, 0.01 um - 1 mm)."""
        import miepython

        from pycalima import _paths
        from pycalima.collapse.optics import _read_columns, qabs_small, refractive_index

        lam = np.geomspace(0.01, 1000.0, 300) if lam_um is None else np.asarray(lam_um, float)
        m = refractive_index(lam, *_read_columns(_paths.get_optical_props_path("stx3141_supp", nk_file)))
        x = 2.0 * np.pi * float(a_um) / lam
        Q = np.empty(lam.size)
        small = np.abs(m) * x < 0.1
        Q[small] = qabs_small(m[small], x[small])
        if (~small).any():
            qext, qsca, _, _ = miepython.efficiencies(m[~small], 2.0 * float(a_um), lam[~small])
            Q[~small] = qext - qsca
        g = cls.__new__(cls)
        g.material = name
        g.a_um = float(a_um)
        g.a_cm = g.a_um * 1e-4
        g._build(lam * 1e-4, np.maximum(Q, 1e-30) * np.pi * g.a_cm ** 2, T_table)
        return g

    def _build(self, wav_cm, C_abs, T_table):
        self.wav_cm = wav_cm
        self.C_abs = C_abs
        self.T_table = (np.logspace(np.log10(3.0), np.log10(5000.0), 1500)
                        if T_table is None else np.asarray(T_table, dtype=float))
        B = planck_lambda(self.wav_cm[None, :], self.T_table[:, None])
        self.P_em_table = 4.0 * np.pi * np.trapezoid(self.C_abs * B, self.wav_cm, axis=1)
        self._lnP = np.log(self.P_em_table)
        self._lnT = np.log(self.T_table)

    def cross_section(self, lam_cm):
        """C_abs [cm^2] at lam_cm (log-log interpolation, clamped at the table ends)."""
        return np.exp(np.interp(np.log(lam_cm), np.log(self.wav_cm), np.log(self.C_abs)))

    def absorbed_power(self, lam_cm, F_lambda):
        """Power absorbed [erg s^-1] from a flux F_lambda = 4 pi J_lambda
        [erg s^-1 cm^-2 cm^-1] given on an ascending grid lam_cm."""
        return np.trapezoid(self.cross_section(lam_cm) * F_lambda, lam_cm)

    def emitted_power(self, T):
        """4 pi int C_abs B_lambda(T) dlambda [erg s^-1]."""
        return np.exp(np.interp(np.log(T), self._lnT, self._lnP))

    def temperature(self, P_abs):
        """Equilibrium temperature [K] for an absorbed power P_abs [erg s^-1].

        NaN outside the table (below 3 K or above 5000 K)."""
        lnP = np.log(np.asarray(P_abs, dtype=float))
        T = np.exp(np.interp(lnP, self._lnP, self._lnT))
        return np.where((lnP < self._lnP[0]) | (lnP > self._lnP[-1]), np.nan, T)

    def luminosity_density(self, lam_cm, T):
        """Emission of one grain, L_lambda = 4 pi C_abs B_lambda(T)
        [erg s^-1 cm^-1]; broadcasts lam_cm against T."""
        return 4.0 * np.pi * self.cross_section(lam_cm) * planck_lambda(lam_cm, T)


class AGNIllumination:
    """Optically thin illumination of grains by an AGN.

    Parameters
    ----------
    L_bol : float
        Disk bolometric luminosity [erg s^-1] (isotropic equivalent of the
        angle-averaged emission).
    law : str
        Anisotropy law, see ``disk_anisotropy``.
    lam_cm : array, optional
        Wavelength grid of the SED [cm]; default 1 nm - 1 mm, 3000 points.
    L_lambda : array, optional
        SED [erg s^-1 cm^-1] on lam_cm; default ``skirtor_sed``.
    """

    def __init__(self, L_bol, law='skirtor', lam_cm=None, L_lambda=None):
        self.L_bol = float(L_bol)
        self.law = law
        self.lam_cm = np.logspace(-7, -1, 3000) if lam_cm is None else np.asarray(lam_cm)
        self.L_lambda = (skirtor_sed(self.lam_cm, self.L_bol) if L_lambda is None
                         else np.asarray(L_lambda, dtype=float))
        self._P1 = weakref.WeakKeyDictionary()

    def flux(self, r_cm, cos_theta=None):
        """F_lambda = f(theta) L_lambda / (4 pi r^2) [erg s^-1 cm^-2 cm^-1]
        on ``lam_cm`` (isotropic f = 1 when cos_theta is None)."""
        f = 1.0 if cos_theta is None else disk_anisotropy(cos_theta, self.law)
        return f * self.L_lambda / (4.0 * np.pi * r_cm ** 2)

    def _absorbed_at_1pc(self, grain):
        if grain not in self._P1:
            self._P1[grain] = grain.absorbed_power(self.lam_cm, self.flux(PC_CM))
        return self._P1[grain]

    def _f(self, cos_theta):
        return 1.0 if cos_theta is None else disk_anisotropy(cos_theta, self.law)

    def temperature(self, grain, r_cm, cos_theta=None):
        """Equilibrium temperature [K] of ``grain`` at radius r_cm and polar
        angle theta; broadcasts r_cm against cos_theta."""
        P = self._absorbed_at_1pc(grain) * self._f(cos_theta) * (PC_CM / np.asarray(r_cm)) ** 2
        return grain.temperature(P)

    def radius_at_temperature(self, grain, T, cos_theta=None):
        """Radius [cm] at which ``grain`` reaches temperature T at theta."""
        P1 = self._absorbed_at_1pc(grain) * self._f(cos_theta)
        return PC_CM * np.sqrt(P1 / grain.emitted_power(T))
