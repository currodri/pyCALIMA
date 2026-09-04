def Draine_1978_isrf(l):
    """Wavelength dependent UV intensity as given by the Draine (1978)
    fitting to observational data.

    Args:
        l (float): Wavelength in nm.

    Returns:
        float: Radiation intensity in photons cm^-2 s^-1 nm^-1
    """    
    
    I = 3.2028e13 * l**-3. - 5.1542e15 * l **-4. + 2.0546e17 * l**-5.
    
    return I


def _planck_nu_cgs(T, nu):
    """Planck function B_nu in cgs units.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    nu : float or ndarray
        Frequency in Hz.

    Returns
    -------
    float or ndarray
        B_nu in erg cm^-2 s^-1 Hz^-1 sr^-1.
    """
    import numpy as np

    h_SI = 6.62607015e-34
    kB_SI = 1.380649e-23
    c_SI = 2.99792458e8

    nu = np.asarray(nu, dtype=float)
    T = float(T)
    with np.errstate(over='ignore', invalid='ignore'):
        B_nu_SI = 2.0 * h_SI * nu**3 / c_SI**2 / (np.expm1(h_SI * nu / (kB_SI * T)))
    return B_nu_SI * 1e3


def Mathis83_radiation_field(E):
    """Mathis et al. (1983) ISRF energy density per unit energy.

    Parameters
    ----------
    E : float
        Photon energy in eV.

    Returns
    -------
    float
        u_E in erg cm^-3 eV^-1.
    """
    import numpy as np

    E = float(E)
    h_SI = 6.62607015e-34
    eV2J = 1.602176634e-19

    if E <= 0.0 or E > 13.6:
        return 0.0

    nu = E * eV2J / h_SI
    dnu_dE = eV2J / h_SI

    if 11.2 < E <= 13.6:
        return float(3.328e-9 * E**(-4.4172) / nu * dnu_dE)
    if 9.26 < E <= 11.2:
        return float(8.463e-13 / E / nu * dnu_dE)
    if 5.04 < E <= 9.26:
        return float(2.055e-14 * E**0.6678 / nu * dnu_dE)

    B1 = _planck_nu_cgs(7500.0, nu)
    B2 = _planck_nu_cgs(4000.0, nu)
    B3 = _planck_nu_cgs(3000.0, nu)
    I_nu = 1e-14 * B1 + 1.65e-13 * B2 + 4e-13 * B3
    c_cgs = 2.99792458e10
    u_nu = (4.0 * np.pi * I_nu) / c_cgs
    u_E = u_nu * dnu_dE
    return float(u_E)