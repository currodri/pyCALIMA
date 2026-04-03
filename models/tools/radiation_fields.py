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