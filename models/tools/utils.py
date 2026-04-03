def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

def sigmoid_function(k,x0,x):
    import numpy as np
    x = x / x0
    x0 = 1.
    return 1. / (1. + np.exp(-k*(x-x0)))

def Nc_from_size(a):
    """This function returns the effective number of Carbon atoms
    for a PAH molecule, following Eq. 8 in Draine et al. (2021).

    Args:
        a (float): grain radius in Angstrom
    """

    return int(418*(a/10)**3)

def size_from_Nc(Nc):
    """This function returns the PAH radius (in Angstrom) from the number of Carbon atoms,
    following Eq. 8 in Draine et al. (2021).

    Args:
        Nc (int): number of Carbon atoms
    """

    return 10*((float(Nc)/418))**(1/3)

def mass_from_Nc(Nc):
    """This function returns the mass of a PAH molecule (in grams) from the number of Carbon atoms,
    assuming full hydrogenation.

    Args:
        Nc (int): number of Carbon atoms

    Returns:
        n.float: mass of the PAH molecule in grams
    """
    
    # Calculate the number of hydrogen atoms (assumming ful hydrogenation)
    num_hydrogen_atoms = 2. * float(Nc) + 2.
    
    # Calculate the mass of the PAH molecule
    mass = mC_amu * float(Nc) + mH_amu * num_hydrogen_atoms
    mass = mass * 1.66053906660e-24 # Convert to grams
    
    return mass