import numpy as np

# Atomic masses in amu. mC matches the carbon target mass already used by
# models/PAH_gas_collisions/PAH_sputtering.py, so PAH masses are consistent
# between the sputtering and coalescence paths.
mC_amu = 12.0107
mH_amu = 1.00794
AMU_TO_G = 1.66053906660e-24


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
    mass = mass * AMU_TO_G # Convert to grams

    return mass

def has_uniform_bins(array: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Checks if the bin sizes (differences between consecutive elements) 
    in an array are identical within a specified precision tolerance.
    """
    # Ensure the input is a numpy array
    arr = np.asarray(array)
    
    # An array with fewer than 3 elements has at most 1 spacing interval,
    # so by definition, its bin sizes are "uniform".
    if len(arr) < 3:
        return True
        
    # Calculate the step size between every consecutive pair of elements
    bin_sizes = np.diff(arr)
    print(bin_sizes)
    
    # Check if the maximum variation in bin sizes is within our tolerance
    # (max bin size minus min bin size)
    return bool((np.max(bin_sizes) - np.min(bin_sizes))/np.min(bin_sizes) <= tolerance)