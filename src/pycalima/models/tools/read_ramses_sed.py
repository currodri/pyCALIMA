import numpy as np
import matplotlib.pyplot as plt
from unyt import Gyr, nm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from unyt import Gyr, nm
from scipy.io import FortranFile

def read_sed_tables(sed_dir):
    """
    Reads the SED tables from the given directory using the same Fortran-style loop.

    Parameters
    ----------
        sed_dir (str): Path to the directory containing SED files.

    Returns:
        metallicities (numpy array): Metallicity bins.
        ages (numpy array): Age bins in Gyr.
        wavelengths (numpy array): Wavelength bins in nm.
        SEDs (numpy array): SED values (wavelength, age, metallicity).
    """
    # Read metallicity bins
    metallicity_file = f"{sed_dir}/metallicity_bins.dat"
    with open(metallicity_file, "r") as f:
        nzs = int(f.readline().strip())  # Number of metallicity bins
        metallicities = np.array([float(f.readline().strip()) for _ in range(nzs)])

    # Read age bins
    age_file = f"{sed_dir}/age_bins.dat"
    with open(age_file, "r") as f:
        nAges = int(f.readline().strip())  # Number of age bins
        ages = np.array([float(f.readline().strip()) for _ in range(nAges)]) * 1e-9  # Convert yr to Gyr

    # Read SEDs using FortranFile
    sed_file = f"{sed_dir}/all_seds.dat"
    with FortranFile(sed_file, "r") as f:
        # Read the first two integers
        nLs, dum = f.read_ints(np.int32)  # Read nLs and dummy value
        wavelengths = f.read_reals(np.float64) * 0.1  # Read wavelength bins

        # Allocate space for SEDs
        SEDs = np.zeros((nzs, nAges, nLs), dtype=np.float64)

        # Read SEDs in Fortran-style order: metallicity → age → wavelength
        for iz in range(nzs):
            for ia in range(nAges):
                SEDs[iz, ia, :] = f.read_reals(np.float64)  # Read nLs values

    return metallicities, ages * Gyr, wavelengths * nm, SEDs


def plot_seds_by_metallicity(sed_dir, fixed_age=1.0, save_path="seds_by_metallicity.png"):
    """
    Plots the SEDs color-coded by metallicity for a fixed age and saves the plot.

    Parameters
    ----------
        sed_dir (str): Path to the directory containing SED files.
        fixed_age (float): Age in Gyr to filter the SEDs.
        save_path (str): Path to save the figure.
    """
    metallicities, ages, wavelengths, SEDs = read_sed_tables(sed_dir)

    # Find the index of the closest age to the desired fixed age
    age_index = np.argmin(np.abs(ages - fixed_age * Gyr))

    # Set up colormap
    norm = mcolors.LogNorm(vmin=np.min(metallicities), vmax=np.max(metallicities))
    cmap = cm.get_cmap("viridis")

    plt.figure(figsize=(8, 5))
    for iz, Z in enumerate(metallicities):
        plt.plot(wavelengths.value, SEDs[iz, age_index, :], color=cmap(norm(Z)))

    plt.xscale("log")
    plt.yscale("log")
    # Make sure that the y axis spans only 5 dex up and down the median SED value
    median_sed = np.median(SEDs[:, age_index, :])
    plt.ylim([median_sed / 1e5, median_sed * 1e5])

    plt.xlabel(f"Wavelength ({wavelengths.units})")
    plt.ylabel("SED (arbitrary units)")
    plt.title(f"SEDs at Age = {fixed_age} Gyr (Color-coded by Metallicity)")

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Metallicity (Z)")

    plt.grid()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {save_path}")


def plot_seds_by_age(sed_dir, fixed_metallicity=0.02, save_path="seds_by_age.png"):
    """
    Plots the SEDs color-coded by age for a fixed metallicity and saves the plot.

    Parameters
    ----------
        sed_dir (str): Path to the directory containing SED files.
        fixed_metallicity (float): Metallicity value to filter the SEDs.
        save_path (str): Path to save the figure.
    """
    metallicities, ages, wavelengths, SEDs = read_sed_tables(sed_dir)

    # Find the index of the closest metallicity to the desired fixed metallicity
    metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))

    # Set up colormap
    norm = mcolors.LogNorm(vmin=np.min(ages.value)+1e-3, vmax=np.max(ages.value))
    cmap = cm.get_cmap("plasma")

    plt.figure(figsize=(8, 5))
    for ia, age in enumerate(ages):
        plt.plot(wavelengths.value, SEDs[metallicity_index, ia, :], color=cmap(norm(age.value)))

    plt.xscale("log")
    plt.yscale("log")
    # Make sure that the y axis spans only 5 dex up and down the median SED value
    median_sed = np.median(SEDs[metallicity_index, :, :])
    plt.ylim([median_sed / 1e5, median_sed * 1e5])
    
    plt.xlabel(f"Wavelength ({wavelengths.units})")
    plt.ylabel("SED (arbitrary units)")
    plt.title(f"SEDs at Metallicity = {fixed_metallicity} (Color-coded by Age)")

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Age (Gyr)")

    plt.grid()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {save_path}")


# Example usage
# sed_directory = "/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300"  # Change to your SED directory
# plot_seds_by_metallicity(sed_directory, fixed_age=0.1, save_path="seds_metallicity.png")
# plot_seds_by_age(sed_directory, fixed_metallicity=0.02, save_path="seds_age.png")
