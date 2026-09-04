"""Fit a power law with an exponential cut-off to the Nozawa+2007 grain-size
distributions for Mg2SiO4 and carbonaceous dust.

Run as a script:

    python -m pycalima.models.tools.fit_nozawa2007_gsd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pycalima import _paths
from pycalima.plotting_style import use_calima_style


# Define a power-law with an exponential cut-off function
def power_law_exponential(x, A, alpha, x_cut):
    return np.log10(A * (x ** alpha) * np.exp(-x / x_cut))


def load_data():
    """Load the two bundled Nozawa+2007 grain-size distributions."""
    columns = ['Grain Size', 'N_grain']
    frames = {}
    for material, filename in (('Mg2SiO4', 'nozawa_2007_Mg2SiO4_gsd.dat'),
                               ('C', 'nozawa_2007_C_gsd.dat')):
        data = pd.read_csv(_paths.get_external_data_path(filename), names=columns)
        # Ensure numerical data is in float format
        data['Grain Size'] = pd.to_numeric(data['Grain Size'], errors='coerce')
        data['N_grain'] = pd.to_numeric(data['N_grain'], errors='coerce')
        frames[material] = data
    return frames


def main(outdir=None):
    """Fit both distributions, print the parameters and save the figure."""
    use_calima_style()
    frames = load_data()

    plt.figure()

    # Choose different line colours and marker colour for each material
    line_color = ['red', 'blue']
    marker_color = ['orange', 'green']

    for material, color, mcolor in zip(frames, line_color, marker_color):
        data = frames[material]
        x_data = data['Grain Size'].values
        y_data = data['N_grain'].values

        # Initial guess for parameters
        p0 = [1e19, -0.5, 1e-2]
        bounds = ([1e10, -10, 1e-3], [1e22, 10, 0.06])  # Ensure A and x_cut are non-negative

        # Fit curve
        popt, cov = curve_fit(power_law_exponential, x_data, np.log10(y_data), p0=p0, bounds=bounds)
        A_fit, alpha_fit, x_cut_fit = popt

        # Generate fitted curve
        x_fit = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 100)
        y_fit = 10**power_law_exponential(x_fit, *popt)

        # Plot
        plt.scatter(x_data, y_data, label=f'Observed {material}', color=mcolor)
        plt.plot(x_fit, y_fit,
                 label=f'Fit: A={A_fit:.2e}, alpha={alpha_fit:.2f}, x_cut={x_cut_fit:.2e}',
                 color=color)

        print(f'{material} Fit Parameters: A={A_fit:.2e}, alpha={alpha_fit:.2f}, x_cut={x_cut_fit:.2e}')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e14, 1e20])
    plt.xlabel('Grain Size [μm]', fontweight='bold')
    plt.ylabel('N_grain', fontweight='bold')
    plt.title('Power-law with Exponential Cut-off Fit (Nozawa+2007)', fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')

    out_dir = _paths.get_plots_dir('tools') if outdir is None else outdir
    out_path = f'{out_dir}/nozawa2007_gsd_fit.png'
    plt.savefig(out_path, dpi=200, format='png')
    print(f'  saved {out_path}')
    return out_path


if __name__ == '__main__':
    main()
