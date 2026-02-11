import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data
columns = ['Grain Size', 'N_grain']
mg2sio4_data = pd.read_csv('nozawa_2007_Mg2SiO4_gsd.dat', names=columns)
c_data = pd.read_csv('nozawa_2007_C_gsd.dat', names=columns)

# Ensure numerical data is in float format
mg2sio4_data['Grain Size'] = pd.to_numeric(mg2sio4_data['Grain Size'], errors='coerce')
mg2sio4_data['N_grain'] = pd.to_numeric(mg2sio4_data['N_grain'], errors='coerce')
c_data['Grain Size'] = pd.to_numeric(c_data['Grain Size'], errors='coerce')
c_data['N_grain'] = pd.to_numeric(c_data['N_grain'], errors='coerce')

# Define a power-law with an exponential cut-off function
def power_law_exponential(x, A, alpha, x_cut):
    return np.log10(A * (x ** alpha) * np.exp(-x / x_cut))

# Fit function to data
plt.figure()

# Choose different line colours and marker colour for each material
line_color = ['red', 'blue']
marker_color = ['orange', 'green']

for material, data, color, mcolor in zip(['Mg2SiO4', 'C'], [mg2sio4_data, c_data], line_color, marker_color):
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
    plt.plot(x_fit, y_fit, label=f'Fit: A={A_fit:.2e}, alpha={alpha_fit:.2f}, x_cut={x_cut_fit:.2e}', color=color)

    print(f'{material} Fit Parameters: A={A_fit:.2e}, alpha={alpha_fit:.2f}, x_cut={x_cut_fit:.2e}')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e14,1e20])
plt.xlabel('Grain Size [μm]', fontweight='bold')
plt.ylabel('N_grain', fontweight='bold')
plt.title(f'Power-law with Exponential Cut-off Fit for {material}', fontweight='bold')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.savefig('nozawa2007_gsd_fit.png',dpi=200,format='png')

