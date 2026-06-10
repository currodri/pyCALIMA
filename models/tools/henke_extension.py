
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d

# Physical constants
C_EV_UM = 1.23984193  # Energy (eV) * Wavelength (um)
R_E = 2.81794e-13     # Classical electron radius in cm
N_A = 6.02214e23      # Avogadro constant

# Atomic weights (g/mol)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38
}

class HenkeExtension:
    def __init__(self, dat_path='external_data/henke/f1f2_Henke.dat'):
        self.atomic_factors = {}
        if os.path.exists(dat_path):
            self.parse_henke_file(dat_path)
        else:
            print(f"Warning: Henke data file not found at {dat_path}")

    def parse_henke_file(self, path):
        """Parse the consolidated f1f2_Henke.dat file."""
        with open(path, 'r') as f:
            content = f.read()
        
        # Split by element sections
        sections = content.split('#S')
        for sec in sections[1:]: # Skip preamble
            lines = sec.strip().split('\n')
            # Header line: e.g. " 6  C"
            header = lines[0].split()
            z = int(header[0])
            symbol = header[1]
            
            energies, f1s, f2s = [], [], []
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    energies.append(float(parts[0]))
                    f1s.append(float(parts[1]))
                    f2s.append(float(parts[2]))
            
            self.atomic_factors[symbol] = {
                'energy_ev': np.array(energies),
                'f1': np.array(f1s),
                'f2': np.array(f2s),
                'f1_interp': interp1d(energies, f1s, kind='linear', fill_value='extrapolate'),
                'f2_interp': interp1d(energies, f2s, kind='linear', fill_value='extrapolate')
            }

    def compute_refractive_index(self, composition, density, wavelength_um):
        """
        Compute complex refractive index m = n - ik using Henke factors.
        
        composition: dict e.g. {'Mg': 0.5, 'Fe': 0.5, 'Si': 1, 'O': 3}
        density: float in g/cm3
        wavelength_um: float or array in microns
        """
        wavelength_cm = np.atleast_1d(wavelength_um) * 1e-4
        energy_ev = C_EV_UM / np.atleast_1d(wavelength_um)
        
        # Molar mass of the compound
        M = sum(ATOMIC_WEIGHTS[el] * count for el, count in composition.items())
        
        # Total atomic number density sum(n_i f_i)
        # n_i = (rho * N_A / M) * count_i
        sum_f1 = np.zeros_like(energy_ev)
        sum_f2 = np.zeros_like(energy_ev)
        
        for el, count in composition.items():
            if el not in self.atomic_factors:
                raise ValueError(f"Atomic factors for {el} not loaded.")
            
            # Interpolate f1, f2 at given energies
            f1 = self.atomic_factors[el]['f1_interp'](energy_ev)
            f2 = self.atomic_factors[el]['f2_interp'](energy_ev)
            
            sum_f1 += count * f1
            sum_f2 += count * f2
            
        prefactor = (R_E * (wavelength_cm**2)) / (2.0 * np.pi) * (density * N_A / M)
        
        delta = prefactor * sum_f1
        beta = prefactor * sum_f2
        
        n = 1.0 - delta
        k = beta
        
        m = n - 1j * k
        return m if len(m) > 1 else m[0]

# Metadata for Kitzmann & Heng 2018 species
DUST_METADATA = {
    'Al2O3': {'comp': {'Al': 2, 'O': 3}, 'rho': 3.95},
    'C': {'comp': {'C': 1}, 'rho': 1.81},
    'CaTiO3': {'comp': {'Ca': 1, 'Ti': 1, 'O': 3}, 'rho': 3.98},
    'Cr': {'comp': {'Cr': 1}, 'rho': 7.19},
    'Fe': {'comp': {'Fe': 1}, 'rho': 7.87},
    'Fe2O3': {'comp': {'Fe': 2, 'O': 3}, 'rho': 5.24},
    'Fe2SiO4': {'comp': {'Fe': 2, 'Si': 1, 'O': 4}, 'rho': 4.39},
    'FeO': {'comp': {'Fe': 1, 'O': 1}, 'rho': 5.74},
    'FeS': {'comp': {'Fe': 1, 'S': 1}, 'rho': 4.84},
    'H2O_s': {'comp': {'H': 2, 'O': 1}, 'rho': 0.92},
    'KCl': {'comp': {'K': 1, 'Cl': 1}, 'rho': 1.98},
    'Mg04Fe06SiO3_amorph_glass': {'comp': {'Mg': 0.4, 'Fe': 0.6, 'Si': 1, 'O': 3}, 'rho': 3.7},
    'Mg05Fe05SiO3_amorph_glass': {'comp': {'Mg': 0.5, 'Fe': 0.5, 'Si': 1, 'O': 3}, 'rho': 3.6},
    'Mg08Fe02SiO3_amorph_glass': {'comp': {'Mg': 0.8, 'Fe': 0.2, 'Si': 1, 'O': 3}, 'rho': 3.3},
    'Mg08Fe12SiO4_amorph_glass': {'comp': {'Mg': 0.8, 'Fe': 1.2, 'Si': 1, 'O': 4}, 'rho': 3.9},
    'Mg2SiO4_amorph_sol-gel': {'comp': {'Mg': 2, 'Si': 1, 'O': 4}, 'rho': 3.22},
    'MgAl2O4': {'comp': {'Mg': 1, 'Al': 2, 'O': 4}, 'rho': 3.58},
    'MgFeSiO4_amorph_glass': {'comp': {'Mg': 1, 'Fe': 1, 'Si': 1, 'O': 4}, 'rho': 3.71},
    'MgO': {'comp': {'Mg': 1, 'O': 1}, 'rho': 3.58},
    'MgSiO3_amorph_glass': {'comp': {'Mg': 1, 'Si': 1, 'O': 3}, 'rho': 3.2},
    'MgSiO3_amorph_sol-gel': {'comp': {'Mg': 1, 'Si': 1, 'O': 3}, 'rho': 3.2},
    'MnS': {'comp': {'Mn': 1, 'S': 1}, 'rho': 3.99},
    'Na2S': {'comp': {'Na': 2, 'S': 1}, 'rho': 1.86},
    'NaCl': {'comp': {'Na': 1, 'Cl': 1}, 'rho': 2.16},
    'SiC': {'comp': {'Si': 1, 'C': 1}, 'rho': 3.21},
    'SiO': {'comp': {'Si': 1, 'O': 1}, 'rho': 2.13},
    'SiO2_alpha': {'comp': {'Si': 1, 'O': 2}, 'rho': 2.65},
    'SiO2_amorph': {'comp': {'Si': 1, 'O': 2}, 'rho': 2.2},
    'TiC': {'comp': {'Ti': 1, 'C': 1}, 'rho': 4.93},
    'TiO2_anatase': {'comp': {'Ti': 1, 'O': 2}, 'rho': 4.23},
    'Titan_tholin': {'comp': {'C': 1, 'H': 1.5, 'N': 0.1}, 'rho': 1.45},
    'ZnS': {'comp': {'Zn': 1, 'S': 1}, 'rho': 4.09},
}
