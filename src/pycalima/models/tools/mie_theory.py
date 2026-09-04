
import numpy as np
import miepython
from scipy.interpolate import interp1d
import pandas as pd
import os

from pycalima.models.tools.henke_extension import HenkeExtension, DUST_METADATA

class MieTheory:
    """
    A class to compute optical properties of dust grains using Mie theory via miepython.
    Supports both isotropic and anisotropic grains (using the 1/3-2/3 approximation).
    Can be extended to X-ray regimes using Henke atomic scattering factors.
    """

    def __init__(self):
        self.dielectric_data = {}
        self.henke = None

    def _init_henke(self):
        if self.henke is None:
            # Assumes the script is run from a location where this path makes sense
            # or the file is in the same directory as henke_extension.py
            self.henke = HenkeExtension()

    def load_dielectric_constants(self, file_path, species_label):
        """
        Load dielectric constants from a Draine-style file.
        
        The file should have columns: wave(um), eps_1-1, eps_2, Re(n)-1, Im(n)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dielectric file not found: {file_path}")

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Find where the data starts
        data_start = 0
        for i, line in enumerate(lines):
            if 'wave' in line.lower() or 'w(micron)' in line.lower():
                data_start = i + 1
                break
            parts = line.split()
            if not parts: continue
            try:
                float(parts[0])
                if len(parts) >= 3 and '=' not in line:
                    data_start = i
                    break
            except ValueError:
                continue

        table_data = []
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    vals = [float(p) for p in parts[:5]]
                    table_data.append(vals)
                except ValueError:
                    continue
        
        df = pd.DataFrame(table_data, columns=['wavelength_um', 'eps1_minus_1', 'eps2', 'n_minus_1', 'k'])
        df['n'] = df['n_minus_1'] + 1
        
        # Sort by wavelength
        df = df.sort_values('wavelength_um')
        
        # Create interpolators
        self.dielectric_data[species_label] = {
            'wavelengths': df['wavelength_um'].values,
            'n_interp': interp1d(df['wavelength_um'].values, df['n'].values, kind='linear', fill_value='extrapolate'),
            'k_interp': interp1d(df['wavelength_um'].values, df['k'].values, kind='linear', fill_value='extrapolate')
        }
        print(f"Loaded dielectric constants for {species_label} from {file_path}")

    def load_kitzmann_heng(self, file_path, species_label):
        """
        Load dielectric constants from Kitzmann & Heng (2018) format.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        table_data = []
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vals = [float(p) for p in parts[:3]]
                    table_data.append(vals)
                except ValueError:
                    continue
        
        df = pd.DataFrame(table_data, columns=['wavelength_um', 'n', 'k'])
        df = df.sort_values('wavelength_um')
        
        self.dielectric_data[species_label] = {
            'wavelengths': df['wavelength_um'].values,
            'n_interp': interp1d(df['wavelength_um'].values, df['n'].values, kind='linear', fill_value='extrapolate'),
            'k_interp': interp1d(df['wavelength_um'].values, df['k'].values, kind='linear', fill_value='extrapolate')
        }
        print(f"Loaded {species_label} from {file_path} (Kitzmann & Heng format)")

    def get_refractive_index(self, species_label, wavelength_um, extend_xrays=True):
        """Get interpolated complex refractive index m = n - ik."""
        if species_label not in self.dielectric_data:
            raise ValueError(f"Species {species_label} not loaded.")
        
        # Check if wavelength is below the minimum tabulated
        w_min = self.dielectric_data[species_label]['wavelengths'].min()
        
        if extend_xrays and wavelength_um < w_min:
            # Try to use Henke extension if metadata is available
            # Map labels that might have suffixes back to base names for metadata
            base_label = species_label.replace('_pa', '').replace('_pe', '')
            if base_label in DUST_METADATA:
                self._init_henke()
                meta = DUST_METADATA[base_label]
                return self.henke.compute_refractive_index(meta['comp'], meta['rho'], wavelength_um)
        
        n = self.dielectric_data[species_label]['n_interp'](wavelength_um)
        k = self.dielectric_data[species_label]['k_interp'](wavelength_um)
        return n - 1j * k

    def compute_efficiencies(self, radius_um, wavelength_um, m, use_fast_path=True):
        """
        Compute Qabs, Qsca, and g for a single grain.
        """
        x = 2 * np.pi * radius_um / wavelength_um
        abs_m = np.abs(m)
        mx = abs_m * x
        
        if use_fast_path and mx > 1000:
            m_minus_1_x = np.abs(m - 1) * x
            k = -np.imag(m)
            n = np.real(m)

            if m_minus_1_x < 0.001:
                qabs = (8.0/3.0) * x * k
                m_minus_1_abs_sq = (n - 1.0)**2 + k**2
                qsca = (32.0 * m_minus_1_abs_sq * (x**4)) / (27.0 + 16.0 * (x**2))
                g = (0.3 * (x**2)) / (1.0 + 0.3 * (x**2))
                return qabs, qsca, g
            else:
                n_int = 40
                ws = np.linspace(0.5/n_int, 1.0-0.5/n_int, n_int)
                d_w = 1.0 / n_int
                qabs_sum, qrefl_sum, grefl_sum, qrefr_sum, grefr_sum = 0, 0, 0, 0, 0
                for w in ws:
                    sin_theta_sq = 1.0 - w**2
                    cos_theta_prime = np.sqrt(1.0 - sin_theta_sq / m**2)
                    m_cos_tp = m * cos_theta_prime
                    rs = (w - m_cos_tp) / (w + m_cos_tp)
                    rp = (m * w - cos_theta_prime) / (m * w + cos_theta_prime)
                    R_w = 0.5 * (np.abs(rs)**2 + np.abs(rp)**2)
                    tau_w = 4.0 * x * (-np.imag(m * cos_theta_prime))
                    tau_w = np.clip(tau_w, 0, 100)
                    exp_tau = np.exp(-tau_w)
                    denom = 1.0 - R_w * exp_tau
                    qabs_w = (1.0 - R_w) * (1.0 - exp_tau) / denom
                    qabs_sum += qabs_w * 2.0 * w * d_w
                    qrefl_w = R_w
                    qrefl_sum += qrefl_w * 2.0 * w * d_w
                    grefl_sum += qrefl_w * (1.0 - 2.0 * w**2) * 2.0 * w * d_w
                    qrefr_w = (1.0 - R_w)**2 * exp_tau / denom
                    qrefr_sum += qrefr_w * 2.0 * w * d_w
                    theta = np.arccos(w)
                    theta_prime = np.arccos(np.real(cos_theta_prime / np.abs(cos_theta_prime)))
                    grefr_sum += qrefr_w * np.cos(2.0 * (theta - theta_prime)) * 2.0 * w * d_w
                qext = 2.0
                qabs = qabs_sum
                qsca = 1.0 + qrefl_sum + qrefr_sum
                g = (1.0 + grefl_sum + grefr_sum) / qsca
                g = np.clip(g, -1.0, 1.0)
                return qabs, qsca, g

        d = 2 * radius_um
        qext, qsca, qback, g = miepython.efficiencies(m, d, wavelength_um)
        qabs = qext - qsca
        return qabs, qsca, g

    def compute_grain_properties(self, radius_um, wavelength_um, species_info, use_fast_path=True, extend_xrays=True):
        """
        Compute properties for a grain, possibly with the 1/3-2/3 approximation.
        """
        if isinstance(species_info, str):
            m = self.get_refractive_index(species_info, wavelength_um, extend_xrays=extend_xrays)
            return self.compute_efficiencies(radius_um, wavelength_um, m, use_fast_path=use_fast_path)
        elif isinstance(species_info, dict):
            m_pa = self.get_refractive_index(species_info['parallel'], wavelength_um, extend_xrays=extend_xrays)
            m_pe = self.get_refractive_index(species_info['perpendicular'], wavelength_um, extend_xrays=extend_xrays)
            qabs_pa, qsca_pa, g_pa = self.compute_efficiencies(radius_um, wavelength_um, m_pa, use_fast_path=use_fast_path)
            qabs_pe, qsca_pe, g_pe = self.compute_efficiencies(radius_um, wavelength_um, m_pe, use_fast_path=use_fast_path)
            qabs = (1/3) * qabs_pa + (2/3) * qabs_pe
            qsca = (1/3) * qsca_pa + (2/3) * qsca_pe
            g = ((1/3) * g_pa * qsca_pa + (2/3) * g_pe * qsca_pe) / qsca if qsca > 0 else 0.0
            return qabs, qsca, g
        else:
            raise ValueError("Invalid species_info format.")

def read_draine_q_table(filename):
    """
    Helper to read Draine's Q tables (e.g., Sil_81, Gra_81).
    """
    data = {}
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    nrad = int(lines[3].split('=')[0].split()[0])
    nwav = int(lines[4].split('=')[0].split()[0])
    current_line = 5
    for _ in range(nrad):
        rad_line = lines[current_line]
        radius_um = float(rad_line.split('=')[0].strip())
        current_line += 2
        wavs, qabs, qsca, gs = [], [], [], []
        for _ in range(nwav):
            parts = lines[current_line].split()
            wavs.append(float(parts[0]))
            qabs.append(float(parts[1]))
            qsca.append(float(parts[2]))
            gs.append(float(parts[3]))
            current_line += 1
        df = pd.DataFrame({'wavelength_um': wavs, 'Qabs': qabs, 'Qsca': qsca, 'g': gs})
        data[radius_um] = df
        if current_line >= len(lines): break
    return data
