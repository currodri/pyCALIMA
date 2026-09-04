"""
Type Ia Supernova calculations and properties.

Implements Type Ia supernova rates, yields, and properties based on
single degenerate (SD) or double degenerate (DD) progenitor models.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from . import constants


class SNIaModel:
    """
    Type Ia Supernova model.
    
    Computes Type Ia SNe rates and yields based on progenitor population synthesis.
    """
    
    def __init__(self, asnia=0.05, imf=None):
        """
        Initialize SNIa model.
        
        Parameters
        ----------
        asnia : float
            Fraction of stars that become Type Ia SNe
        imf : IMF, optional
            Initial Mass Function object
        """
        self.asnia = asnia
        self.imf = imf
        self._tau_m_lookup = {}  # Cache for stellar lifetimes
    
    def tau_m_padova(self, mass):
        """
        Stellar lifetime using Padovani & Matteucci (1993) relation.
        
        Parameters
        ----------
        mass : float or array
            Stellar mass in solar masses
            
        Returns
        -------
        float or array
            Stellar lifetime in years
        """
        mass = np.atleast_1d(mass)
        result = np.zeros_like(mass, dtype=float)
        
        # Critical mass for Padova relation
        mcrit = 10.0**(7.764 - 1.79 / 0.2232) * 1.1
        
        # Low mass stars (m < mcrit): long lifetime
        low_mass = mass < mcrit
        if np.any(low_mass):
            result[low_mass] = 1e12
        
        # High mass stars (m >= mcrit)
        high_mass = mass >= mcrit
        if np.any(high_mass):
            m = mass[high_mass]
            a0 = (1.338 - np.sqrt(1.79 - 0.2232 * (7.764 - np.log10(m)))) / 0.1116
            result[high_mass] = 10.0**a0
        
        # Additional scaling for specific masses
        scale_mass = mass >= 6.6
        if np.any(scale_mass):
            m = mass[scale_mass]
            result[scale_mass] = (1.2 * m**(-1.85) + 0.003) * 1e9
        
        # Apply rescaling factor
        result = result * 1.3
        
        return result if result.size > 1 else result[0]
    
    def tau_m_simple(self, mass):
        """
        Simple stellar lifetime relation.
        
        Parameters
        ----------
        mass : float or array
            Stellar mass in solar masses
            
        Returns
        -------
        float or array
            Stellar lifetime in years
        """
        mass = np.atleast_1d(mass)
        result = np.zeros_like(mass, dtype=float)
        
        # For m > 8 Msun
        high = mass > 8.0
        if np.any(high):
            m = mass[high]
            result[high] = (1.2 * m**(-1.85) + 0.003) * 1e9
        
        # For m <= 80 Msun
        mid = mass <= 80.0
        if np.any(mid):
            m = mass[mid]
            result[mid] = (5.0 * m**(-2.7) + 0.012) * 1e9
        
        return result if result.size > 1 else result[0]
    
    def tau_m_rood(self, mass):
        """
        Stellar lifetime using Rood (1972) relation via Greggio & Renzini (1983).
        
        Parameters
        ----------
        mass : float or array
            Stellar mass in solar masses
            
        Returns
        -------
        float or array
            Stellar lifetime in years
        """
        mass = np.atleast_1d(mass)
        result = np.zeros_like(mass, dtype=float)
        
        # For m <= 8 Msun
        low_mass = mass <= 8.0
        if np.any(low_mass):
            m = mass[low_mass]
            log_tau = 10.0 - 4.319 * np.log10(m) + 1.543 * np.log10(m)**2
            result[low_mass] = 10.0**log_tau
        
        # For m > 8 Msun: use lifetime at 8 Msun
        high_mass = mass > 8.0
        if np.any(high_mass):
            log_tau_8 = 10.0 - 4.319 * np.log10(8.0) + 1.543 * np.log10(8.0)**2
            result[high_mass] = 10.0**log_tau_8
        
        return result if result.size > 1 else result[0]
    
    def snia_rate_delay_time(self, time, tau_min=100e6, tau_max=10e9):
        """
        Type Ia SNe rate using delay time distribution (DTD).
        
        Implements power-law DTD: dN/dt ∝ t^-1.1
        
        Parameters
        ----------
        time : float or array
            Time in years since star formation
        tau_min : float
            Minimum delay time (typical: 100 Myr)
        tau_max : float
            Maximum delay time (typical: 10 Gyr)
            
        Returns
        -------
        float or array
            Fraction of initial stellar population exploding as SNIa
        """
        time = np.atleast_1d(time)
        result = np.zeros_like(time, dtype=float)
        
        # Power-law index (Maoz & Mannucci 2012)
        alpha_ddt = 1.1
        
        # Normalize DTD
        # Integral of t^-1.1 from tau_min to tau_max
        if alpha_ddt == 1.0:
            norm = np.log(tau_max / tau_min)
        else:
            norm = (tau_max**(1 - alpha_ddt) - tau_min**(1 - alpha_ddt)) / (1 - alpha_ddt)
        
        in_range = (time >= tau_min) & (time <= tau_max)
        if np.any(in_range):
            result[in_range] = (time[in_range]**(-alpha_ddt)) / norm * self.asnia
        
        return result if result.size > 1 else result[0]
    
    def snia_rate_progenitor_age(self, stellar_age, metallicity=None):
        """
        Type Ia SNe rate from progenitor age.
        
        Parameters
        ----------
        stellar_age : float
            Age of stellar population in years
        metallicity : float, optional
            Metallicity (not used in current implementation)
            
        Returns
        -------
        float
            Type Ia SNe rate (fraction of stellar population)
        """
        # Implement progenitor model-based rate
        return self.snia_rate_delay_time(stellar_age)
    
    def yields_snia(self, model='nomoto84'):
        """
        Type Ia supernova yields.
        
        Parameters
        ----------
        model : str
            Yield model ('nomoto84', 'iwamoto99', 'thielemann03')
            
        Returns
        -------
        dict
            Dictionary with element yields in solar masses
        """
        # Example yields (simplified)
        yields_db = {
            'nomoto84': {
                'H': 0.0,
                'He': 0.0,
                'C': 0.074,
                'N': 0.0,
                'O': 0.503,
                'F': 0.0,
                'Ne': 0.0,
                'Mg': 0.011,
                'Si': 0.107,
                'S': 0.013,
                'Fe': 0.744,
            },
            'iwamoto99': {
                'H': 0.0,
                'He': 0.0,
                'C': 0.08,
                'N': 0.0,
                'O': 0.52,
                'F': 0.0,
                'Ne': 0.0,
                'Mg': 0.012,
                'Si': 0.11,
                'S': 0.014,
                'Fe': 0.77,
            },
        }
        
        if model not in yields_db:
            raise ValueError(f"Unknown model: {model}. Available: {list(yields_db.keys())}")
        
        return yields_db[model]
    
    def snia_mass_return(self, asnia=None):
        """
        Total mass returned by Type Ia SNe per unit stellar mass formed.
        
        Parameters
        ----------
        asnia : float, optional
            Fraction of stars that become SNIa (overrides instance value)
            
        Returns
        -------
        float
            Total mass returned per unit stellar mass formed
        """
        if asnia is None:
            asnia = self.asnia
        
        # Typical Type Ia explosion ejects ~1.4 Msun
        return asnia * 1.4
    
    def inverse_mass_sampler(self, s, rmu, rml, n_samples=100, seed=None):
        """
        Sample stellar masses from a power-law distribution using inverse method.
        
        Used for Salpeter-like IMF sampling.
        
        Parameters
        ----------
        s : float
            Power-law index
        rmu : float
            Upper mass limit
        rml : float
            Lower mass limit
        n_samples : int
            Number of samples
        seed : int, optional
            Random seed
            
        Returns
        -------
        array
            Sampled masses in solar masses
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Ensure rmu > rml
        if rmu < rml:
            rmu, rml = rml, rmu
        
        # Inverse transform sampling
        b = rmu**(1 + s) / np.abs(1 + s)
        c = rml**(1 + s) / np.abs(1 + s)
        a = np.abs(c - b)
        
        u = np.random.uniform(0, 1, n_samples)
        x = a * u + min(b, c)
        
        masses = (x * np.abs(1 + s))**(1 / (1 + s))
        
        return masses
