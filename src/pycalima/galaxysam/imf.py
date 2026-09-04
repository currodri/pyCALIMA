"""
Initial Mass Function (IMF) utilities for stellar population synthesis.

Implements different IMF prescriptions including Salpeter and Chabrier IMFs,
and provides methods for IMF weighting and integration.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings


class IMF:
    """
    Initial Mass Function base class.
    
    Provides methods for calculating IMF values, normalizations, and transformations.
    """
    
    def __init__(self, mmin=0.1, mmax=100.0):
        """
        Initialize IMF.
        
        Parameters
        ----------
        mmin : float
            Minimum mass in solar masses
        mmax : float
            Maximum mass in solar masses
        """
        self.mmin = mmin
        self.mmax = mmax
        self._norm = 1.0
        self.normalize()
    
    def phi(self, m):
        """
        IMF density function (number of stars per unit mass).
        Must be implemented by subclasses.
        
        Parameters
        ----------
        m : float or array
            Stellar mass in solar masses
            
        Returns
        -------
        float or array
            IMF value (d N/d log m)
        """
        raise NotImplementedError("Subclasses must implement phi()")
    
    def normalize(self):
        """Normalize the IMF so integral equals 1."""
        # Default: already normalized
        self._norm = 1.0
    
    def __call__(self, m):
        """Evaluate IMF at mass m."""
        return self.phi(m) / self._norm


class SalpeterIMF(IMF):
    """
    Salpeter IMF with power-law form phi(m) = m^alpha.
    
    Default: alpha = -2.35 (Salpeter 1955)
    """
    
    def __init__(self, alpha=-2.35, mmin=0.1, mmax=100.0):
        """
        Initialize Salpeter IMF.
        
        Parameters
        ----------
        alpha : float
            Power-law exponent (default: -2.35)
        mmin : float
            Minimum mass in solar masses
        mmax : float
            Maximum mass in solar masses
        """
        self.alpha = alpha
        super().__init__(mmin=mmin, mmax=mmax)
    
    def phi(self, m):
        """Salpeter IMF: phi(m) = m^alpha"""
        m_arr = np.asarray(m)
        values = m_arr ** self.alpha
        if np.ndim(m_arr) == 0:
            return float(values)
        return values
    
    def normalize(self):
        """Normalize Salpeter IMF."""
        # Integral of m^alpha from mmin to mmax
        if self.alpha == -1.0:
            # Special case: integral is logarithmic
            norm_integral = np.log(self.mmax / self.mmin)
        else:
            # General case: m^(alpha+1) / (alpha+1)
            norm_integral = (self.mmax**(self.alpha + 1) - 
                           self.mmin**(self.alpha + 1)) / (self.alpha + 1)
        self._norm = norm_integral


class ChabrierIMF(IMF):
    """
    Chabrier (2003) IMF - piecewise function combining:
    - Lognormal form for m < 1 Msun
    - Power law m^-2.3 for m >= 1 Msun
    """
    
    def __init__(self, mmin=0.1, mmax=100.0):
        """
        Initialize Chabrier IMF.
        
        Parameters
        ----------
        mmin : float
            Minimum mass in solar masses
        mmax : float
            Maximum mass in solar masses
        """
        super().__init__(mmin=mmin, mmax=mmax)
    
    def phi(self, m):
        """Chabrier IMF combining lognormal and power law components."""
        m_arr = np.asarray(m)
        m = np.atleast_1d(m_arr)
        result = np.zeros_like(m, dtype=float)
        
        # Lognormal component for m < 1 Msun
        low_mass = m < 1.0
        if np.any(low_mass):
            m_low = m[low_mass]
            lnm = np.log(m_low)
            result[low_mass] = (0.141 / m_low * 
                               np.exp(-(lnm + 0.405)**2 / (2 * 0.288**2)))
        
        # Power law component for m >= 1 Msun
        high_mass = m >= 1.0
        if np.any(high_mass):
            m_high = m[high_mass]
            result[high_mass] = 0.061 * m_high**(-2.3)
        
        if np.ndim(m_arr) == 0:
            return float(result[0])
        return result
    
    def normalize(self):
        """Normalize Chabrier IMF by integration."""
        try:
            norm_integral, _ = quad(self.phi, self.mmin, self.mmax, limit=100)
            self._norm = norm_integral
        except Exception as e:
            warnings.warn(f"Normalization integration failed: {e}. Using default.")
            self._norm = 0.5  # Empirical value


class BrokenPowerLawIMF(IMF):
    """
    Broken power-law IMF with multiple slope segments.
    
    Example: 
    - m^-1.3 for 0.1 < m < 0.5 Msun
    - m^-2.3 for 0.5 < m < 100 Msun
    """
    
    def __init__(self, alpha_slopes, mass_bounds, mmin=0.1, mmax=100.0):
        """
        Initialize broken power-law IMF.
        
        Parameters
        ----------
        alpha_slopes : list
            List of power-law slopes
        mass_bounds : list
            List of mass boundaries (one more element than alpha_slopes)
            Must have form [mmin, m1, m2, ..., mmax]
        mmin : float
            Minimum mass in solar masses
        mmax : float
            Maximum mass in solar masses
        """
        if len(mass_bounds) != len(alpha_slopes) + 1:
            raise ValueError(
                f"Number of mass bounds ({len(mass_bounds)}) must be "
                f"one more than slopes ({len(alpha_slopes)})"
            )
        
        self.alpha_slopes = alpha_slopes
        self.mass_bounds = mass_bounds
        super().__init__(mmin=mmin, mmax=mmax)
    
    def phi(self, m):
        """Evaluate broken power-law IMF."""
        m_arr = np.asarray(m)
        m = np.atleast_1d(m_arr)
        result = np.zeros_like(m, dtype=float)
        
        for i in range(len(self.alpha_slopes)):
            m_low = self.mass_bounds[i]
            m_high = self.mass_bounds[i + 1]
            alpha = self.alpha_slopes[i]
            
            in_range = (m >= m_low) & (m < m_high)
            if np.any(in_range):
                # Normalize at lower bound to ensure continuity
                norm_at_low = m_low ** alpha
                result[in_range] = (m[in_range] / m_low) ** alpha / norm_at_low
        
        if np.ndim(m_arr) == 0:
            return float(result[0])
        return result
    
    def normalize(self):
        """Normalize broken power-law IMF by integration."""
        try:
            norm_integral, _ = quad(self.phi, self.mmin, self.mmax, limit=100)
            self._norm = norm_integral
        except Exception as e:
            warnings.warn(f"Normalization integration failed: {e}. Using default.")
            self._norm = 1.0


def create_imf(imf_type='chabrier', **kwargs):
    """
    Factory function to create IMF objects.
    
    Parameters
    ----------
    imf_type : str
        Type of IMF: 'salpeter', 'chabrier', or 'broken_powerlaw'
    **kwargs : dict
        Additional keyword arguments for specific IMF types
        
    Returns
    -------
    IMF
        Initialized IMF object
    """
    if imf_type.lower() == 'salpeter':
        alpha = kwargs.pop('alpha', -2.35)
        return SalpeterIMF(alpha=alpha, **kwargs)
    elif imf_type.lower() == 'chabrier':
        return ChabrierIMF(**kwargs)
    elif imf_type.lower() == 'broken_powerlaw':
        alpha_slopes = kwargs.pop('alpha_slopes')
        mass_bounds = kwargs.pop('mass_bounds')
        return BrokenPowerLawIMF(alpha_slopes, mass_bounds, **kwargs)
    else:
        raise ValueError(f"Unknown IMF type: {imf_type}")


def imf_weighted_quantity(masses, quantities, imf, normalize_by_mass=False):
    """
    Calculate IMF-weighted average of a quantity.
    
    Parameters
    ----------
    masses : array
        Stellar masses in solar masses
    quantities : array
        Quantities to average (same shape as masses)
    imf : IMF
        IMF object
    normalize_by_mass : bool
        If True, return mass-weighted average; if False, return number-weighted
        
    Returns
    -------
    float
        IMF-weighted average
    """
    masses = np.atleast_1d(masses)
    quantities = np.atleast_1d(quantities)
    
    weights = imf(masses)
    
    if normalize_by_mass:
        weights *= masses
    
    return np.sum(weights * quantities) / np.sum(weights)
