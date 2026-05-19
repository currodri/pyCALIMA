"""
Stellar yield models for SNII, AGB, and SNIa nucleosynthesis.

Handles reading, interpolation, and computation of yields from various
sources including Kobayashi et al., Limongi & Chieffi (LC18), and Karakas.
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from pathlib import Path
from . import constants

# Default yield data directory
DEFAULT_YIELD_DIR = Path(__file__).parent / 'yield_files' / 'yield_files'


class YieldModel:
    """Base class for stellar yield models."""
    
    def __init__(self, name, metallicity=0.02):
        """
        Initialize yield model.
        
        Parameters
        ----------
        name : str
            Model name (e.g., 'kobayashi', 'lc18', 'karakas')
        metallicity : float
            Metallicity in log(Z/Zsun) or linear Z
        """
        self.name = name
        self.metallicity = metallicity
        self.elements = None
        self.masses = None
        self.yields = None
        self.mass_loss = None
        self.final_mass = None
    
    def get_yield(self, mass, element):
        """
        Get yield for a specific mass and element.
        
        Parameters
        ----------
        mass : float
            Stellar mass in solar masses
        element : str
            Element symbol (e.g., 'Fe', 'O', 'Mg')
            
        Returns
        -------
        float
            Yield in solar masses
        """
        raise NotImplementedError("Subclasses must implement get_yield()")
    
    def interpolate_yield(self, masses, element):
        """
        Interpolate yields across mass range.
        
        Parameters
        ----------
        masses : array
            Array of stellar masses
        element : str
            Element symbol
            
        Returns
        -------
        array
            Interpolated yields
        """
        if self.masses is None or self.yields is None:
            raise ValueError("Yield data not loaded")
        
        if element not in self.elements:
            raise ValueError(f"Element {element} not in model")
        
        elem_idx = self.elements.index(element)
        
        # Sort by mass for interpolation
        sort_idx = np.argsort(self.masses)
        m_sorted = self.masses[sort_idx]
        y_sorted = self.yields[sort_idx, elem_idx]
        
        # Linear interpolation in log-log space
        log_masses = np.log10(m_sorted)
        log_yields = np.log10(y_sorted + 1e-30)  # Avoid log(0)
        
        f = interp1d(log_masses, log_yields, kind='linear', 
                     fill_value='extrapolate', bounds_error=False)
        
        log_masses_interp = np.log10(masses)
        log_yields_interp = f(log_masses_interp)
        
        return 10.0**log_yields_interp


class KobayashiYields(YieldModel):
    """
    Kobayashi et al. (2006) supernova yields.
    
    Provides SNII yields for different metallicities.
    """
    
    # Available metallicities in the model
    AVAILABLE_Z = [0.0, 0.001, 0.004, 0.008, 0.02, 0.05]
    
    # Mass grids for different metallicities
    MASS_GRIDS = {
        0.0: np.array([0.9, 1.0, 1.2, 1.5, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 
                      4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 11.0, 13.0, 
                      15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 100.0, 140.0, 140.0, 
                      150.0, 170.0, 200.0, 270.0, 300.0]),
        0.02: np.array([0.9, 1.0, 1.2, 1.5, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 
                       4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 13.0, 15.0, 
                       18.0, 20.0, 25.0, 30.0, 40.0]),
    }
    
    # Mass ranges for intermediate and massive stars
    MASS_INTERMEDIATE = {
        0.0: np.array([0.9, 1.0, 1.2, 1.5, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 
                      4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]),
        0.02: np.array([0.9, 1.0, 1.2, 1.5, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 3.5, 
                       4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]),
    }
    
    MASS_MASSIVE = {
        0.0: np.array([8.0, 10.0, 11.0, 13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 
                      40.0, 100.0, 140.0, 140.0, 150.0, 170.0, 200.0, 270.0, 300.0]),
        0.02: np.array([8.0, 10.0, 13.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0]),
    }
    
    # Mass remaining (remnant mass) values
    MASS_REMAINING = {
        0.02: np.array([0.473, 0.564, 0.574, 0.600, 0.615, 0.630, 0.640, 0.660, 
                       0.663, 0.682, 0.718, 0.792, 0.852, 0.879, 0.900, 0.929, 
                       0.963, 1.010, 1.120, 1.150, 1.600, 1.500, 1.580, 1.550, 
                       1.804, 2.100, 2.210]),
    }
    
    def __init__(self, metallicity=0.02, data_dir=None):
        """
        Initialize Kobayashi yields model.
        
        Parameters
        ----------
        metallicity : float
            Metallicity (linear Z value, e.g., 0.02 for solar)
        data_dir : Path or str, optional
            Directory containing yield files (defaults to yield_files folder)
        """
        super().__init__('kobayashi', metallicity)
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_YIELD_DIR
        
        # Find closest available metallicity
        self.metallicity_actual = self._get_closest_metallicity(metallicity)
        
        # Element tracking (H prefix distinguishes from IDL 'p' for proton)
        self.elements = constants.ELEMENTS_LC18
        
        self._load_yields()
    
    def _get_closest_metallicity(self, z):
        """Find closest available metallicity."""
        z_available = np.array(self.AVAILABLE_Z)
        idx = np.argmin(np.abs(z_available - z))
        return z_available[idx]
    
    def _load_yields(self):
        """Load yield data for the selected metallicity from default folder."""
        # Try to auto-load simplified yield file from yield_files folder
        z_formatted = f"{self.metallicity_actual:.6g}".lstrip('0')
        if z_formatted.startswith('.'):
            z_formatted = '0' + z_formatted
        
        # Look for simplified SNII file
        snii_file = self.data_dir / f'kobayashi13snii_z{z_formatted}_simplified.txt'
        if snii_file.exists():
            self.load_from_file(snii_file)
        else:
            # Fallback: create placeholder structure
            if self.metallicity_actual in self.MASS_GRIDS:
                self.masses = self.MASS_GRIDS[self.metallicity_actual]
                self.yields = np.zeros((len(self.masses), len(self.elements)))
            else:
                self.masses = np.array([])
                self.yields = np.array([]).reshape(0, len(self.elements))
    
    def load_from_file(self, filename):
        """
        Load yields from ASCII file.
        
        Parameters
        ----------
        filename : Path or str
            Path to yield file
        """
        filename = Path(filename)
        
        if not filename.exists():
            raise FileNotFoundError(f"Yield file not found: {filename}")
        
        # Read file and parse yields
        # Format varies by source - implement parsing for specific format
        df = pd.read_csv(filename, sep=r'\s+', comment='#', engine='python')
        
        # Extract masses and yields
        self.masses = df.iloc[:, 0].values
        self.yields = df.iloc[:, 1:].values
    
    def get_yield(self, mass, element):
        """Get yield for specific mass and element."""
        return self.interpolate_yield(np.array([mass]), element)[0]


class LC18Yields(YieldModel):
    """
    Limongi & Chieffi (2018) supernova yields.
    
    Provides SNII yields for different metallicities and rotation rates.
    """
    
    AVAILABLE_Z_LOG = [-3.0, -2.0, -1.0, -0.6, -0.3, 0.0, 0.3]
    AVAILABLE_VELOCITIES = [0, 25, 50, 75, 100, 150, 200, 250, 300]
    
    def __init__(self, metallicity_log=-0.3, velocity=0, data_dir=None):
        """
        Initialize LC18 yields model.
        
        Parameters
        ----------
        metallicity_log : float
            Metallicity in log(Z/Zsun)
        velocity : float
            Rotation velocity in km/s
        data_dir : Path or str, optional
            Directory containing yield files (defaults to yield_files folder)
        """
        # Convert to linear Z
        z_sun = constants.ZSUN_ASPLUND
        metallicity = 10.0**metallicity_log * z_sun
        
        super().__init__('lc18', metallicity)
        self.metallicity_log = metallicity_log
        self.velocity = velocity
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_YIELD_DIR
        
        self.elements = constants.ELEMENTS_LC18
        self._load_yields()
    
    def _load_yields(self):
        """Load yield data for selected metallicity and velocity from default folder."""
        # Try to auto-load simplified yield file from yield_files folder
        # LC18 files are named like: limongichieffi_z-0.3_vel150_simplified.txt
        logz_str = f"{self.metallicity_log:.1f}".replace('-', '-')
        vel_str = int(self.velocity)
        
        lc18_file = self.data_dir / f'limongichieffi_z{logz_str}_vel{vel_str}_simplified.txt'
        if lc18_file.exists():
            self.load_from_file(lc18_file)
        else:
            # Placeholder if file not found
            self.masses = np.array([])
            self.yields = np.array([]).reshape(0, len(self.elements))
    
    def load_from_file(self, filename):
        """Load yields from ASCII file."""
        filename = Path(filename)
        
        if not filename.exists():
            raise FileNotFoundError(f"Yield file not found: {filename}")
        
        df = pd.read_csv(filename, sep=r'\s+', comment='#', engine='python')
        self.masses = df.iloc[:, 0].values
        self.yields = df.iloc[:, 1:].values
    
    def get_yield(self, mass, element):
        """Get yield for specific mass and element."""
        return self.interpolate_yield(np.array([mass]), element)[0]


class KarakasYields(YieldModel):
    """
    Karakas (2010) AGB yield grid.
    
    Provides AGB nucleosynthesis yields.
    """
    
    AVAILABLE_Z = [0.001, 0.004, 0.008, 0.02]
    
    def __init__(self, metallicity=0.02, data_dir=None):
        """
        Initialize Karakas yields model.
        
        Parameters
        ----------
        metallicity : float
            Metallicity (linear Z)
        data_dir : Path or str, optional
            Directory containing yield files (defaults to yield_files folder)
        """
        super().__init__('karakas', metallicity)
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_YIELD_DIR
        
        self.elements = constants.ELEMENTS_LC18
        self._load_yields()
    
    def _load_yields(self):
        """Load yield data for selected metallicity from default folder."""
        # Try to auto-load simplified yield file from yield_files folder
        # Find closest available metallicity
        z_available = np.array(self.AVAILABLE_Z)
        idx = np.argmin(np.abs(z_available - self.metallicity))
        z_closest = z_available[idx]
        
        # Karakas files are named like: karakas_z0.02_simplified.txt
        kar_file = self.data_dir / f'karakas_z{z_closest}_simplified.txt'
        if kar_file.exists():
            self.load_from_file(kar_file)
        else:
            # Placeholder if file not found
            self.masses = np.array([])
            self.yields = np.array([]).reshape(0, len(self.elements))
    
    def load_from_file(self, filename):
        """Load yields from ASCII file."""
        filename = Path(filename)
        
        if not filename.exists():
            raise FileNotFoundError(f"Yield file not found: {filename}")
        
        # Parse Karakas-format file
        # Columns: M0 Z0 M1 El Yield M(i)lost M(i)0 M(i)lostall
        df = pd.read_csv(filename, sep=r'\s+', comment='#', engine='python')
        
        # Extract unique masses and reshape yields
        masses_unique = np.unique(df.iloc[:, 0].values)
        n_masses = len(masses_unique)
        n_elements = len(self.elements)
        
        self.masses = masses_unique
        self.yields = np.zeros((n_masses, n_elements))
        
        # Fill yields table from file
        for i, element in enumerate(self.elements):
            mask = df.iloc[:, 3] == element
            if np.any(mask):
                elem_yields = df[mask].iloc[:, 4].values
                self.yields[:len(elem_yields), i] = elem_yields
    
    def get_yield(self, mass, element):
        """Get yield for specific mass and element."""
        return self.interpolate_yield(np.array([mass]), element)[0]


class CombinedYieldModel:
    """
    Combined yield model for SNII + AGB + SNIa nucleosynthesis.
    """
    
    def __init__(self, snii_model=None, agb_model=None, snia_model=None,
                 mass_separatrix=8.0):
        """
        Initialize combined yield model.
        
        Parameters
        ----------
        snii_model : YieldModel, optional
            SNII yield model
        agb_model : YieldModel, optional
            AGB yield model
        snia_model : YieldModel, optional
            SNIa yield model
        mass_separatrix : float
            Mass boundary between intermediate and massive stars (Msun)
        """
        self.snii_model = snii_model
        self.agb_model = agb_model
        self.snia_model = snia_model
        self.mass_separatrix = mass_separatrix
    
    def get_total_yield(self, mass, element, snii=True, agb=True, snia=True):
        """
        Get total yield from all sources.
        
        Parameters
        ----------
        mass : float
            Stellar mass in solar masses
        element : str
            Element symbol
        snii : bool
            Include SNII yields
        agb : bool
            Include AGB yields
        snia : bool
            Include SNIa yields
            
        Returns
        -------
        float
            Total yield in solar masses
        """
        total = 0.0
        
        if snii and self.snii_model:
            total += self.snii_model.get_yield(mass, element)
        
        if agb and self.agb_model and mass < self.mass_separatrix:
            total += self.agb_model.get_yield(mass, element)
        
        if snia and self.snia_model:
            total += self.snia_model.get_yield(mass, element)
        
        return total
    
    def get_mass_return(self, mass):
        """
        Get total mass returned for a star of given mass.
        
        Parameters
        ----------
        mass : float
            Stellar mass in solar masses
            
        Returns
        -------
        float
            Mass returned in solar masses
        """
        # Default: 25-30% of initial mass returned
        if mass < 1.0:
            return 0.0
        elif mass < self.mass_separatrix:
            # AGB stars return mass over their lifetime
            return max(0.0, mass - 0.5)
        else:
            # Massive stars: neutron star/BH remnant is ~1-3 Msun
            # Typically 10-50% mass returned as ejecta
            return max(0.0, mass * 0.3)


def create_yield_model(model_name, metallicity=0.02, **kwargs):
    """
    Factory function to create yield model objects.
    
    Parameters
    ----------
    model_name : str
        Model name: 'kobayashi', 'lc18', 'karakas'
    metallicity : float
        Metallicity
    **kwargs : dict
        Additional arguments for specific models
        
    Returns
    -------
    YieldModel
        Initialized yield model
    """
    if model_name.lower() == 'kobayashi':
        return KobayashiYields(metallicity, **kwargs)
    elif model_name.lower() == 'lc18':
        # GalaxySAM passes linear metallicity; LC18Yields expects log10(Z/Zsun)
        z_sun = constants.ZSUN_ASPLUND
        metallicity_log = np.log10(max(float(metallicity), 1e-12) / z_sun)
        return LC18Yields(metallicity_log=metallicity_log, **kwargs)
    elif model_name.lower() == 'karakas':
        return KarakasYields(metallicity, **kwargs)
    else:
        raise ValueError(f"Unknown yield model: {model_name}")
