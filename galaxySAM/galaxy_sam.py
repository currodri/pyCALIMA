"""
Galaxy Semi-Analytic Model (SAM) for galactic chemical evolution.

Implements the main evolution equations for star formation, gas accretion,
chemical enrichment, and feedback processes.
"""

import numpy as np
from scipy.integrate import odeint
from pathlib import Path
import pandas as pd
from . import constants
from . import yield_models
from . import imf
from . import sn1a as snia


class GalaxySAM:
    """
    Semi-Analytic Model for galaxy evolution.
    
    Integrates the differential equations for:
    - Gas accretion onto the galaxy
    - Star formation
    - Chemical enrichment from SNII, AGB, and SNIa
    - Galactic winds/outflows
    """
    
    def __init__(self, yield_model='kobayashi', metallicity=0.02,
                 imf_type='chabrier', **params):
        """
        Initialize galaxy SAM.
        
        Parameters
        ----------
        yield_model : str or YieldModel
            Yield model name or object
        metallicity : float
            Initial metallicity
        imf_type : str
            IMF type ('salpeter', 'chabrier', etc.)
        **params : dict
            Additional parameters (see constants.DEFAULT_PARAMS)
        """
        # Setup parameters with defaults
        self.params = constants.DEFAULT_PARAMS.copy()
        self.params.update(params)
        
        # Physical setup
        self.metallicity_init = metallicity
        self.z_sun = constants.ZSUN_ASPLUND
        
        # Initialize yield model
        if isinstance(yield_model, str):
            self.yield_model = yield_models.create_yield_model(
                yield_model, metallicity=metallicity
            )
        else:
            self.yield_model = yield_model
        
        # Initialize IMF
        self.imf = imf.create_imf(imf_type)
        
        # Initialize Type Ia model
        self.snia_model = snia.SNIaModel(asnia=self.params['asnia'])
        
        # Time stepping
        self.hubble_time = constants.HUBBLE_TIME * 1e9  # Convert to years
        self.nbint = self.params['nbint']
        self.dt = self.hubble_time / self.nbint
        self.time_grid = np.linspace(0, self.hubble_time, self.nbint)
        
        # Evolution arrays (initialized in run())
        self.mgas = None
        self.mstar = None
        self.metals = None
        self.sfr = None
        self.elements = list(getattr(self.yield_model, 'elements', constants.ELEMENTS_LC18))
        self._elements_norm = ['H' if e == 'p' else e for e in self.elements]
        self._element_index = {e: i for i, e in enumerate(self._elements_norm)}
        self._metal_elements = [e for e in self._elements_norm if e not in ('H', 'He')]
        self._metal_indices = [self._element_index[e] for e in self._metal_elements if e in self._element_index]
        self._infall_fractions = self._build_infall_fractions()
        self._yield_per_sfr = self._build_yield_per_sfr()

        self.mchemgas = None
        self.mchemstar = None
        self.mzgas = None
        self.mzstar = None
        self.msnia_reservoir = None

    def _build_infall_fractions(self):
        """Build infall element fractions from initial metallicity and solar pattern."""
        nchem = len(self._elements_norm)
        ff = np.zeros(nchem, dtype=float)

        z_init = float(np.clip(self.metallicity_init, 0.0, 0.5))
        x_h = (1.0 - z_init) * constants.ASPLUND_ABUNDANCES.get('H', 0.7381)
        x_he = (1.0 - z_init) * constants.ASPLUND_ABUNDANCES.get('He', 0.2485)

        if 'H' in self._element_index:
            ff[self._element_index['H']] = x_h
        if 'He' in self._element_index:
            ff[self._element_index['He']] = x_he

        metals_norm = 0.0
        for el in self._metal_elements:
            metals_norm += constants.ASPLUND_ABUNDANCES.get(el, 0.0)
        metals_norm = max(metals_norm, 1e-30)

        for el in self._metal_elements:
            i = self._element_index[el]
            ff[i] = z_init * constants.ASPLUND_ABUNDANCES.get(el, 0.0) / metals_norm

        total = np.sum(ff)
        if total > 0:
            ff /= total
        return ff

    def _build_yield_per_sfr(self):
        """Precompute IMF-integrated element return rates per unit stellar mass formed."""
        nchem = len(self._elements_norm)
        yp = np.zeros(nchem, dtype=float)

        masses = np.asarray(getattr(self.yield_model, 'masses', []), dtype=float)
        if masses.size < 2:
            # Fallback to simple total-metal yield distributed on solar pattern
            ytot = float(self.params.get('yield', 0.1))
            if self._metal_indices:
                metal_weights = np.array([
                    constants.ASPLUND_ABUNDANCES.get(self._elements_norm[i], 0.0)
                    for i in self._metal_indices
                ], dtype=float)
                metal_weights_sum = np.sum(metal_weights)
                if metal_weights_sum <= 0:
                    metal_weights = np.ones_like(metal_weights)
                    metal_weights_sum = np.sum(metal_weights)
                metal_weights /= metal_weights_sum
                for w, i in zip(metal_weights, self._metal_indices):
                    yp[i] = ytot * w
            return yp

        masses = np.sort(masses)
        phi = self.imf(masses)
        denom = np.trapz(phi * masses, masses)
        denom = max(float(denom), 1e-30)

        for i, el in enumerate(self._elements_norm):
            try:
                y_m = np.asarray(self.yield_model.interpolate_yield(masses, el), dtype=float)
            except Exception:
                y_m = np.zeros_like(masses)
            # Negative net yields are allowed in principle, but clip large negative noise.
            y_m = np.clip(y_m, -1e2, 1e2)
            yp[i] = np.trapz(phi * y_m, masses) / denom

        return yp
    
    def infall_rate_exponential(self, t):
        """
        Exponential infall rate: dM_gas/dt ∝ exp(-t/tau).
        
        Parameters
        ----------
        t : float or array
            Time in years
            
        Returns
        -------
        float or array
            Infall rate in Msun/Gyr
        """
        tau = self.params['tscale_infall'] * 1e9  # Convert to years
        prefactor = self.params['prefactor']
        
        rate = prefactor * np.exp(-t / tau)
        return rate / 1e9  # Convert to Msun/year
    
    def infall_rate_double_exponential(self, t):
        """
        Double exponential infall: (t/tau2)*exp(-t/tau1).
        
        Parameters
        ----------
        t : float or array
            Time in years
            
        Returns
        -------
        float or array
            Infall rate in Msun/year
        """
        tscales = self.params['tscale_infall']
        if not isinstance(tscales, (list, tuple)) or len(tscales) != 2:
            return self.infall_rate_exponential(t)
        
        tau1 = tscales[0] * 1e9
        tau2 = tscales[1] * 1e9
        prefactor = self.params['prefactor']
        
        rate = prefactor * (t / tau2) * np.exp(-t / tau1)
        return rate / 1e9
    
    def infall_rate_no_accretion(self, t):
        """
        No accretion model: initial gas mass only.
        
        Parameters
        ----------
        t : float or array
            Time in years
            
        Returns
        -------
        float or array
            Infall rate (always 0)
        """
        return np.zeros_like(t)
    
    def star_formation_rate(self, mgas, mstar=None):
        """
        Star formation rate using Schmidt-Kennicutt relation.
        
        SFR ∝ Mgas^alphaks / tau_sfr
        
        Optional starburst enhancement: SFR *= min(Mgas/Mstar, 10) if Mgas > Mstar
        
        Parameters
        ----------
        mgas : float
            Gas mass in Msun
        mstar : float, optional
            Stellar mass (for starburst calculation)
            
        Returns
        -------
        float
            Star formation rate in Msun/year
        """
        tau_sfr = self.params['tscale_sfr'] * 1e9  # Convert to years
        alpha = self.params['alphaks']
        
        sfr = (mgas ** alpha) / tau_sfr
        
        # Apply starburst enhancement if Mgas > Mstar
        if self.params.get('starburst', False) and mstar is not None:
            if mgas > mstar:
                boost = min(mgas / mstar, 10.0)
                sfr *= boost
        
        return sfr
    
    def outflow_rate(self, sfr, mstar=None):
        """
        Galactic wind/outflow rate.
        
        Multiple models available via wind_model parameter.
        
        Parameters
        ----------
        sfr : float
            Star formation rate in Msun/year
        mstar : float, optional
            Stellar mass (for wind scaling)
            
        Returns
        -------
        float
            Outflow rate in Msun/year
        """
        if self.params['wind_loading'] > 0:
            # Fixed wind loading
            return self.params['wind_loading'] * sfr
        
        if self.params.get('windmodel') is None:
            return 0.0
        
        windmodel = self.params['windmodel']
        
        if windmodel == 1 or windmodel > 2:
            # Variable wind loading scaling with halo mass
            if mstar is None:
                mstar = 1e10  # Default halo mass
            mstar_eff = max(float(mstar), 1e-30)
            
            # Scaling: min_mload < sqrt(2.5e10/Mstar) < max_mload
            sqrt_term = np.sqrt(2.5e10 / mstar_eff)
            min_load = self.params['minmload']
            max_load = self.params['maxmload']
            
            loading = np.clip(sqrt_term, min_load, max_load)
            return loading * sfr
        
        elif windmodel == 2:
            # Hayward & Hopkins 2017 wind model
            # (Simplified version - full model more complex)
            if mstar is None:
                mstar = 1e10
            
            # Empirical relation
            loading = 0.1 * (mstar / 1e10)**(-0.25)
            return loading * sfr
        
        return 0.0
    
    def mass_return_snii(self, sfr):
        """
        Mass and metals returned from SNII.
        
        Parameters
        ----------
        sfr : float
            Star formation rate in Msun/year
            
        Returns
        -------
        tuple
            (mass_returned, metals_returned) in Msun/year
        """
        # Average mass and metal return fractions
        mass_fraction_return = 0.3  # ~30% of stellar mass returned
        
        mass_returned = sfr * mass_fraction_return
        
        # Metal yield (yield in solar masses per solar mass of star formed)
        yield_per_star = self.params.get('yield', 0.1)
        metals_returned = sfr * yield_per_star
        
        return mass_returned, metals_returned
    
    def mass_return_agb(self, sfr_past, tau_agb=1e9):
        """
        Mass and metals returned from AGB stars.
        
        Parameters
        ----------
        sfr_past : array
            Historical star formation rates
        tau_agb : float
            AGB lifetime in years
            
        Returns
        -------
        tuple
            (mass_returned, metals_returned) in Msun/year
        """
        # AGB return is delayed - approximately lifetime tau_agb
        # Approximate with simple exponential delay
        
        mass_returned = 0.0
        metals_returned = 0.0
        
        # This would require tracking SFR history
        # Simplified version: neglect AGB contribution
        
        return mass_returned, metals_returned
    
    def mass_return_snia(self, sfr, delay_time=1e9):
        """
        Mass and metals returned from Type Ia SNe.
        
        Parameters
        ----------
        sfr : float
            Star formation rate in Msun/year
        delay_time : float
            Delay time for SNIa in years
            
        Returns
        -------
        tuple
            (mass_returned, metals_returned) in Msun/year
        """
        if self.params.get('nosnia', False):
            return 0.0, 0.0
        
        # SNIa rate scales with stellar mass
        asnia = self.params['asnia']
        snia_rate = sfr * asnia  # This is simplified
        
        # Each SNIa returns ~1.4 Msun (mostly Fe-peak)
        snia_ejecta = 1.4  # Msun
        mass_returned = snia_rate * snia_ejecta
        
        # Iron yield per SNIa (~0.6-0.8 Msun of Fe56)
        iron_yield = 0.7
        metals_returned = snia_rate * iron_yield
        
        return mass_returned, metals_returned
    
    def evolve(self, output_file=None):
        """
        Evolve the galaxy using integrated differential equations.
        
        Parameters
        ----------
        output_file : Path or str, optional
            File to save evolution data
            
        Returns
        -------
        dict
            Dictionary with evolution arrays
        """
        # Initial conditions
        prefactor = self.params['prefactor']
        accmodel = self.params['accmodel']
        
        if accmodel == 3:
            # No accretion: prefactor is initial gas mass
            mgas_init = prefactor
        else:
            # With accretion: use fixed initial gas mass
            mgas_init = 1e10  # Msun (arbitrary, will scale)
        
        mstar_init = 0.0
        metals_init = mgas_init * self.metallicity_init

        nchem = len(self._elements_norm)
        mchemgas_init = mgas_init * self._infall_fractions
        mchemstar_init = np.zeros(nchem, dtype=float)

        y_init = np.concatenate((
            np.array([mgas_init, mstar_init, metals_init], dtype=float),
            mchemgas_init,
            mchemstar_init,
            np.array([0.0], dtype=float),
        ))
        
        # Integrate differential equations
        solution = odeint(self._dydt, y_init, self.time_grid / 1e9,  # Convert to Gyr
                         args=(accmodel,), full_output=False)
        
        # Extract results
        self.mgas = solution[:, 0]
        self.mstar = solution[:, 1]
        self.metals = solution[:, 2]
        self.mchemgas = solution[:, 3:3 + nchem]
        self.mchemstar = solution[:, 3 + nchem:3 + 2 * nchem]
        self.msnia_reservoir = solution[:, 3 + 2 * nchem]
        
        # Compute derived quantities
        self.metallicity = self.metals / (self.mgas + 1e-10)
        if self._metal_indices:
            self.mzgas = np.sum(self.mchemgas[:, self._metal_indices], axis=1)
            self.mzstar = np.sum(self.mchemstar[:, self._metal_indices], axis=1)
        else:
            self.mzgas = np.zeros_like(self.mgas)
            self.mzstar = np.zeros_like(self.mstar)
        self.sfr = np.array([self.star_formation_rate(mgas=mgas) 
                            for mgas in self.mgas])
        
        # Save results if requested
        if output_file:
            self._save_evolution(output_file)
        
        return self._get_evolution_dict()
    
    def _dydt(self, y, t, accmodel):
        """
        Differential equations for galaxy evolution.
        
        Parameters
        ----------
        y : array
            State vector [Mgas, Mstar, Metals]
        t : float
            Time in Gyr
        accmodel : int
            Accretion model choice
            
        Returns
        -------
        array
            Derivatives [dMgas/dt, dMstar/dt, dMetals/dt]
        """
        nchem = len(self._elements_norm)
        mgas = y[0]
        mstar = y[1]
        metals = y[2]
        mchemgas = y[3:3 + nchem]
        mchemstar = y[3 + nchem:3 + 2 * nchem]
        msnia_res = y[3 + 2 * nchem]
        
        # Convert time back to years
        t_years = t * 1e9
        
        # Star formation
        sfr = self.star_formation_rate(mgas, mstar)
        
        # Accretion
        if accmodel == 1:
            accr = self.infall_rate_exponential(t_years)
        elif accmodel == 2:
            accr = self.infall_rate_double_exponential(t_years)
        elif accmodel == 3:
            accr = 0.0
        else:
            accr = self.infall_rate_exponential(t_years)
        
        # Outflow
        outf = self.outflow_rate(sfr, mstar)
        
        # Element-by-element enrichment/locking terms.
        zcgas = mchemgas / (mgas + 1e-30)
        ejecta_elem = np.clip(self._yield_per_sfr, -1e2, 1e2) * sfr

        # Delayed SNIa channel: reservoir fed by star formation and drained on tau_Ia.
        tau_ia = float(self.params.get('snia_delay_gyr', 1.5)) * 1e9
        tau_ia = max(tau_ia, 1e6)
        asnia = float(self.params.get('asnia', 0.05))
        dmsnia_res_dt = asnia * sfr - msnia_res / tau_ia
        if (not self.params.get('nosnia', False)) and ('Fe' in self._element_index):
            fe_ia_yield = float(self.params.get('snia_fe_yield', 0.7))
            ejecta_elem[self._element_index['Fe']] += (msnia_res / tau_ia) * fe_ia_yield

        dmchemgas_dt = (
            -sfr * zcgas
            -outf * zcgas
            +accr * self._infall_fractions
            +ejecta_elem
        )
        dmchemstar_dt = sfr * zcgas - ejecta_elem

        dmgas_dt = np.sum(dmchemgas_dt)
        dmstar_dt = np.sum(dmchemstar_dt)

        if self._metal_indices:
            dmetal_dt = np.sum(dmchemgas_dt[self._metal_indices])
        else:
            dmetal_dt = 0.0

        dydt = np.concatenate((
            np.array([dmgas_dt, dmstar_dt, dmetal_dt], dtype=float),
            dmchemgas_dt,
            dmchemstar_dt,
            np.array([dmsnia_res_dt], dtype=float),
        ))

        # odeint integrates against time in Gyr, so convert derivatives to per-Gyr units.
        return dydt * 1e9
    
    def _get_evolution_dict(self):
        """Return evolution results as dictionary."""
        result = {
            'time': self.time_grid / 1e9,  # Convert to Gyr
            'mgas': self.mgas,
            'mstar': self.mstar,
            'metals': self.metals,
            'metallicity': self.metallicity,
            'sfr': self.sfr,
            'elements': self._elements_norm,
            'mchemgas': self.mchemgas,
            'mchemstar': self.mchemstar,
            'mzgas': self.mzgas,
            'mzstar': self.mzstar,
            'msnia_reservoir': self.msnia_reservoir,
        }
        return result
    
    def _save_evolution(self, output_file):
        """Save evolution to file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = pd.DataFrame({
            'time_gyr': self.time_grid / 1e9,
            'mgas_msun': self.mgas,
            'mstar_msun': self.mstar,
            'metals_msun': self.metals,
            'metallicity': self.metallicity,
            'sfr_msun_per_yr': self.sfr,
        })
        
        data.to_csv(output_file, index=False, sep='\t')
        print(f"Evolution saved to {output_file}")


class MultiMetallicitySAM:
    """
    SAM for tracking multiple metallicity bins simultaneously.
    
    Tracks evolution of gas and metals in different metallicity bins
    to properly account for chemical evolution.
    """
    
    def __init__(self, nz_bins=10, yield_model='kobayashi', imf_type='chabrier',
                 **params):
        """
        Initialize multi-metallicity SAM.
        
        Parameters
        ----------
        nz_bins : int
            Number of metallicity bins
        yield_model : str
            Yield model name
        imf_type : str
            IMF type
        **params : dict
            Additional parameters
        """
        self.nz_bins = nz_bins
        self.z_bins = np.logspace(-4, np.log10(0.1), nz_bins)
        
        # Initialize SAM for each metallicity
        self.sams = []
        for z in self.z_bins:
            sam = GalaxySAM(yield_model=yield_model, metallicity=z,
                          imf_type=imf_type, **params)
            self.sams.append(sam)
        
        # Track mass in each bin
        self.mgas_bins = np.zeros(nz_bins)
        self.mstar_bins = np.zeros(nz_bins)
    
    def evolve(self, output_file=None):
        """
        Evolve multi-metallicity system.
        
        Parameters
        ----------
        output_file : Path or str, optional
            File to save results
            
        Returns
        -------
        dict
            Evolution results
        """
        # Evolve each metallicity bin
        results = []
        for i, sam in enumerate(self.sams):
            result = sam.evolve()
            results.append(result)
        
        return {
            'z_bins': self.z_bins,
            'results': results,
        }
