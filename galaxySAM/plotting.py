"""
Plotting utilities for galaxy SAM yields and evolution.

Provides visualization of stellar yields, chemical evolution,
and galaxy properties over cosmic time.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from . import constants


class YieldPlotter:
    """Plots for stellar yield data."""
    
    def __init__(self, figsize=(12, 8), dpi=100):
        """
        Initialize yield plotter.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_yields_vs_mass(self, masses, yields_dict, elements=None,
                           title='Stellar Yields', output_file=None):
        """
        Plot yields as function of stellar mass.
        
        Parameters
        ----------
        masses : array
            Stellar masses in solar masses
        yields_dict : dict
            Dictionary mapping element names to yield arrays
        elements : list, optional
            Elements to plot (defaults to all)
        title : str
            Plot title
        output_file : Path or str, optional
            File to save plot
            
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if elements is None:
            elements = list(yields_dict.keys())
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Colors for elements
        colors = plt.cm.tab10(np.linspace(0, 1, len(elements)))
        
        for i, element in enumerate(elements):
            if element in yields_dict:
                yields = yields_dict[element]
                ax.loglog(masses, yields, 'o-', color=colors[i], 
                         label=element, linewidth=2, markersize=6)
        
        ax.set_xlabel(r'M$_{\mathrm{ZAMS}}$ (M$_{\odot}$)', fontsize=12)
        ax.set_ylabel(r'Yield (M$_{\odot}$)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Yield plot saved to {output_file}")
        
        return fig
    
    def plot_mass_loss_vs_mass(self, masses, mass_loss_dict, metallicity=None,
                              output_file=None):
        """
        Plot mass loss (wind) as function of stellar mass.
        
        Parameters
        ----------
        masses : array
            Stellar masses in solar masses
        mass_loss_dict : dict
            Dictionary mapping element names to mass loss arrays
        metallicity : float, optional
            Metallicity for plot label
        output_file : Path or str, optional
            File to save plot
            
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        elements = list(mass_loss_dict.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(elements)))
        
        for i, element in enumerate(elements):
            if element in mass_loss_dict:
                mloss = mass_loss_dict[element]
                # Only plot positive values
                positive = mloss > 1e-30
                ax.loglog(masses[positive], mloss[positive], 'o-', 
                         color=colors[i], label=element, linewidth=2, markersize=6)
        
        ax.set_xlabel(r'M$_{\mathrm{ZAMS}}$ (M$_{\odot}$)', fontsize=12)
        ax.set_ylabel(r'Mass Loss (M$_{\odot}$)', fontsize=12)
        
        if metallicity is not None:
            z_sun = constants.ZSUN_ASPLUND
            logz = np.log10(metallicity / z_sun)
            title = f'Mass Loss (Z = {metallicity:.4f}, log(Z/Z$_\\odot$) = {logz:.2f})'
        else:
            title = 'Mass Loss vs Stellar Mass'
        
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Mass loss plot saved to {output_file}")
        
        return fig
    
    def plot_yields_comparison(self, masses_list, yields_list, model_names,
                              element='Fe', output_file=None):
        """
        Compare yields from different models.
        
        Parameters
        ----------
        masses_list : list
            List of mass arrays for each model
        yields_list : list
            List of yield arrays for each model
        model_names : list
            Names of models
        element : str
            Element to plot
        output_file : Path or str, optional
            File to save plot
            
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        
        for i, (masses, yields, name) in enumerate(zip(masses_list, yields_list, 
                                                        model_names)):
            ax.loglog(masses, yields, 'o-', color=colors[i], 
                     label=name, linewidth=2, markersize=6)
        
        ax.set_xlabel(r'M$_{\mathrm{ZAMS}}$ (M$_{\odot}$)', fontsize=12)
        ax.set_ylabel(f'Yield ({element}, M$_{{\\odot}}$)', fontsize=12)
        ax.set_title(f'{element} Yields: Model Comparison', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Comparison plot saved to {output_file}")
        
        return fig


class EvolutionPlotter:
    """Plots for galaxy evolution results."""
    
    def __init__(self, figsize=(14, 10), dpi=100):
        """
        Initialize evolution plotter.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.obs_data_dir = (
            Path(__file__).parent / 'yield_files' / 'yield_files' / 'ObservationnalData'
        )

    def _load_observational_points(self):
        """Load observational [X/Fe] vs [Fe/H] points used in the IDL workflow."""
        file_map = {
            'C': 'saga_feh_cfe_all.txt',
            'N': 'saga_feh_nfe_all.txt',
            'O': 'cgisess_a5feab70ff4f90f6ea35e5e7922c6fab_data.tsv',
            'Mg': 'saga_feh_mgfe_all.txt',
            'Si': 'saga_feh_sife_all.txt',
            'S': 'saga_feh_sfe_all.txt',
        }

        obs = {}
        for element, filename in file_map.items():
            path = self.obs_data_dir / filename
            if not path.exists():
                continue

            try:
                data = pd.read_csv(path, sep='\t', engine='python')
            except Exception:
                continue

            if data.empty:
                continue

            feh_col = None
            xfe_col = None
            for col in data.columns:
                c = str(col).strip()
                if '[Fe/H]' in c:
                    feh_col = col
                if f'[{element}/Fe]' in c:
                    xfe_col = col

            if feh_col is None or xfe_col is None:
                numeric_cols = []
                for col in data.columns:
                    series = pd.to_numeric(data[col], errors='coerce')
                    if series.notna().sum() > 0:
                        numeric_cols.append(col)
                if len(numeric_cols) >= 2:
                    feh_col = numeric_cols[-2]
                    xfe_col = numeric_cols[-1]

            if feh_col is None or xfe_col is None:
                continue

            x = pd.to_numeric(data[feh_col], errors='coerce').to_numpy()
            y = pd.to_numeric(data[xfe_col], errors='coerce').to_numpy()
            good = np.isfinite(x) & np.isfinite(y)
            if np.any(good):
                obs[element] = (x[good], y[good])

        return obs
    
    def plot_evolution(self, evolution_dict, output_file=None):
        """
        Plot main galaxy evolution quantities.
        
        Parameters
        ----------
        evolution_dict : dict
            Dictionary with evolution data from GalaxySAM.evolve()
        output_file : Path or str, optional
            File to save plot
            
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('Galaxy Evolution', fontsize=16)
        
        time = evolution_dict['time']
        
        # Gas mass
        ax = axes[0, 0]
        ax.semilogy(time, evolution_dict['mgas'], 'b-', linewidth=2)
        ax.set_xlabel('Time (Gyr)', fontsize=11)
        ax.set_ylabel(r'Gas Mass (M$_{\odot}$)', fontsize=11)
        ax.set_title('Gas Mass Evolution')
        ax.grid(True, alpha=0.3)
        
        # Stellar mass
        ax = axes[0, 1]
        ax.semilogy(time, evolution_dict['mstar'], 'r-', linewidth=2)
        ax.set_xlabel('Time (Gyr)', fontsize=11)
        ax.set_ylabel(r'Stellar Mass (M$_{\odot}$)', fontsize=11)
        ax.set_title('Stellar Mass Evolution')
        ax.grid(True, alpha=0.3)
        
        # Metallicity
        ax = axes[1, 0]
        z = evolution_dict['metallicity']
        z_sun = constants.ZSUN_ASPLUND
        logz = np.log10(np.clip(z, 1e-5, 1.0) / z_sun)
        ax.plot(time, logz, 'g-', linewidth=2)
        ax.set_xlabel('Time (Gyr)', fontsize=11)
        ax.set_ylabel(r'log(Z/Z$_{\odot}$)', fontsize=11)
        ax.set_title('Metallicity Evolution')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Star formation rate
        ax = axes[1, 1]
        sfr = evolution_dict['sfr']
        ax.semilogy(time, np.clip(sfr, 1e-10, np.max(sfr)), 'orange', linewidth=2)
        ax.set_xlabel('Time (Gyr)', fontsize=11)
        ax.set_ylabel(r'SFR (M$_{\odot}$/yr)', fontsize=11)
        ax.set_title('Star Formation Rate')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Evolution plot saved to {output_file}")
        
        return fig
    
    def plot_abundance_ratios(self, evolution_dict, output_file=None):
        """
        Plot element abundance ratios vs metallicity.
        
        Parameters
        ----------
        evolution_dict : dict
            Dictionary with evolution data
        output_file : Path or str, optional
            File to save plot
            
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        time = evolution_dict['time']
        
        ax.plot(time, time / constants.HUBBLE_TIME, 'b-', linewidth=2, label='Age/Hubble time')
        ax.set_xlabel('Time (Gyr)', fontsize=12)
        ax.set_ylabel('Evolution Parameter', fontsize=12)
        ax.set_title('Normalized Evolution Timescale')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Abundance ratio plot saved to {output_file}")
        
        return fig

    def plot_abundances_vs_iron(self, evolution_dict, output_file=None, include_observations=True):
        """
        Plot IDL-like abundance panels [X/Fe] versus [Fe/H].

        Expected keys in evolution_dict:
        - mchemstar: array of shape (ntime, nchem)
        - mchemgas: array of shape (ntime, nchem) (optional)
        - mzstar: array of shape (ntime,) (optional for [Z/Fe])
        - mzgas: array of shape (ntime,) (optional for [Z/Fe] gas curve)
        - elements: list of element labels matching chemistry columns (optional)

        Parameters
        ----------
        include_observations : bool
            If True, overlay observational points from ObservationnalData.

        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if 'mchemstar' not in evolution_dict:
            raise ValueError(
                "plot_abundances_vs_iron requires 'mchemstar' in evolution_dict. "
                "Provide element mass histories from your ODE solver."
            )

        mchemstar = np.asarray(evolution_dict['mchemstar'], dtype=float)
        mchemgas = np.asarray(evolution_dict['mchemgas'], dtype=float) if 'mchemgas' in evolution_dict else None
        mzstar = np.asarray(evolution_dict['mzstar'], dtype=float) if 'mzstar' in evolution_dict else None
        mzgas = np.asarray(evolution_dict['mzgas'], dtype=float) if 'mzgas' in evolution_dict else None

        if mchemstar.ndim != 2:
            raise ValueError("'mchemstar' must be a 2D array with shape (ntime, nchem).")
        if mchemgas is not None and mchemgas.shape != mchemstar.shape:
            raise ValueError("'mchemgas' must have the same shape as 'mchemstar'.")

        elements = evolution_dict.get('elements', constants.ELEMENTS_LC18)
        elements_norm = ['H' if el == 'p' else el for el in elements]
        index = {el: i for i, el in enumerate(elements_norm)}

        required = ['H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'Mg', 'Si', 'S', 'Fe']
        missing = [el for el in required if el not in index]
        if missing:
            raise ValueError(
                f"Missing required elements in chemistry history: {missing}. "
                "Provide an 'elements' list matching chemistry columns."
            )

        panel_elements = ['Z', 'H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'Mg', 'Si', 'S', 'Fe']
        ytitles = ['[Z/Fe]', '[H/Fe]', '[He/Fe]', '[C/Fe]', '[N/Fe]', '[O/Fe]',
                   '[F/Fe]', '[Ne/Fe]', '[Mg/Fe]', '[Si/Fe]', '[S/Fe]', '[Fe/Fe]']
        obs_points = self._load_observational_points() if include_observations else {}

        fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=self.dpi, sharex=True, sharey=True)
        fig.suptitle('Abundances vs Iron: [X/Fe] vs [Fe/H]', fontsize=16)

        eps = 1e-30

        def bracket_x_over_fe(mchem, element):
            fe = np.clip(mchem[:, index['Fe']], eps, None)
            if element == 'Z':
                if mchem is mchemstar and mzstar is not None:
                    x = np.clip(mzstar, eps, None)
                elif mchem is mchemgas and mzgas is not None:
                    x = np.clip(mzgas, eps, None)
                else:
                    heavy = ['C', 'N', 'O', 'F', 'Ne', 'Mg', 'Si', 'S', 'Fe']
                    heavy_idx = [index[h] for h in heavy]
                    x = np.clip(np.sum(mchem[:, heavy_idx], axis=1), eps, None)
                solar_ratio = constants.ZSUN_ASPLUND / constants.ASPLUND_ABUNDANCES['Fe']
            else:
                x = np.clip(mchem[:, index[element]], eps, None)
                solar_ratio = constants.ASPLUND_ABUNDANCES[element] / constants.ASPLUND_ABUNDANCES['Fe']
            return np.log10((x / fe) / solar_ratio)

        def bracket_fe_over_h(mchem):
            fe = np.clip(mchem[:, index['Fe']], eps, None)
            h = np.clip(mchem[:, index['H']], eps, None)
            solar_fe_h = constants.ASPLUND_ABUNDANCES['Fe'] / constants.ASPLUND_ABUNDANCES['H']
            return np.log10((fe / h) / solar_fe_h)

        x_star = bracket_fe_over_h(mchemstar)
        x_gas = bracket_fe_over_h(mchemgas) if mchemgas is not None else None

        for ax, el, ytitle in zip(axes.flat, panel_elements, ytitles):
            y_star = bracket_x_over_fe(mchemstar, el)
            ax.plot(x_star, y_star, color='tab:red', linewidth=2.0, label='star')

            if mchemgas is not None:
                y_gas = bracket_x_over_fe(mchemgas, el)
                ax.plot(x_gas, y_gas, color='tab:blue', linewidth=1.8, label='gas')

            if el in obs_points:
                x_obs, y_obs = obs_points[el]
                ax.scatter(
                    x_obs,
                    y_obs,
                    s=18,
                    c='black',
                    alpha=0.35,
                    marker='o',
                    linewidths=0,
                    label='obs' if el == 'C' else None,
                )

            ax.set_title(ytitle, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-3.0, 1.0)
            ax.set_ylim(-1.3, 1.3)

        for ax in axes[-1, :]:
            ax.set_xlabel('[Fe/H]', fontsize=11)
        for ax in axes[:, 0]:
            ax.set_ylabel('[X/Fe]', fontsize=11)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Abundances-vs-iron plot saved to {output_file}")

        return fig
    
    def plot_gas_metal_phase(self, evolution_dict, output_file=None):
        """
        Plot gas mass vs metallicity phase diagram.
        
        Parameters
        ----------
        evolution_dict : dict
            Dictionary with evolution data
        output_file : Path or str, optional
            File to save plot
            
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        mgas = evolution_dict['mgas']
        z = evolution_dict['metallicity']
        z_sun = constants.ZSUN_ASPLUND
        logz = np.log10(np.clip(z, 1e-5, 1.0) / z_sun)
        
        # Color by time
        time = evolution_dict['time']
        scatter = ax.scatter(mgas, logz, c=time, cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel(r'Gas Mass (M$_{\odot}$)', fontsize=12)
        ax.set_ylabel(r'log(Z/Z$_{\odot}$)', fontsize=12)
        ax.set_title('Gas-Metallicity Phase Space')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (Gyr)', fontsize=11)
        
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Phase space plot saved to {output_file}")
        
        return fig


class YieldComparisonPlotter:
    """Specialized plots for comparing yield models."""
    
    def __init__(self, figsize=(16, 12), dpi=100):
        """
        Initialize yield comparison plotter.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        dpi : int
            Resolution
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_all_yields_multi_model(self, models_dict, elements=None,
                                   output_file=None):
        """
        Create multi-panel plot comparing yields across models and elements.
        
        Parameters
        ----------
        models_dict : dict
            Dict mapping model names to dicts with 'masses' and 'yields'
        elements : list, optional
            Elements to show
        output_file : Path or str, optional
            File to save plot
            
        Returns
        -------
        Figure
            Matplotlib figure object
        """
        if elements is None:
            elements = ['O', 'Fe', 'Mg', 'Si']
        
        n_elem = len(elements)
        n_model = len(models_dict)
        
        fig, axes = plt.subplots(n_elem, n_model, figsize=self.figsize, dpi=self.dpi)
        
        if n_model == 1:
            axes = axes.reshape(-1, 1)
        if n_elem == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_model))
        
        for i, element in enumerate(elements):
            for j, (model_name, model_data) in enumerate(models_dict.items()):
                ax = axes[i, j]
                
                masses = model_data['masses']
                yields = model_data['yields']
                
                if element in yields:
                    y = yields[element]
                    ax.loglog(masses, y, 'o-', color=colors[j], linewidth=2)
                
                if j == 0:
                    ax.set_ylabel(f'{element} Yield (M$_{{\\odot}}$)', fontsize=10)
                else:
                    ax.set_ylabel('')
                
                if i == n_elem - 1:
                    ax.set_xlabel(r'M$_{\mathrm{ZAMS}}$ (M$_{\odot}$)', fontsize=10)
                
                ax.set_title(f'{element} - {model_name}' if i == 0 else model_name,
                           fontsize=11)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            print(f"Multi-model comparison saved to {output_file}")
        
        return fig


def create_all_plots(evolution_dict, yield_dict=None, output_dir=None):
    """
    Create a complete set of plots from evolution results.
    
    Parameters
    ----------
    evolution_dict : dict
        Galaxy evolution dictionary
    yield_dict : dict, optional
        Yield data dictionary
    output_dir : Path or str, optional
        Directory to save plots
        
    Returns
    -------
    dict
        Dictionary with all created figures
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Evolution plots
    ev_plotter = EvolutionPlotter()
    figures['evolution'] = ev_plotter.plot_evolution(
        evolution_dict,
        output_file=output_dir / 'evolution.png' if output_dir else None
    )
    
    figures['gas_metal_phase'] = ev_plotter.plot_gas_metal_phase(
        evolution_dict,
        output_file=output_dir / 'gas_metal_phase.png' if output_dir else None
    )

    if 'mchemstar' in evolution_dict:
        figures['abundances_vs_iron'] = ev_plotter.plot_abundances_vs_iron(
            evolution_dict,
            output_file=output_dir / 'abundances_vs_iron.png' if output_dir else None
        )
    
    # Yield plots if available
    if yield_dict:
        yield_plotter = YieldPlotter()
        figures['yields'] = yield_plotter.plot_yields_vs_mass(
            yield_dict.get('masses', []),
            yield_dict.get('yields', {}),
            output_file=output_dir / 'yields.png' if output_dir else None
        )
    
    return figures
