"""
Stellar yield models for SNII, AGB, and SNIa nucleosynthesis.

Handles reading, interpolation, and computation of yields from various
sources including Kobayashi et al., Limongi & Chieffi (LC18), and Karakas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, LSQUnivariateSpline
import pandas as pd
from pathlib import Path
from . import constants

# Default yield data directory
DEFAULT_YIELD_DIR = Path(__file__).parent / 'yield_files' / 'yield_files'
DEFAULT_KOBAYASHI_RAW_DAT = Path(__file__).parent / 'external_yields' / 'SN2SAGBYIELD.DAT'
DEFAULT_NOZAWA2003_DUST_YIELDS = Path(__file__).parent / 'external_yields' / 'Nozawa2003_dust_yields.dat'
DEFAULT_NOZAWA2003_DUST_DIST = Path(__file__).parent / 'external_yields' / 'Nozawa2003_dust_dist.dat'
KOBAYASHI_RAW_BLOCK_SIZE = 87


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


def _format_kobayashi_metallicity(z):
    """Format a metallicity value the same way Kobayashi files are named."""
    if np.isclose(float(z), 0.0):
        return '0'
    z_formatted = f"{float(z):.6g}".lstrip('0')
    if z_formatted.startswith('.'):
        z_formatted = '0' + z_formatted
    return z_formatted


def _load_kobayashi_isotope_labels(data_dir, metallicity):
    """Load the isotope ordering used to expand Kobayashi elemental yields."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_YIELD_DIR
    z_formatted = _format_kobayashi_metallicity(metallicity)
    isotope_file = data_dir / f'yield_ck13_z{z_formatted}.txt'
    if not isotope_file.exists():
        raise FileNotFoundError(f"Kobayashi isotope file not found: {isotope_file}")

    df = pd.read_csv(
        isotope_file,
        sep=r'\s+',
        comment='#',
        engine='python',
        header=None,
    )
    if df.shape[1] < 3:
        raise ValueError(f"Unexpected Kobayashi isotope file format: {isotope_file}")

    isotope_rows = df.iloc[: KOBAYASHI_RAW_BLOCK_SIZE - 4, :2].copy()
    isotope_rows.columns = ['element', 'mass_number']
    isotope_rows['mass_number'] = isotope_rows['mass_number'].astype(int)
    return isotope_rows


def _load_kobayashi_raw_sn_block(raw_dat_file, metallicity):
    """Read one SN metallicity block from the raw Kobayashi DAT table."""
    raw_dat_file = Path(raw_dat_file) if raw_dat_file else DEFAULT_KOBAYASHI_RAW_DAT
    if not raw_dat_file.exists():
        raise FileNotFoundError(f"Kobayashi raw DAT file not found: {raw_dat_file}")

    numeric_rows = []
    with raw_dat_file.open('r', encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            numeric_rows.append([float(value) for value in stripped.split()])

    nz = len(KobayashiYields.AVAILABLE_Z)
    expected_sn_rows = nz * KOBAYASHI_RAW_BLOCK_SIZE
    if len(numeric_rows) < expected_sn_rows:
        raise ValueError(
            f"Unexpected raw Kobayashi DAT length in {raw_dat_file}: "
            f"found {len(numeric_rows)} numeric rows, expected at least {expected_sn_rows}"
        )

    z_available = np.array(KobayashiYields.AVAILABLE_Z)
    z_actual = z_available[np.argmin(np.abs(z_available - float(metallicity)))]
    z_index = int(np.where(np.isclose(z_available, z_actual))[0][0])
    block_start = z_index * KOBAYASHI_RAW_BLOCK_SIZE
    block_rows = numeric_rows[block_start:block_start + KOBAYASHI_RAW_BLOCK_SIZE]

    masses = np.asarray(block_rows[1], dtype=float)
    explosion_energy = np.asarray(block_rows[2], dtype=float)
    remnant_mass = np.asarray(block_rows[3], dtype=float)
    isotope_yields = np.asarray(block_rows[4:], dtype=float)

    if isotope_yields.shape[0] != KOBAYASHI_RAW_BLOCK_SIZE - 4:
        raise ValueError(
            f"Unexpected isotope row count in {raw_dat_file} for Z={z_actual}: "
            f"found {isotope_yields.shape[0]} rows"
        )

    return {
        'metallicity': z_actual,
        'masses': masses,
        'explosion_energy': explosion_energy,
        'remnant_mass': remnant_mass,
        'isotope_yields': isotope_yields,
    }


def _aggregate_kobayashi_raw_elemental_yields(raw_block, isotope_labels, elements):
    """Aggregate raw Kobayashi isotope yields into elemental yields."""
    element_names = list(elements)
    element_index = {element: idx for idx, element in enumerate(element_names)}
    elemental_yields = np.zeros((len(raw_block['masses']), len(element_names)))

    if len(isotope_labels) != raw_block['isotope_yields'].shape[0]:
        raise ValueError(
            "Isotope labels and raw DAT rows do not match: "
            f"{len(isotope_labels)} labels vs {raw_block['isotope_yields'].shape[0]} rows"
        )

    for isotope_idx, isotope_row in isotope_labels.iterrows():
        raw_element = isotope_row['element']
        element_name = 'H' if raw_element == 'H' else raw_element
        if element_name not in element_index:
            continue
        elemental_yields[:, element_index[element_name]] += raw_block['isotope_yields'][isotope_idx]

    return elemental_yields


def compare_kobayashi_yields_to_raw_dat(
    metallicity=0.02,
    raw_dat_file=None,
    data_dir=None,
    elements=None,
):
    """
    Compare simplified Kobayashi SNII yields against the raw SN2SAGBYIELD.DAT table.

    The raw DAT stores one metallicity block as 87 rows: metallicity, mass grid,
    explosion energy, remnant mass, and 83 isotope-yield rows. This helper reads
    the SN block for the metallicity closest to ``metallicity``, aggregates those
    isotope rows into elemental yields using the repository's ``yield_ck13_z*.txt``
    isotope ordering, and compares them to the yields loaded by ``KobayashiYields``.

    Parameters
    ----------
    metallicity : float
        Linear metallicity, e.g. 0.02 for solar.
    raw_dat_file : Path or str, optional
        Path to the raw SN2SAGBYIELD.DAT file.
    data_dir : Path or str, optional
        Directory containing Kobayashi yield files.
    elements : sequence of str, optional
        Elements to compare. Defaults to the Kobayashi model element list.

    Returns
    -------
    pandas.DataFrame
        One row per ``(mass, element)`` comparison with model and raw values and
        absolute/relative differences. Only masses present in both datasets are
        included.
    """
    model = KobayashiYields(metallicity=metallicity, data_dir=data_dir)
    element_names = list(elements) if elements is not None else list(model.elements)
    invalid_elements = [element for element in element_names if element not in model.elements]
    if invalid_elements:
        raise ValueError(f"Unknown Kobayashi comparison elements: {invalid_elements}")

    raw_block = _load_kobayashi_raw_sn_block(raw_dat_file, model.metallicity_actual)
    isotope_labels = _load_kobayashi_isotope_labels(model.data_dir, model.metallicity_actual)
    raw_elemental_yields = _aggregate_kobayashi_raw_elemental_yields(
        raw_block,
        isotope_labels,
        model.elements,
    )

    common_masses = np.intersect1d(model.masses, raw_block['masses'])
    if common_masses.size == 0:
        raise ValueError(
            f"No overlapping masses found between Kobayashi model and raw DAT for Z={model.metallicity_actual}"
        )

    rows = []
    for mass in common_masses:
        model_idx = int(np.where(np.isclose(model.masses, mass))[0][0])
        raw_idx = int(np.where(np.isclose(raw_block['masses'], mass))[0][0])
        explosion_energy = float(raw_block['explosion_energy'][raw_idx])
        remnant_mass = float(raw_block['remnant_mass'][raw_idx])

        for element in element_names:
            element_idx = model.elements.index(element)
            model_yield = float(model.yields[model_idx, element_idx])
            raw_yield = float(raw_elemental_yields[raw_idx, element_idx])
            scale = max(abs(model_yield), abs(raw_yield), 1e-30)
            rows.append(
                {
                    'metallicity': model.metallicity_actual,
                    'mass': float(mass),
                    'element': element,
                    'raw_explosion_energy': explosion_energy,
                    'raw_remnant_mass': remnant_mass,
                    'model_yield': model_yield,
                    'raw_dat_yield': raw_yield,
                    'abs_diff': model_yield - raw_yield,
                    'rel_diff': (model_yield - raw_yield) / scale,
                }
            )

    return pd.DataFrame(rows)


def plot_kobayashi_yield_differences(
    metallicities=None,
    raw_dat_file=None,
    data_dir=None,
    elements=None,
    metric='abs_rel_diff',
    floor=1e-6,
    output_file=None,
    show=False,
):
    """
    Plot where bundled Kobayashi yields differ from the raw DAT table.

    Parameters
    ----------
    metallicities : sequence of float, optional
        Metallicities to compare. Defaults to all Kobayashi metallicities.
    raw_dat_file : Path or str, optional
        Path to ``SN2SAGBYIELD.DAT``.
    data_dir : Path or str, optional
        Directory containing Kobayashi yield files.
    elements : sequence of str, optional
        Elements to include. Defaults to all model elements.
    metric : {'abs_rel_diff', 'abs_diff', 'signed_rel_diff'}, optional
        Difference metric to visualize.
    floor : float, optional
        Minimum value used before taking ``log10`` for the heatmap.
    output_file : Path or str, optional
        If provided, save the figure to this path.
    show : bool, optional
        If True, display the figure.

    Returns
    -------
    tuple
        ``(fig, comparison_df)`` where ``comparison_df`` stacks all metallicities.
    """
    if metallicities is None:
        metallicities = list(KobayashiYields.AVAILABLE_Z)

    metric_options = {'abs_rel_diff', 'abs_diff', 'signed_rel_diff'}
    if metric not in metric_options:
        raise ValueError(f"metric must be one of {sorted(metric_options)}, got {metric!r}")

    comparison_frames = []
    for metallicity in metallicities:
        frame = compare_kobayashi_yields_to_raw_dat(
            metallicity=metallicity,
            raw_dat_file=raw_dat_file,
            data_dir=data_dir,
            elements=elements,
        ).copy()
        frame['abs_rel_diff'] = np.abs(frame['rel_diff'])
        frame['signed_rel_diff'] = frame['rel_diff']
        comparison_frames.append(frame)

    if not comparison_frames:
        raise ValueError("No Kobayashi comparisons were generated")

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    plot_elements = list(elements) if elements is not None else list(comparison_frames[0]['element'].unique())

    n_panels = len(comparison_frames)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.4 * nrows),
        dpi=150,
        squeeze=False,
    )

    image = None
    for idx, frame in enumerate(comparison_frames):
        ax = axes[idx // ncols, idx % ncols]
        mass_grid = np.sort(frame['mass'].unique())
        matrix = np.full((len(plot_elements), len(mass_grid)), np.nan)

        for elem_idx, element in enumerate(plot_elements):
            element_frame = frame[frame['element'] == element]
            for mass_idx, mass in enumerate(mass_grid):
                matched = element_frame[np.isclose(element_frame['mass'], mass)]
                if matched.empty:
                    continue
                value = float(matched.iloc[0][metric])
                if metric == 'signed_rel_diff':
                    value = np.sign(value) * np.log10(max(abs(value), floor))
                else:
                    value = np.log10(max(abs(value), floor))
                matrix[elem_idx, mass_idx] = value

        image = ax.imshow(matrix, aspect='auto', cmap='magma', interpolation='nearest')
        ax.set_title(f"Z = {float(frame['metallicity'].iloc[0]):.3g}")
        ax.set_xticks(np.arange(len(mass_grid)))
        ax.set_xticklabels([f"{mass:g}" for mass in mass_grid], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(plot_elements)))
        ax.set_yticklabels(plot_elements)
        ax.set_xlabel(r'M$_{\mathrm{ZAMS}}$ (M$_\odot$)')
        ax.set_ylabel('Element')

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    if image is not None:
        if metric == 'signed_rel_diff':
            cbar_label = r'sign(diff) $\times$ log$_{10}$(|relative difference|)'
        elif metric == 'abs_diff':
            cbar_label = r'log$_{10}$(|yield difference|)'
        else:
            cbar_label = r'log$_{10}$(|relative difference|)'
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label(cbar_label)

    fig.suptitle('Kobayashi simplified yields vs raw DAT table', fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.12, top=0.9, wspace=0.3, hspace=0.45)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, comparison_df


def load_nozawa2003_dust_yields(data_file=None):
    """
    Load Nozawa et al. (2003) Pop III dust yields from the repository table.

    Parameters
    ----------
    data_file : Path or str, optional
        Path to ``Nozawa2003_dust_yields.dat``.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``channel``, ``composition``, ``progenitor_mass``,
        and ``dust_mass`` (both masses in solar masses).
    """
    data_path = Path(data_file) if data_file else DEFAULT_NOZAWA2003_DUST_YIELDS
    if not data_path.exists():
        raise FileNotFoundError(f"Nozawa dust-yield file not found: {data_path}")

    rows = []
    current_channel = None
    current_composition = None

    with data_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('#'):
                header = line[1:].strip()
                if not header:
                    continue
                if set(header) == {'='}:
                    continue
                if header.startswith('==========') and header.endswith('=========='):
                    current_channel = header.strip('=').strip()
                    current_composition = None
                    continue
                current_composition = header
                continue

            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 2:
                continue

            progenitor_mass = float(parts[0])
            dust_mass = float(parts[1])
            rows.append(
                {
                    'channel': current_channel,
                    'composition': current_composition,
                    'progenitor_mass': progenitor_mass,
                    'dust_mass': dust_mass,
                }
            )

    if not rows:
        raise ValueError(f"No dust-yield data rows found in {data_path}")

    frame = pd.DataFrame(rows)
    if frame['channel'].isna().any() or frame['composition'].isna().any():
        raise ValueError(
            f"Malformed Nozawa data in {data_path}: each numeric row must be under a channel and composition header"
        )
    return frame


def load_nozawa2003_dust_dist(data_file=None):
    """
    Load Nozawa et al. (2003) Pop III dust size distributions from the repository table.

    Parameters
    ----------
    data_file : Path or str, optional
        Path to ``Nozawa2003_dust_dist.dat``.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``channel``, ``composition``, ``grain_size``,
        and ``dN_da``.
    """
    data_path = Path(data_file) if data_file else DEFAULT_NOZAWA2003_DUST_DIST
    if not data_path.exists():
        raise FileNotFoundError(f"Nozawa dust-distribution file not found: {data_path}")

    rows = []
    current_channel = None
    current_composition = None

    with data_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('#'):
                header = line[1:].strip()
                if not header:
                    continue
                if set(header) == {'='}:
                    continue
                if header.startswith('==========') and header.endswith('=========='):
                    current_channel = header.strip('=').strip()
                    current_composition = None
                    continue
                current_composition = header
                continue

            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 2:
                continue

            grain_size = float(parts[0])
            dn_da = float(parts[1])
            rows.append(
                {
                    'channel': current_channel,
                    'composition': current_composition,
                    'grain_size': grain_size,
                    'dN_da': dn_da,
                }
            )

    if not rows:
        raise ValueError(f"No dust-distribution data rows found in {data_path}")

    frame = pd.DataFrame(rows)
    if frame['channel'].isna().any() or frame['composition'].isna().any():
        raise ValueError(
            f"Malformed Nozawa data in {data_path}: each numeric row must be under a channel and composition header"
        )
    return frame


def plot_nozawa2003_dust_dist(
    data_file=None,
    channels=None,
    output_file=None,
    show=False,
):
    """
    Plot Nozawa (2003) grain size distributions with enhanced aesthetics.

    Parameters
    ----------
    data_file : Path or str, optional
        Path to ``Nozawa2003_dust_dist.dat``.
    channels : sequence of str, optional
        Channels to plot. Defaults to all in the file.
    output_file : Path or str, optional
        If provided, save figure to this path.
    show : bool, optional
        If True, display the figure.

    Returns
    -------
    tuple
        ``(fig, df)`` with the generated figure and data.
    """
    import matplotlib.ticker as ticker
    
    df = load_nozawa2003_dust_dist(data_file=data_file)
    if channels is None:
        channels = df['channel'].unique()

    n_channels = len(channels)
    fig, axes = plt.subplots(
        1,
        n_channels,
        figsize=(7 * n_channels, 6),
        dpi=150,
        squeeze=False,
    )

    # Use a professional color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, channel in enumerate(channels):
        ax = axes[0, i]
        channel_df = df[df['channel'] == channel]
        compositions = channel_df['composition'].unique()

        for j, comp in enumerate(compositions):
            comp_df = channel_df[channel_df['composition'] == comp]
            ax.plot(
                comp_df['grain_size'], 
                comp_df['dN_da'], 
                label=comp,
                color=colors[j % 10],
                linewidth=2.0,
                alpha=0.9
            )

        # Log scales with minor ticks
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Enhanced labels and titles
        ax.set_xlabel(r'Radius $r$ (cm)', fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel(r'Distribution $f_j(r)$ (relative)', fontsize=14, fontweight='bold')
        
        ax.set_title(channel, fontsize=16, fontweight='bold', pad=15)
        
        # Formatting ticks
        ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
        ax.tick_params(axis='both', which='minor', length=3, width=1.0)
        
        # Grid and Legend
        ax.grid(True, which='both', linestyle='--', alpha=0.4, color='gray')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8, edgecolor='none', ncol=2)
        
        # Boundary limits based on data
        ax.set_xlim(df['grain_size'].min() * 0.8, df['grain_size'].max() * 1.2)
        ax.set_ylim(bottom=1e0)

    fig.suptitle('Nozawa et al. (2003) Pop III Dust Models', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, df


def plot_nozawa2003_dust_yields_with_splines(
    data_file=None,
    channels=None,
    min_mass=10.0,
    max_mass=40.0,
    spline_degree=2,
    output_file=None,
    show=False,
):
    """
    Plot Nozawa (2003) dust masses with per-composition markers and LSQ splines.

    Parameters
    ----------
    data_file : Path or str, optional
        Path to ``Nozawa2003_dust_yields.dat``.
    channels : sequence of str, optional
        Sections/channels to plot in separate panels. Defaults to
        ``['CCSNe (unmixed)', 'PISNe (unmixed)']``.
    min_mass : float, optional
        Minimum progenitor mass for plotting range (Msun).
    max_mass : float, optional
        Maximum progenitor mass for plotting range (Msun).
    spline_degree : int, optional
        Polynomial degree ``k`` used by ``LSQUnivariateSpline``.
    output_file : Path or str, optional
        If provided, save figure to this path.
    show : bool, optional
        If True, display the figure.

    Returns
    -------
    tuple
        ``(fig, filtered_df)`` with the generated figure and plotted data.
    """

    all_data = load_nozawa2003_dust_yields(data_file=data_file)
    if channels is None:
        channels = ['CCSNe (unmixed)', 'PISNe (unmixed)', 'CCSNe (mixed)', 'PISNe (mixed)']
    channels = [channels] if isinstance(channels, str) else list(channels)
    if not channels:
        raise ValueError("channels must contain at least one channel name")

    subset = all_data[
        all_data['channel'].isin(channels)
    ].copy()

    compositions = sorted(subset['composition'].unique())
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', '*', 'h', '8']
    color_map = plt.get_cmap('tab10')

    fig, axes = plt.subplots(
        1,
        len(channels),
        figsize=(6.8 * len(channels), 5.4),
        dpi=150,
        squeeze=False,
    )
    axes = axes[0]
    min_CCSNe_mass, max_CCSNe_mass = 10., 40.
    min_PISNe_mass, max_PISNe_mass = 150., 250.
    ymin_CCSNe, ymax_CCSNe = 1e-4, 1
    ymin_PISNe, ymax_PISNe = 1e-2, 100
    positive_floor = 1e-30

    for panel_idx, channel_name in enumerate(channels):
        ax = axes[panel_idx]
        channel_data = subset[subset['channel'] == channel_name].copy()
        min_mass = min_CCSNe_mass if 'CCSNe' in channel_name else min_PISNe_mass
        max_mass = max_CCSNe_mass if 'CCSNe' in channel_name else max_PISNe_mass
        x_fit = np.linspace(min_mass, max_mass, 200)

        for comp_idx, composition in enumerate(compositions):
            comp_data = channel_data[channel_data['composition'] == composition].copy()
            if comp_data.empty:
                continue
            comp_data = comp_data.sort_values('progenitor_mass')
            comp_data = comp_data.drop_duplicates(subset=['progenitor_mass'], keep='last')

            x = comp_data['progenitor_mass'].to_numpy(dtype=float)
            y = comp_data['dust_mass'].to_numpy(dtype=float)
            marker = marker_cycle[comp_idx % len(marker_cycle)]
            color = color_map(comp_idx % 10)

            y_safe = np.where(y > positive_floor, y, positive_floor)

            ax.scatter(
                x,
                y_safe,
                marker=marker,
                s=42,
                linewidths=0.6,
                edgecolors='black',
                color=color,
                alpha=0.9,
                label=composition,
                zorder=3,
            )

            if len(x) < 2:
                continue

            x_unique = np.unique(x)
            max_degree = min(int(spline_degree), len(x_unique) - 1)
            if max_degree < 1:
                continue

            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y_safe[order]

            n_points = len(x_sorted)
            max_internal_knots = n_points - (max_degree + 1)
            n_internal_knots = max(0, min(2, max_internal_knots))
            knots = np.array([])
            if n_internal_knots > 0:
                candidate = np.linspace(x_sorted.min(), x_sorted.max(), n_internal_knots + 2)[1:-1]
                mask = (candidate > x_sorted.min()) & (candidate < x_sorted.max())
                knots = candidate[mask]

            spline = LSQUnivariateSpline(x_sorted, y_sorted, t=knots, k=max_degree)
            y_fit = spline(x_fit)

            y_fit = np.where(y_fit > positive_floor, y_fit, positive_floor)
            ax.plot(
                x_fit,
                y_fit,
                color=color,
                linewidth=1.8,
                alpha=0.95,
                label='_nolegend_',
                zorder=2,
            )

        ax.set_xlim(float(min_mass), float(max_mass))
        if 'CCSNe' in channel_name:
            ax.set_ylim(ymin_CCSNe, ymax_CCSNe)
        else:
            ax.set_ylim(ymin_PISNe, ymax_PISNe)
        ax.set_yscale('log')
        ax.set_xlabel(r'Progenitor mass (M$_\odot$)')
        if panel_idx == 0:
            ax.set_ylabel(r'Dust mass (M$_\odot$)')
        ax.set_title(channel_name)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc='best', fontsize=8, ncol=2)

    fig.suptitle('Nozawa et al. (2003) Pop III dust yields (Umeda & Nomoto 2002 SN models)')

    fig.tight_layout()

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, subset.sort_values(['composition', 'progenitor_mass']).reset_index(drop=True)


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
        
        # Read file and parse yields.
        # Kobayashi simplified files are row-wise tables with one row per
        # (mass, element). Raw intermediate files are matrix-like tables.
        df = pd.read_csv(filename, sep=r'\s+', comment='#', engine='python', header=None)

        if df.shape[1] >= 7:
            masses = np.sort(df.iloc[:, 0].astype(float).unique())
            yields = np.zeros((len(masses), len(self.elements)))

            for mass_idx, mass in enumerate(masses):
                mass_rows = df[np.isclose(df.iloc[:, 0].astype(float), mass)]
                for _, row in mass_rows.iterrows():
                    raw_element = str(row.iloc[4]).strip()
                    element = 'H' if raw_element == 'p' else raw_element
                    if element not in self.elements:
                        continue
                    element_idx = self.elements.index(element)
                    yields[mass_idx, element_idx] = float(row.iloc[5])

            self.masses = masses
            self.yields = yields
            return

        self.masses = df.iloc[:, 0].astype(float).values
        self.yields = df.iloc[:, 1:].astype(float).values
    
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
