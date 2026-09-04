"""Load initial conditions for the dust chemistry ODE solver from a JSON file.

Public API
----------
``load_initial_conditions(config_path)``
    Parse the JSON file and return ``(DustChemistryState, y_gas_0, y_dust_0)``.

JSON schema (see ``configs/example_ic.json`` for a complete example)
---------------------------------------------------------------------
{
    "model_data_dir": "model_data",          // optional, default auto-detected
    "environment": {
        "gas_temperature_K": 100.0,
        "hydrogen_number_density_cm3": 100.0,
        "electron_number_density_cm3": 1e-3,
        "radiation_field_G0": 1.0,
        "radiation_field_model": "habing",   // informational
        "mean_molecular_weight": 1.4
    },
    "elemental_abundances": {
        "H":  {"mass_fraction": 0.706},
        ...
    },
    "dust_bins": [
        {
            "id": "DustBin_01",
            "composition": "graphite",
            "grain_size_micron": 0.01,
            "grain_density_gcm3": 2.24,
            "elements": [{"name": "C", "mass_fraction": 1.0}],
            "initial_mass_density_gcm3": 8e-28,
            "sticking_coefficient": 1.0,
            "nhmax_acc": 1e4,
            "nh_coa": 0.1,
            "coagulation_partner": "DustBin_02"   // optional bin id
        },
        ...
    ],
    "pah_bins": [
        {
            "id": "PAHbin_01",
            "nc": 54,
            "nc_min": 24,
            "initial_mass_density_gcm3": 1e-29
        },
        ...
    ],
    "physics": {
        "dust_accretion": true,
        "dust_sputtering": true,
        "dust_sublimation": true,
        "dust_coagulation": true,
        "pah_accretion": false,
        "pah_photolysis": false
    },
    "solver": {
        "type": "rk4",
        "t_end_Myr": 100.0,
        "h_init_s": 1e10,
        "h_min_s": 1.0,
        "h_max_Myr": 1.0,
        "errmax": 0.1,
        "countmax": 10000
    }
}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np

# pycalima._paths is a leaf module that imports nothing from models/ or
# solvers/, so this preserves the property that solvers/ does not depend on
# models/. Note this is the model-agnostic model_data root: unlike
# models.grain_size_config.get_model_data_dir() it does not append a
# `model_name` subdirectory, matching the behaviour the solver configs have
# always had.
from pycalima._paths import get_model_data_dir

from .chemistry_state import (
    AU2G,
    ELEMENT_ATOMIC_MASSES_G,
    ELEMENT_NAMES,
    N_ELEMENTS,
    DustBinParams,
    DustChemistryState,
    PAHBinParams,
)
from .table_io import (
    build_sputtering_interpolator,
    build_pah_photolysis_interpolator,
    build_pah_sputtering_interpolator,
    build_charge_interpolator,
)

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------

KB_CGS: float = 1.3806488e-16   # Boltzmann constant [erg K⁻¹]
MH_CGS: float = 1.6726219e-24   # Proton mass [g]
GNEWT: float  = 6.6743e-8        # Gravitational constant [cm³ g⁻¹ s⁻²]
PI: float = math.pi
SEC2YR: float = 3.1536e7         # seconds per year [s yr⁻¹]
SEC2MYR: float = 3.1536e13       # seconds per Myr  [s Myr⁻¹]


def freefall_time_s(nH: float, mu: float = 1.4) -> float:
    """Local free-fall time in seconds.

    ``t_ff = sqrt(3π / (32 G ρ))``

    Parameters
    ----------
    nH : float
        Hydrogen number density [cm⁻³].
    mu : float
        Mean molecular weight (default 1.4 for neutral gas).

    Returns
    -------
    float
        Free-fall time [s].
    """
    rho = max(nH, 1e-300) * MH_CGS * mu
    return math.sqrt(3.0 * PI / (32.0 * GNEWT * rho))

# Sputtering tables exist for these ion species (atomic-number-keyed)
# Must match the naming used in CALIMA's export_sputtering_rates_bins.py
_ION_SPECIES = [
    ("H",  1),
    ("He", 2),
    ("C",  6),
    ("N",  7),
    ("O",  8),
    ("Ne", 10),
    ("Mg", 12),
    ("Si", 14),
    ("S",  16),
    ("Fe", 26),
]

# PAH sputtering species: electrons (Z=0) plus ions
_PAH_SPUTTERING_SPECIES = [
    ("electrons", 0),
    ("H",  1),
    ("He", 2),
    ("C",  6),
    ("O",  8),
]


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def _k0_accretion(asize_cm: float, sgrain: float, sticking: float = 1.0) -> float:
    """Accretion rate constant k₀_acc [cm³ g⁻⁰·⁵ K⁻⁰·⁵ s⁻¹].

    The grain-growth rate per unit dust density (Dubois et al. 2024 / CALIMA):

        rate [s⁻¹] = k0_acc × √T × y_gas[e] / (el_mfrac × √m_e_g)

    where  k0_acc = π a² S √(8 k_B / π) / m_grain .
    The sticking coefficient S is included here; no further T-dependent
    suppression is applied in accretion_rate().
    """
    m_grain = (4.0 / 3.0) * PI * asize_cm ** 3 * sgrain
    cross_section = PI * asize_cm ** 2
    return cross_section * sticking * math.sqrt(8.0 * KB_CGS / PI) / m_grain


def _k0_coagulation(asize_cm: float, sgrain: float) -> float:
    """Coagulation rate constant k₀_coa [cm³ g⁻¹ s⁻¹].

    From Aoyama et al. (2017)

        rate [s⁻¹] = k0_coa × y_dust[small_bin] / n_H

    k0_coa = √(8/(3π)) × π (2a)² / m_grain  ≡ collision cross section / mass.
    """
    m_grain = (4.0 / 3.0) * PI * asize_cm ** 3 * sgrain
    return math.sqrt(8.0 / (3.0 * PI)) * PI * (2.0 * asize_cm) ** 2 / m_grain


def _load_sputtering_tables(bin_id: str, data_dir: Path) -> dict:
    """Load thermal sputtering tables for all ion species for *bin_id*.

    Returns a dict mapping ``element_name → evaluate_fn`` where
    ``evaluate_fn(T, phi=0.0)`` gives the rate in [µm yr⁻¹ cm³].
    Missing table files are silently skipped.
    """
    interps: dict = {}
    for el_name, Z in _ION_SPECIES:
        fname = data_dir / f"thermal_sputtering_{bin_id}_Z_{Z}"
        if not fname.exists():
            continue
        try:
            evaluate_fn, _ = build_sputtering_interpolator(fname)
            interps[el_name] = evaluate_fn
        except Exception:
            pass  # corrupt table – skip gracefully
    return interps


def _load_pah_photolysis_table(bin_id: str, data_dir: Path):
    """Load PAH photolysis 2-D table for *bin_id*.

    Returns a callable ``(log10_G0, log10_nH) → log10(rate [s⁻¹])`` or
    ``None`` if the file does not exist.
    """
    fname = data_dir / f"dissociation_{bin_id}.dat"
    if not fname.exists():
        return None
    try:
        evaluate_fn, _ = build_pah_photolysis_interpolator(fname)
        return evaluate_fn
    except Exception:
        return None


def _load_pah_sputtering_tables(bin_id: str, data_dir: Path) -> dict:
    """Load PAH sputtering 1-D T-tables for all ion species for *bin_id*.

    Returns a dict mapping ``species_name → evaluate_fn(T)``
    where ``evaluate_fn(T)`` gives J [cm³ s⁻¹].
    Keys: ``'electrons'``, ``'H'``, ``'He'``, ``'C'``, ``'O'`` etc.
    """
    interps: dict = {}
    for species_name, Z in _PAH_SPUTTERING_SPECIES:
        fname = data_dir / f"sputtering_{bin_id}_Z_{Z}"
        if not fname.exists():
            continue
        try:
            evaluate_fn, _ = build_pah_sputtering_interpolator(fname)
            interps[species_name] = evaluate_fn
        except Exception:
            pass
    return interps


def _load_erosion_rate_table(bin_id: str, data_dir: Path):
    """Load pre-computed GD89 sublimation/erosion-rate table for *bin_id*.

    Returns a callable ``(T_K) -> epsilon [s^-1]`` that linearly
    interpolates the table produced by
    ``models.dust_radiation.dust_sublimation.write_sublimation_rate_tables``.
    Returns ``None`` when the file does not exist or cannot be read.

    The expected file is ``<data_dir>/sublimation_rate_{bin_id}.dat`` (falling back to
    ``erosion_rate_{bin_id}.dat``), with two columns (temperature [K], rate [s⁻¹]).
    Lines starting with ``#`` are treated as comments and skipped.
    """
    fname = data_dir / f"sublimation_rate_{bin_id}.dat"
    if not fname.exists():
        fname = data_dir / f"erosion_rate_{bin_id}.dat"
    if not fname.exists():
        return None
    try:
        data = np.loadtxt(fname, comments="#")
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        T_tab = data[:, 0]
        eps_tab = data[:, 1]
        # Return a simple linear-log interpolant (extrapolate as 0)
        from scipy.interpolate import interp1d
        interp = interp1d(
            T_tab, eps_tab,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, eps_tab[-1]),
        )
        return interp
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_initial_conditions(
    config_path: str | Path,
) -> Tuple[DustChemistryState, np.ndarray, np.ndarray]:
    """Load dust chemistry initial conditions from a JSON file.

    Parameters
    ----------
    config_path :
        Path to the initial conditions JSON file.

    Returns
    -------
    state : DustChemistryState
    y_gas : ndarray, shape (n_elements,)
        Gas-phase element mass densities [g cm⁻³].
    y_dust : ndarray, shape (npah + ndust,)
        PAH (first) and dust mass densities [g cm⁻³].
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    # "model_data_dir" is optional. Missing, null or "auto" means "resolve via
    # pycalima._paths" ($CALIMA_MODEL_DATA -> $CALIMA_DATA/model_data ->
    # ./model_data -> the per-user data directory). A *relative* value is
    # resolved against the config file rather than the process CWD, so a
    # config and its tables can be moved together.
    raw_mdd = cfg.get("model_data_dir")
    if raw_mdd in (None, "", "auto"):
        model_data_dir = get_model_data_dir()
    else:
        model_data_dir = Path(raw_mdd).expanduser()
        if not model_data_dir.is_absolute():
            model_data_dir = (config_path.parent / model_data_dir).resolve()

    if not model_data_dir.is_dir():
        raise FileNotFoundError(
            f"model_data directory not found: {model_data_dir}\n"
            f"Generate the rate tables first:  calima-export\n"
            f"Or point at an existing set:      export CALIMA_MODEL_DATA=/path/to/model_data\n"
            f'Or set "model_data_dir" in {config_path}.'
        )

    sputtering_dir = model_data_dir / "thermal_sputtering_data"
    pah_photolysis_dir = model_data_dir / "PAH_dissociation_data"
    pah_sputtering_dir = model_data_dir / "pah_sputtering_data"
    sublimation_dir = model_data_dir / "dust_sublimation"

    # ------------------------------------------------------------------
    # Gas environment
    # ------------------------------------------------------------------
    env = cfg["environment"]
    local_Tk = float(env["gas_temperature_K"])
    local_nH = float(env["hydrogen_number_density_cm3"])
    local_mu = float(env.get("mean_molecular_weight", 1.4))
    local_rho = float(
        env.get("gas_mass_density_gcm3", local_nH * MH_CGS * local_mu)
    )
    local_ne = float(env.get("electron_number_density_cm3", 1.0e-4 * local_nH))
    local_G0 = float(env.get("radiation_field_G0", 1.0))

    # ------------------------------------------------------------------
    # Elemental abundances → y_gas
    # ------------------------------------------------------------------
    y_gas = np.zeros(N_ELEMENTS, dtype=np.float64)
    for el_cfg in cfg.get("elemental_abundances", {}).items():
        el_name, props = el_cfg
        if el_name in ELEMENT_NAMES:
            mfrac = float(props.get("mass_fraction", 0.0))
            y_gas[ELEMENT_NAMES.index(el_name)] = mfrac * local_rho

    # ------------------------------------------------------------------
    # Dust bins
    # ------------------------------------------------------------------
    dust_cfgs = cfg.get("dust_bins", [])
    dust_bins: list[DustBinParams] = []

    for bd in dust_cfgs:
        bin_id = bd["id"]
        composition = bd.get("composition", "graphite")
        asize_um = float(bd["grain_size_micron"])
        asize_cm = asize_um * 1.0e-4
        sgrain = float(bd["grain_density_gcm3"])
        mgrain = (4.0 / 3.0) * PI * asize_cm ** 3 * sgrain
        sticking = float(bd.get("sticking_coefficient", 1.0))

        # Element composition
        el_names: list[str] = []
        el_indices: list[int] = []
        el_mfracs: list[float] = []
        for ec in bd.get("elements", []):
            ename = ec["name"]
            if ename in ELEMENT_NAMES:
                el_names.append(ename)
                el_indices.append(ELEMENT_NAMES.index(ename))
                el_mfracs.append(float(ec["mass_fraction"]))

        k0_acc = _k0_accretion(asize_cm, sgrain, sticking)
        k0_coa = _k0_coagulation(asize_cm, sgrain)
        sput_interps = _load_sputtering_tables(bin_id, sputtering_dir)
        erosion_interp = _load_erosion_rate_table(bin_id, sublimation_dir)

        # Grain charge tables (for Coulomb enhancement in accretion/sputtering)
        charge_Z_interp = None
        charge_sigma_interp = None
        charge_dir = model_data_dir / "dust_charging_data"
        _fZ = charge_dir / f"dust_charge_Z_vs_T_{bin_id}"
        _fS = charge_dir / f"dust_charge_sigma_vs_T_{bin_id}"
        if _fZ.exists() and _fS.exists():
            try:
                charge_Z_interp = build_charge_interpolator(_fZ, clamp_range=(-60.0, 60.0))
                charge_sigma_interp = build_charge_interpolator(_fS, clamp_range=(0.0, 20.0))
            except Exception:
                pass  # table unreadable — disable Coulomb enhancement for this bin

        # Grain mass range for fragment distribution (shattering)
        amin_um = float(bd.get("amin_micron", asize_um * 0.5))
        amax_um = float(bd.get("amax_micron", asize_um * 2.0))
        amin_cm = amin_um * 1.0e-4
        amax_cm = amax_um * 1.0e-4
        mgrain_min = (4.0 / 3.0) * PI * amin_cm ** 3 * sgrain
        mgrain_max = (4.0 / 3.0) * PI * amax_cm ** 3 * sgrain

        db = DustBinParams(
            bin_id=bin_id,
            composition=composition,
            bin_index=len(dust_bins),
            asize_micron=asize_um,
            asize_cm=asize_cm,
            sgrain=sgrain,
            mgrain=mgrain,
            el_names=el_names,
            el_indices=el_indices,
            el_mfractions=el_mfracs,
            k0_acc=k0_acc,
            k0_coa=k0_coa,
            sputtering_interps=sput_interps,
            erosion_rate_interp=erosion_interp,
            charge_Z_interp=charge_Z_interp,
            charge_sigma_interp=charge_sigma_interp,
            nhmax_acc=float(bd.get("nhmax_acc", 1.0e4)),
            nh_coa=float(bd.get("nh_coa", 0.1)),
            catastrophic_spec_energy=float(
                bd.get("catastrophic_specific_energy_erg_g", 1.0e7)
            ),
            mgrain_min=mgrain_min,
            mgrain_max=mgrain_max,
            interact_pah=bool(bd.get("interact_pah", False)),
            vthresh_coag=float(bd.get("vthresh_coag_cm_s", 1.0e4)),
        )
        # Store the coag-partner ID as a string; resolve to index below
        if "coagulation_partner" in bd:
            db.coag_partner_index = bd["coagulation_partner"]  # type: ignore[assignment]
        dust_bins.append(db)

    # Resolve coagulation partner string IDs → integer indices
    id_to_idx = {db.bin_id: db.bin_index for db in dust_bins}
    for db in dust_bins:
        if isinstance(db.coag_partner_index, str):
            db.coag_partner_index = id_to_idx.get(db.coag_partner_index)

    # ------------------------------------------------------------------
    # PAH bins
    # ------------------------------------------------------------------
    pah_cfgs = cfg.get("pah_bins", [])
    pah_bins: list[PAHBinParams] = []

    for pb in pah_cfgs:
        bin_id = pb["id"]
        nc = int(pb.get("nc", 54))
        nc_min = int(pb.get("nc_min", 24))
        nc_max = int(pb.get("nc_max", nc * 2))
        spah = float(pb.get("grain_density_gcm3", 2.24))
        mpah = float(pb.get("mpah_g", nc * 12.011 * AU2G))
        mpah_min = nc_min * 12.011 * AU2G
        mpah_max = nc_max * 12.011 * AU2G
        apah_cm = (3.0 * mpah / (4.0 * PI * spah)) ** (1.0 / 3.0)

        # Load photolysis and sputtering tables
        dissociation_interp = _load_pah_photolysis_table(bin_id, pah_photolysis_dir)
        sput_interps_pah = _load_pah_sputtering_tables(bin_id, pah_sputtering_dir)

        pah_bin = PAHBinParams(
            bin_id=bin_id,
            bin_index=len(pah_bins),
            nc=nc,
            nc_min=nc_min,
            nc_max=nc_max,
            mpah=mpah,
            spah=spah,
            fpah=float(pb.get("fpah", 0.1)),
            apah_cm=apah_cm,
            mpah_min=mpah_min,
            mpah_max=mpah_max,
            dissociation_interp=dissociation_interp,
            sputtering_interps=sput_interps_pah,
            is_cluster=bool(pb.get("is_cluster", False)),
            dust_index_interact=int(pb.get("dust_index_interact", -1)),
            nd_bins=int(pb.get("nd_bins_interact", 0)),
        )
        pah_bins.append(pah_bin)

    # ------------------------------------------------------------------
    # y_dust initial conditions  (PAH bins first, then dust bins)
    # ------------------------------------------------------------------
    npah = len(pah_bins)
    ndust = len(dust_bins)
    y_dust = np.zeros(npah + ndust, dtype=np.float64)

    for i, bd in enumerate(dust_cfgs):
        y_dust[npah + i] = float(bd.get("initial_mass_density_gcm3", 0.0))

    for i, pb in enumerate(pah_cfgs):
        y_dust[i] = float(pb.get("initial_mass_density_gcm3", 0.0))

    # ------------------------------------------------------------------
    # Subtract initial dust metals from gas-phase budget.
    #
    # elemental_abundances in the config represents the *total* metal budget
    # (gas + dust) at the reference state.  The metals already locked in the
    # initial dust must be removed from y_gas so that the conserved quantity
    #   y_gas[el] + Σ_bins( y_dust[b] × el_mfrac[b] )
    # equals the true gas-phase metal supply at t = 0.
    # Without this correction, y_gas over-counts by the initial dust content,
    # allowing accretion to produce more dust than the total metal budget.
    # ------------------------------------------------------------------
    for i, bd in enumerate(dust_cfgs):
        rho_dust_i = y_dust[npah + i]
        if rho_dust_i <= 0.0:
            continue
        for ec in bd.get("elements", []):
            ename = ec.get("name", "")
            if ename in ELEMENT_NAMES:
                el_idx = ELEMENT_NAMES.index(ename)
                el_mfrac = float(ec.get("mass_fraction", 0.0))
                y_gas[el_idx] = max(0.0, y_gas[el_idx] - rho_dust_i * el_mfrac)

    # ------------------------------------------------------------------
    # Physics flags and solver settings
    # ------------------------------------------------------------------
    phys = cfg.get("physics", {})
    solver_cfg = cfg.get("solver", {})
    models_cfg = cfg.get("models", {})
    turb_cfg = cfg.get("turbulence", {})

    # Turbulent velocity and injection scale (convert from km/s and pc)
    _PC2CM = 3.085677581e18  # 1 pc in cm
    local_sigma = float(turb_cfg.get("local_sigma_km_s", 0.0)) * 1.0e5
    local_dx = float(turb_cfg.get("local_dx_pc", 0.0)) * _PC2CM

    state = DustChemistryState(
        local_Tk=local_Tk,
        local_nH=local_nH,
        local_rho=local_rho,
        local_ne=local_ne,
        local_G0=local_G0,
        local_mu=local_mu,
        ndust=ndust,
        npah=npah,
        dust_bins=dust_bins,
        pah_bins=pah_bins,
        el_names=list(ELEMENT_NAMES),
        el_atomic_mass_g=list(ELEMENT_ATOMIC_MASSES_G),
        errmax=float(solver_cfg.get("errmax", 0.1)),
        countmax=int(solver_cfg.get("countmax", 10000)),
        # Physics flags
        dust_accretion=bool(phys.get("dust_accretion", False)),
        dust_sputtering=bool(phys.get("dust_sputtering", False)),
        dust_sublimation=bool(phys.get("dust_sublimation", False)),
        dust_coagulation=bool(phys.get("dust_coagulation", False)),
        dust_shattering=bool(phys.get("dust_shattering", False)),
        pah_accretion=bool(phys.get("pah_accretion", False)),
        pah_photolysis=bool(phys.get("pah_photolysis", False)),
        pah_sputtering=bool(phys.get("pah_sputtering", False)),
        pah_coalescence=bool(phys.get("pah_coalescence", False)),
        pah_cluster_evaporation=bool(phys.get("pah_cluster_evaporation", False)),
        pah_freezing=bool(phys.get("pah_freezing", False)),
        # Model selection
        coagulation_model=str(models_cfg.get("coagulation_model", "Aoyama2017")),
        shattering_model=str(models_cfg.get("shattering_model", "turbulent")),
        dust_velocity_model=str(models_cfg.get("dust_velocity_model", "Ormel2007")),
        coalescence_model=str(models_cfg.get("coalescence_model", "Totton2012")),
        dust_sputtering_model=str(models_cfg.get("dust_sputtering_model", "kirchschlager")),
        photolysis_model=str(models_cfg.get("photolysis_model", "RM2026")),
        pah_sputtering_model=str(models_cfg.get("pah_sputtering_model", "RM2026")),
        cluster_evaporation_model=str(
            models_cfg.get("cluster_evaporation_model", "Montillaud2014")
        ),
        slope_frag_func=float(models_cfg.get("slope_frag_func", 1.3 / 3.0)),
        # Turbulent environment
        local_sigma=local_sigma,
        local_dx=local_dx,
        radiation_field_model=env.get("radiation_field_model", "habing"),
    )

    return state, y_gas, y_dust
