from .pah_mol_data import (
    load_pah_modes,
    compute_rrkm_dissociation_rate,
    compute_dissociation_rate_from_table,
    compute_thermal_ir_rate,
)
from .pah_charge_utils import (
    afromNc,
    ionisation_potential_energy,
    electron_affinity_energy,
    IONISATION_POTENTIAL,
    ELECTRON_AFFINITY,
)
from .pah_temperature import (
    compute_adaptive_temperature_distribution,
    compute_base_g0,
    get_absorption_cross_section,
)
from .pah_dissociation import (
    compute_branching_integrated_rates,
    compute_branching_integrated_rates_montillaud,
    compute_total_photon_absorption_rate,
    compare_dissociation_methods,
    print_method_comparison,
)
from .pah_radiation import load_kurucz_u_E, load_kurucz_I_nu
from .pah_db_lookup import (
    build_pahdb_catalog,
    load_pahdb_catalog,
    find_best_species,
    extract_modes_by_uid,
    get_modes_file,
)
from .pah_h_state import (
    DissociationChannel,
    compute_solo_duo_counts,
    get_dissociation_channels,
    get_dissociation_channels_montillaud,
    channel_params,
    # Andrews (2016) activation parameter constants
    HLOSS_EVEN_EACT_EV, HLOSS_EVEN_DS_JKM,
    HLOSS_ODD_EACT_EV,  HLOSS_ODD_DS_JKM,
    H2LOSS_EACT_EV,     H2LOSS_DS_JKM,
    SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV, SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM,
    SUPERH_HLOSS_CATION_EACT_EV,        SUPERH_HLOSS_CATION_DS_JKM,
    # Montillaud (2013) activation parameter constants
    M13_HLOSS_DEHYD_EACT_EV,   M13_HLOSS_DEHYD_DS_JKM,
    M13_H2LOSS_EACT_EV,        M13_H2LOSS_DS_JKM,
    M13_C2H2LOSS_DEHYD_EACT_EV, M13_C2H2LOSS_DEHYD_DS_JKM,
    M13_SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV, M13_SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM,
    M13_SUPERH_HLOSS_CATION_EACT_EV,        M13_SUPERH_HLOSS_CATION_DS_JKM,
    M13_SUPERH_C2H2LOSS_EACT_EV, M13_SUPERH_C2H2LOSS_DS_JKM,
    # parent topology defaults
    C24H12_NH0, C24H12_SOLO, C24H12_DUO,
    C54H18_NH0, C54H18_SOLO, C54H18_DUO,
    C96H24_NH0, C96H24_SOLO, C96H24_DUO,
)
from .pah_hydrogen_chemistry import (
    h_addition_rate,
    collisional_rate,
    reaction_efficiency_neutral,
    h2_abstraction_rate,
    h2_abstraction_rate_coefficient,
    K_CATION_CM3S,
    K_ANION_CM3S,
    EFF_DEHYDROGENATED,
    K_ER_COEFF_CM3S,
)
from .pah_network_solver import (
    PAHNetworkSolver,
    make_c54_solver,
    make_c96_solver,
    N_SUPERH_MAX,
)
