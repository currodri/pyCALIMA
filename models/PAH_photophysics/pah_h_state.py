"""
pah_h_state.py — PAH hydrogen state tracking and RRKM dissociation parameters.

Tracks solo/duo hydrogen topology as a function of total H count during
de-hydrogenation, and maps each (N_H, N_H0, charge) state to the correct
RRKM activation parameters following Andrews et al. (2016) or
Montillaud, Joblin & Toublanc (2013).

Andrews (2016) parameter table
-------------------------------
Normal / de-hydrogenated  (N_H <= N_H0):
  H-loss, even N_H    :  E_act = 4.60 eV,  dS =  44.8 J/K/mol
  H-loss, odd  N_H    :  E_act = 4.10 eV,  dS =  55.6 J/K/mol
  H2-loss (even N_H, N_H > N_solo):
                          E_act = 3.52 eV,  dS = -53.1 J/K/mol

Super-hydrogenated  (N_H > N_H0), H-loss from extra duo-ring H:
  Z <= 0              :  E_act = 1.40 eV,  dS =  55.6 J/K/mol
  Z >  0              :  E_act = 1.55 eV,  dS =  55.6 J/K/mol

Montillaud, Joblin & Toublanc (2013) / Berne & Tielens (2012) parameters
--------------------------------------------------------------------------
Entropy changes are given in the literature in cal mol^-1 K^-1; here they
are stored in J mol^-1 K^-1 (1 cal/mol/K = 4.184 J/mol/K).

Dehydrogenated regime (N_H <= N_H0):
  H-loss   (any Z)    :  E_act = 4.30 eV,  dS =  49.4 J/K/mol  (11.8 cal/mol/K)
  H2-loss             :  E_act = 3.52 eV,  dS = -53.1 J/K/mol  (-12.7 cal/mol/K)
  C2H2-loss           :  E_act = 4.60 eV,  dS =  41.8 J/K/mol  (10.0 cal/mol/K)

Super-hydrogenated regime (N_H > N_H0), loss of extra H:
  H-loss   (Z <= 0)   :  E_act = 1.40 eV,  dS =  55.6 J/K/mol  (13.3 cal/mol/K)
  H-loss   (Z >  0)   :  E_act = 1.55 eV,  dS =  55.6 J/K/mol  (13.3 cal/mol/K)
  C2H2-loss           :  E_act = 2.00 eV,  dS =  41.8 J/K/mol  (10.0 cal/mol/K)

The C2H2-loss channel becomes dominant for fully dehydrogenated PAHs and is
the primary pathway for PAH photodestruction at high G0. It is absent from the
Andrews (2016) table but included by Montillaud (2013) following Allain (1996).

These parameters feed into compute_rrkm_dissociation_rate from pah_mol_data.py.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Andrews (2016) RRKM activation parameters
# ---------------------------------------------------------------------------

# Normal / de-hydrogenated H-loss (charge-independent)
HLOSS_EVEN_EACT_EV = 4.60
HLOSS_EVEN_DS_JKM  = 44.8

HLOSS_ODD_EACT_EV  = 4.10
HLOSS_ODD_DS_JKM   = 55.6

# Normal / de-hydrogenated H2-loss
H2LOSS_EACT_EV     = 3.52
H2LOSS_DS_JKM      = -53.1

# Super-hydrogenated H-loss (extra H in duo ring)
SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV = 1.40
SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM  = 55.6

SUPERH_HLOSS_CATION_EACT_EV        = 1.55
SUPERH_HLOSS_CATION_DS_JKM         = 55.6

# ---------------------------------------------------------------------------
# Montillaud, Joblin & Toublanc (2013) / Berne & Tielens (2012) parameters
# Entropy values converted from cal/(mol K) to J/(mol K): × 4.184
# Reference: Montillaud+13 A&A 552 A15, Table 1 and Sect. 2.2.2
# ---------------------------------------------------------------------------

# Dehydrogenated H-loss (charge-independent in this parametrisation)
M13_HLOSS_DEHYD_EACT_EV  = 4.30
M13_HLOSS_DEHYD_DS_JKM   = 11.8 * 4.184   # 49.37 J/(mol K)

# Dehydrogenated H2-loss (consistent with Andrews 2016)
M13_H2LOSS_EACT_EV       = 3.52
M13_H2LOSS_DS_JKM        = -12.69 * 4.184  # -53.11 J/(mol K)

# Dehydrogenated C2H2-loss (acetylene channel; Allain 1996 / Berne & Tielens 2012)
M13_C2H2LOSS_DEHYD_EACT_EV = 4.60
M13_C2H2LOSS_DEHYD_DS_JKM  = 10.0 * 4.184  # 41.84 J/(mol K)

# Super-hydrogenated H-loss
M13_SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV = 1.40
M13_SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM  = 13.3 * 4.184  # 55.65 J/(mol K)

M13_SUPERH_HLOSS_CATION_EACT_EV        = 1.55
M13_SUPERH_HLOSS_CATION_DS_JKM         = 13.3 * 4.184

# Super-hydrogenated C2H2-loss (Berne & Tielens 2012)
M13_SUPERH_C2H2LOSS_EACT_EV = 2.00
M13_SUPERH_C2H2LOSS_DS_JKM  = 10.0 * 4.184  # 41.84 J/(mol K)

# ---------------------------------------------------------------------------
# Parent PAH topology defaults  (from NASA Ames PAHdb uid lookup)
# ---------------------------------------------------------------------------
# C24H12 coronene (uid=18, D6h): 0 solo + 12 duo = 12 H
C24H12_NH0        = 12
C24H12_SOLO       = 0
C24H12_DUO        = 12

# C54H18 circumcoronene (uid=37, D6h): 6 solo + 12 duo = 18 H
C54H18_NH0        = 18
C54H18_SOLO       = 6
C54H18_DUO        = 12

# C96H24 circumcircumcoronene (uid=108, Ag): 12 solo + 12 duo = 24 H
C96H24_NH0        = 24
C96H24_SOLO       = 12
C96H24_DUO        = 12


# ---------------------------------------------------------------------------
# Dissociation channel descriptor
# ---------------------------------------------------------------------------

@dataclass
class DissociationChannel:
    """
    One active dissociation channel with its Andrews (2016) RRKM parameters.

    Fields
    ------
    name      : channel identifier ('H_loss', 'H2_loss', 'H_loss_superH')
    E_act_ev  : activation energy [eV]
    dS_jkm    : activation entropy [J/K/mol]
    """
    name: str
    E_act_ev: float
    dS_jkm: float


# ---------------------------------------------------------------------------
# Solo / duo topology tracker
# ---------------------------------------------------------------------------

def compute_solo_duo_counts(Nh: int, parent_solo: int, parent_duo: int) -> dict:
    """
    Return solo and duo H counts for a PAH at hydrogen count Nh.

    Assumes the sequential-dehydrogenation model of Andrews (2016):
    H2-losses consume duo pairs first (2 H at a time).  Once duos are
    exhausted, only solo H atoms remain.

    Parameters
    ----------
    Nh          : int   Current total hydrogen count.
    parent_solo : int   Solo  H atoms in the fully-hydrogenated parent.
    parent_duo  : int   Duo   H atoms in the fully-hydrogenated parent.

    Returns
    -------
    dict:
        N_H               current H count (= Nh, echoed for convenience)
        N_solo            solo H atoms remaining
        N_duo             duo  H atoms remaining
        H2_loss_possible  bool — True iff Nh is even and N_duo >= 2
    """
    if Nh <= parent_solo:
        n_solo = Nh
        n_duo  = 0
    else:
        n_duo  = Nh - parent_solo
        n_solo = parent_solo

    return {
        "N_H":             Nh,
        "N_solo":          n_solo,
        "N_duo":           n_duo,
        "H2_loss_possible": (Nh % 2 == 0) and (n_duo >= 2),
    }


# ---------------------------------------------------------------------------
# Parameter lookup
# ---------------------------------------------------------------------------

def get_dissociation_channels(
    Nh: int,
    Nh0: int,
    charge: int,
    parent_solo: int,
    parent_duo: int,
) -> list:
    """
    Return all active dissociation channels and their RRKM parameters for a
    PAH at hydrogen count Nh, normal count Nh0, and ionisation state charge.

    Parameters
    ----------
    Nh          : int   Current hydrogen count.
    Nh0         : int   Normal fully-hydrogenated H count (e.g. 18 for C54H18).
    charge      : int   PAH charge state (< 0, 0, or > 0).
    parent_solo : int   Solo H atoms in the fully-hydrogenated parent.
    parent_duo  : int   Duo  H atoms in the fully-hydrogenated parent.

    Returns
    -------
    List[DissociationChannel] — one entry per active channel.
    """
    channels = []

    if Nh <= Nh0:
        # ------------------------------------------------------------------ #
        # Normal / de-hydrogenated regime
        # ------------------------------------------------------------------ #
        state = compute_solo_duo_counts(Nh, parent_solo, parent_duo)

        # H-loss always possible; parameters depend on parity of N_H
        if Nh % 2 == 0:
            channels.append(DissociationChannel(
                name="H_loss",
                E_act_ev=HLOSS_EVEN_EACT_EV,
                dS_jkm=HLOSS_EVEN_DS_JKM,
            ))
        else:
            channels.append(DissociationChannel(
                name="H_loss",
                E_act_ev=HLOSS_ODD_EACT_EV,
                dS_jkm=HLOSS_ODD_DS_JKM,
            ))

        # H2-loss only when N_H even and duo H atoms remain
        if state["H2_loss_possible"]:
            channels.append(DissociationChannel(
                name="H2_loss",
                E_act_ev=H2LOSS_EACT_EV,
                dS_jkm=H2LOSS_DS_JKM,
            ))

    else:
        # ------------------------------------------------------------------ #
        # Super-hydrogenated regime — H-loss from extra duo-ring H
        # ------------------------------------------------------------------ #
        if charge > 0:
            channels.append(DissociationChannel(
                name="H_loss_superH",
                E_act_ev=SUPERH_HLOSS_CATION_EACT_EV,
                dS_jkm=SUPERH_HLOSS_CATION_DS_JKM,
            ))
        else:
            channels.append(DissociationChannel(
                name="H_loss_superH",
                E_act_ev=SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV,
                dS_jkm=SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM,
            ))

    return channels


def get_dissociation_channels_montillaud(
    Nh: int,
    Nh0: int,
    charge: int,
    parent_solo: int,
    parent_duo: int,
) -> list:
    """
    Return active dissociation channels using Montillaud, Joblin & Toublanc
    (2013) / Berne & Tielens (2012) RRKM parameters.

    Differences from get_dissociation_channels (Andrews 2016):
    - H-loss activation energy is 4.30 eV (vs 4.60/4.10 eV parity split)
    - Includes a C2H2-loss channel (acetylene channel, Allain 1996)
    - C2H2-loss also appears in the super-hydrogenated regime

    Parameters match get_dissociation_channels exactly.

    Returns
    -------
    List[DissociationChannel]
    """
    channels = []

    if Nh <= Nh0:
        # H-loss (charge-independent, no parity split in this parametrisation)
        channels.append(DissociationChannel(
            name="H_loss",
            E_act_ev=M13_HLOSS_DEHYD_EACT_EV,
            dS_jkm=M13_HLOSS_DEHYD_DS_JKM,
        ))

        # H2-loss only when N_H even and duo H atoms remain
        state = compute_solo_duo_counts(Nh, parent_solo, parent_duo)
        if state["H2_loss_possible"]:
            channels.append(DissociationChannel(
                name="H2_loss",
                E_act_ev=M13_H2LOSS_EACT_EV,
                dS_jkm=M13_H2LOSS_DS_JKM,
            ))

        # C2H2-loss (always active in dehydrogenated regime)
        channels.append(DissociationChannel(
            name="C2H2_loss",
            E_act_ev=M13_C2H2LOSS_DEHYD_EACT_EV,
            dS_jkm=M13_C2H2LOSS_DEHYD_DS_JKM,
        ))

    else:
        # Super-hydrogenated: H-loss and C2H2-loss
        if charge > 0:
            channels.append(DissociationChannel(
                name="H_loss_superH",
                E_act_ev=M13_SUPERH_HLOSS_CATION_EACT_EV,
                dS_jkm=M13_SUPERH_HLOSS_CATION_DS_JKM,
            ))
        else:
            channels.append(DissociationChannel(
                name="H_loss_superH",
                E_act_ev=M13_SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV,
                dS_jkm=M13_SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM,
            ))

        channels.append(DissociationChannel(
            name="C2H2_loss",
            E_act_ev=M13_SUPERH_C2H2LOSS_EACT_EV,
            dS_jkm=M13_SUPERH_C2H2LOSS_DS_JKM,
        ))

    return channels


# ---------------------------------------------------------------------------
# Convenience: parameter dict for a single named channel
# ---------------------------------------------------------------------------

def channel_params(name: str) -> tuple:
    """
    Return (E_act_ev, dS_jkm) for a named RRKM channel.

    Andrews (2016) channels:
      'H_loss_even', 'H_loss_odd', 'H2_loss',
      'H_loss_superH_neutral_anion', 'H_loss_superH_cation'

    Montillaud (2013) channels:
      'M13_H_loss_dehyd', 'M13_H2_loss', 'M13_C2H2_loss_dehyd',
      'M13_H_loss_superH_neutral_anion', 'M13_H_loss_superH_cation',
      'M13_C2H2_loss_superH'
    """
    _TABLE = {
        # Andrews 2016
        "H_loss_even":                   (HLOSS_EVEN_EACT_EV,  HLOSS_EVEN_DS_JKM),
        "H_loss_odd":                    (HLOSS_ODD_EACT_EV,   HLOSS_ODD_DS_JKM),
        "H2_loss":                       (H2LOSS_EACT_EV,      H2LOSS_DS_JKM),
        "H_loss_superH_neutral_anion":   (SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV,
                                          SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM),
        "H_loss_superH_cation":          (SUPERH_HLOSS_CATION_EACT_EV,
                                          SUPERH_HLOSS_CATION_DS_JKM),
        # Montillaud 2013
        "M13_H_loss_dehyd":              (M13_HLOSS_DEHYD_EACT_EV,  M13_HLOSS_DEHYD_DS_JKM),
        "M13_H2_loss":                   (M13_H2LOSS_EACT_EV,       M13_H2LOSS_DS_JKM),
        "M13_C2H2_loss_dehyd":           (M13_C2H2LOSS_DEHYD_EACT_EV,
                                          M13_C2H2LOSS_DEHYD_DS_JKM),
        "M13_H_loss_superH_neutral_anion":(M13_SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV,
                                           M13_SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM),
        "M13_H_loss_superH_cation":      (M13_SUPERH_HLOSS_CATION_EACT_EV,
                                          M13_SUPERH_HLOSS_CATION_DS_JKM),
        "M13_C2H2_loss_superH":          (M13_SUPERH_C2H2LOSS_EACT_EV,
                                          M13_SUPERH_C2H2LOSS_DS_JKM),
    }
    if name not in _TABLE:
        raise KeyError(f"Unknown channel '{name}'. Valid: {list(_TABLE)}")
    return _TABLE[name]


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("C54H18 dehydrogenation pathway (H2-loss dominant):")
    print(f"{'NH':>4}  {'N_solo':>6}  {'N_duo':>5}  {'H2_ok':>5}  "
          f"{'Channels'}")
    print("-" * 72)
    for nh in range(18, 0, -1):
        state = compute_solo_duo_counts(nh, parent_solo=6, parent_duo=12)
        chs   = get_dissociation_channels(nh, Nh0=18, charge=0,
                                          parent_solo=6, parent_duo=12)
        ch_str = "  ".join(
            f"{c.name}(E={c.E_act_ev:.2f}eV, dS={c.dS_jkm:+.1f})" for c in chs
        )
        print(f"{nh:>4}  {state['N_solo']:>6}  {state['N_duo']:>5}  "
              f"{'yes' if state['H2_loss_possible'] else 'no':>5}  {ch_str}")

    print()
    print("Super-hydrogenated states (Nh0=18 baseline):")
    for nh_extra, charge in [(19, -1), (19, 0), (19, +1), (20, 0)]:
        chs = get_dissociation_channels(nh_extra, Nh0=18, charge=charge,
                                        parent_solo=6, parent_duo=12)
        ch_str = "  ".join(
            f"{c.name}(E={c.E_act_ev:.2f}eV, dS={c.dS_jkm:+.1f})" for c in chs
        )
        print(f"  NH={nh_extra}, Z={charge:+d}  ->  {ch_str}")
