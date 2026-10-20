"""The PAH hydrogenation network of Murga, Kirsanova, Wiebe & Boley (2022), model b.

Method card ``method_cards/murga2022.toml``. Model b (Table 4) has H loss and
H addition only: per location, per charge Z and per size bin, the population
moves between five hydrogenation bins, X_H/X_H0 = 0.26, 0.63, 1.0, 1.37 and
1.74. It starts from the normal bin and runs in a fixed environment. The
charge is fixed per population and weighted by f(a, Z) afterwards. There is
no loss from the lowest bin and no addition into the top one (Sect. 3.1).

The pieces:

- **Photo H loss.** ``pahchem.shiva`` (M20 eqs. 4-6, the post-absorption
  yield) in the MMP83 field scaled by chi (the unit the charge gate infers,
  ``murga_charge``):

  - for X_H <= X_H0: H (4.3 eV, 11.8 cal/K), H2 (3.52 eV, -12.69 cal/K) and
    the H of C2H2 (4.6 eV, 10 cal/K), counted in atoms, each channel against
    k_IR alone;
  - for X_H > X_H0: H and C2H2 with E0 = a* (X_H/X_H0) + b* by charge class,
    13.3 and 10 cal/K (Table 3), plus Eley-Rideal abstraction 8.7e-13
    sqrt(T/100) n_H (M20 eq. 10).
  - The C2H2 hydrogen: M22 Sect. 3.1, "R_H is the net rate of detachment of
    all hydrogen atoms which might be detached from a PAH both separately
    and also inside hydrogen or acetylene molecules"; model b keeps the
    carbon.
- **H addition (M20 Sect. 2.2).**

  - Neutrals: 0.5 (N_edge k_edge + N_core k_core) n_H, from Goumans 2011
    (``goumans2011_table3.csv``), N_edge = N_H0, N_core = N_C - N_H0.
  - Ions: 1.4e-10 N_H/N_C n_H (normal); 0.5 (1.4e-10 N_H/N_C + 5e-11) n_H
    (dehydrogenated); 1e-12 n_H (super-H). H2 does not react (M22 Sect.
    4.3.4).

Choices where the card says "not stated", fixed or tested on the Figshare
files (``check_murga2022.py``):

- **Molecule of a size bin.** N_C = 468 (a/10 A)^3 (the DL01 radius law; it
  gives the quoted N_C = 60 at 5 A) and N_H0 = DL01 eq. 8.

  - A bin state has N_H = max(1, round(x N_H0)).
  - The 50 A bin (N_C ~ 58500) has no photo-loss: E/N_C < 3e-4 eV per C.
- **Charge classes.** Rates by anion (Z < 0), neutral and cation (Z > 0).
  The Figshare chains are identical for every Z >= 1, and for Z = -1 and
  -2.
- **Bin flux.** A per-molecule rate R [atoms s^-1] moves the population by
  one bin at R / dN_H, dN_H = 0.371 N_H0 (the spacing of the bin centres).
  The same holds for losses and additions, so equilibrium distributions do
  not depend on it. The deep-MC chains support it: with it, the ion
  loss/addition balance and the neutral step rate give the same factor
  between their rates and ours.
- **Output times.** 0, 1e4, ..., 1e5 yr. The deep-MC neutral chains decay
  by a constant factor per output (linear spacing), and 1e5 yr is the age
  (Sect. 4.1).
- **T(E).** DL01 (``shiva.ShivaMolecule(temperature="dl01")``). The
  deep-MC chains reproduce the N_H dependence of the rate with DL01 and
  invert it with Tielens' relation.
- **Goumans temperature range.** The fits are clamped below 40 K (they
  diverge there, Goumans Sect. 3) and extrapolated above 2000 K.

Measured against the Figshare model b (2026-10-06, check_murga2022.py):

- **Deep MC, 3.3 A neutrals** (locations 134-151; H addition negligible
  there, so these are pure loss chains, fitted exactly):

  - The step rate k(mid -> lowest)/k(normal -> mid) is 1.250 at every chi.
    With DL01 our R(mid)/R(normal) is 1.257 (1.278 without the C2H2
    hydrogen); with Tielens it is 0.787 (the T(E) verdict).
  - The absolute step rate is 0.87-1.44x ours (DL01; 1.5-2.5x without the
    C2H2 hydrogen, 2.8-4.6x with the channels competing as in M19 eq. B2).
    The ratio rises monotonically with depth (chi 1.9e-3 -> 1.2e-5): their
    rate per unit chi grows inward, as a field shape that changes with depth
    (not stated) would make it.
- **3.3 A at 1e5 yr** (f-weighted fraction in the lowest bin, ours /
  theirs):

  - PDR 1: 0.98 / 0.95;
  - PDR 2: 0.92 / 0.93;
  - DF: 1.00 / 1.00;
  - MC: 0.85 / 0.86.
- **>= 6 A outside the HII region.** Ours go super-hydrogenated (60-90 % in
  the top two bins; <X_H/X_H0> +0.3 to +0.5), where Figshare stays normal
  (80-100 %).

  - The M20 H-addition rates as implemented (Goumans for neutrals, 1.4e-10
    N_H/N_C for normal cations) outrun the super-H photo-loss by 10^3-10^4.
  - The Figshare super-H populations do not behave as addition chains: the
    3 -> 4 step is ~2e-13 s^-1 whatever n_H.
  - So M22's addition (or super-H loss) differs from M20 as printed, in a
    way the files do not reveal. M22's one stated change there, the Table 3
    super-H law, makes it worse, not better: at X_H/X_H0 = 1.37 it puts the
    neutral H-loss E0 at 3.3 eV, against M20's 1.4 eV. The C2H2 hydrogen
    changes <X_H/X_H0> by < 0.01.
- **Fig. 8 <X_H/X_H0>(distance)**: +0.09 (MC, PDR 1) to +0.18 (PDR 2)
  above the Figshare values, from the same super-H excess.
- **HII region.** 6-7.8 A are dehydrogenated in the files and normal here:
  M22 add EUV photons there with no stated spectrum.

"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import shiva

YR = 3.15576e7
X_BINS = np.array([0.258, 0.629, 1.0, 1.371, 1.742])          # X_H / X_H0 (Figshare x_h / 1.385)
DX_CENTRES = 0.371                                              # bin-centre spacing in X_H / X_H0
OUTPUT_TIMES_YR = np.arange(11) * 1.0e4
CHI_GRID = np.array([1e-4, 1e-2, 1.0, 10.0, 100.0, 1e3, 3e3, 1e4, 2e4, 4e4])
CLASSES = (-1, 0, 1)                                            # anion, neutral, cation
#: Table 3 super-H H loss: E0 = a* x + b*, dS = 13.3 cal/K, by charge class
SUPERH = {-1: (-2.5, 6.8), 0: (-2.7, 7.0), 1: (-2.4, 6.7)}
DS_SUPERH = 13.3
#: Table 3 super-H C2H2 loss: E0 = a* x + b*, dS = 10.0 cal/K, by charge class
SUPERH_C2H2 = {-1: (-2.7, 7.3), 0: (-2.9, 7.5), 1: (-2.6, 7.2)}
H_LOSS, H2_LOSS, C2H2_LOSS = (4.3, 11.8), (3.52, -12.69), (4.6, 10.0)
K_ION_NORMAL, K_ION_DEHYDRO_CONST, K_ION_SUPERH = 1.4e-10, 5e-11, 1e-12
K_ER = 8.7e-13
NO_PHOTO_LOSS_NC = 20000


def charge_class(z):
    return int(np.sign(z))


def nc_of(a_cm):
    """N_C of a size bin: the DL01 radius law inverted, 468 (a / 10 A)^3."""
    return max(int(round(468.0 * (a_cm / 1e-7) ** 3)), 6)


def nh_of(nc, j):
    return max(1, int(round(X_BINS[j] * shiva.dl01_nh(nc))))


# ---------------------------------------------------------------------------
# H addition
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _goumans():
    t = pd.read_csv(_paths.get_external_data_path("pahchem/goumans2011_table3.csv"), comment="#")
    return t[t.isotope == "H"]


def goumans_k(site, temperature_k):
    """k [cm^3 site^-1 s^-1] of Goumans 2011 eq. 1 / Table 3; clamped below 40 K."""
    t = _goumans()
    r = t[t.site.astype(str) == site]
    T = max(float(temperature_k), 40.0)
    row = r[r.range == ("high" if T > r.t_c_K.iloc[0] else "low")].iloc[0]
    return row.alpha_cm3_site_s * (T / 300.0) ** row.beta * np.exp(-row.gamma_K / T)


def h_addition(nc, j, z, temperature_k, n_h):
    """H additions per molecule per second in bin j (M20 Sect. 2.2)."""
    nh0, nh = shiva.dl01_nh(nc), nh_of(nc, j)
    if charge_class(z) == 0:
        k_edge = 0.5 * (goumans_k("1", temperature_k) + goumans_k("4", temperature_k))
        return 0.5 * (nh0 * k_edge + (nc - nh0) * goumans_k("3a1", temperature_k)) * n_h
    if X_BINS[j] < 1.0:
        return 0.5 * (K_ION_NORMAL * nh / nc + K_ION_DEHYDRO_CONST) * n_h
    if X_BINS[j] == 1.0:
        return K_ION_NORMAL * nh / nc * n_h
    return K_ION_SUPERH * n_h


# ---------------------------------------------------------------------------
# Photo H loss: tabulated on CHI_GRID per (size, bin, charge class)
# ---------------------------------------------------------------------------

def photo_loss_on_grid(nc, j, zc, temperature="dl01", chi_grid=CHI_GRID, acetylene_h=True):
    """H atoms lost per molecule per second [s^-1] on ``chi_grid`` (M20 eqs. 4-6).

    ``acetylene_h``: R_H counts the two H of every C2H2 loss (M22 Sect. 3.1:
    "R_H is the net rate of detachment of all hydrogen atoms which might be
    detached from a PAH both separately and also inside hydrogen or
    acetylene molecules"); model b keeps the carbon (B = 0). False counts H
    and H2 only.
    """
    if nc > NO_PHOTO_LOSS_NC:
        return np.zeros(len(chi_grid))
    m = shiva.ShivaMolecule(nc, nh=nh_of(nc, j), charge=zc, temperature=temperature)
    x = X_BINS[j]
    k = lambda e0, ds: shiva.k_arrhenius_te(m.energy_ev, e0, ds, m.t_m)
    if x > 1.0:
        (a, b), (ac, bc) = SUPERH[zc], SUPERH_C2H2[zc]
        ks = {"H": k(a * x + b, DS_SUPERH), "C2H2": k(ac * x + bc, C2H2_LOSS[1])}
    else:
        ks = {"H": k(*H_LOSS), "H2": k(*H2_LOSS), "C2H2": k(*C2H2_LOSS)}
    atoms = {"H": 1.0, "H2": 2.0, "C2H2": 2.0 if acetylene_h else 0.0}
    out = []
    for chi in chi_grid:
        r = m.rates(shiva.mmp83(chi), ks)
        out.append(sum(atoms[c] * r[c]["m20"] for c in ks))
    return np.array(out)


def _cache_path(temperature):
    """The on-disk table, keyed on the source of shiva.py and this module: any
    edit to the code that computes it invalidates the cache."""
    import hashlib
    from pathlib import Path

    src = Path(shiva.__file__).read_bytes() + Path(__file__).read_bytes()
    key = hashlib.sha1(src).hexdigest()[:12]
    d = _paths.get_dataset_cache_dir() / "pahchem" / "murga2022"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"pahchem_photo_loss_{temperature}_{key}.npz"


def photo_loss_table(a_cm, temperature="dl01", cache=True, progress=None):
    """R[size, bin, class, chi] [H atoms s^-1] for the size bins ``a_cm``; cached."""
    path = _cache_path(temperature)
    if cache and path.exists():
        with np.load(path) as d:
            if np.allclose(d["a_cm"], a_cm) and np.allclose(d["chi"], CHI_GRID):
                return d["R"]
    R = np.zeros((len(a_cm), X_BINS.size, len(CLASSES), CHI_GRID.size))
    for m, a in enumerate(a_cm):
        for j in range(X_BINS.size):
            for c, zc in enumerate(CLASSES):
                R[m, j, c] = photo_loss_on_grid(nc_of(a), j, zc, temperature)
        if progress:
            progress(f"  size {a * 1e8:.1f} A done")
    if cache:
        np.savez(path, R=R, a_cm=np.asarray(a_cm), chi=CHI_GRID)
    return R


def photo_loss_at(R_grid, chi):
    """log-log interpolation in chi; linear (single-photon) scaling below the grid."""
    lc = np.log(CHI_GRID)
    lr = np.log(np.maximum(R_grid, 1e-300))
    if chi <= CHI_GRID[0]:
        return R_grid[..., 0] * chi / CHI_GRID[0]
    i = np.clip(np.searchsorted(lc, np.log(chi)) - 1, 0, lc.size - 2)
    w = (np.log(chi) - lc[i]) / (lc[i + 1] - lc[i])
    return np.exp(lr[..., i] * (1 - w) + lr[..., i + 1] * w)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

def generator(loss, add, nh0, dx=DX_CENTRES):
    """5x5 rate matrix from per-bin loss and addition rates [atoms s^-1]."""
    step = dx * nh0
    G = np.zeros((X_BINS.size, X_BINS.size))
    for j in range(X_BINS.size):
        if j > 0:
            G[j - 1, j] += loss[j] / step
        if j < X_BINS.size - 1:
            G[j + 1, j] += add[j] / step
    G -= np.diag(G.sum(axis=0))
    return G


def evolve(G, times_yr=OUTPUT_TIMES_YR, start=2):
    """Bin fractions at ``times_yr`` from the normal bin (eigendecomposition)."""
    w, V = np.linalg.eig(G)
    c = np.linalg.solve(V, np.eye(X_BINS.size)[start])
    t = np.asarray(times_yr) * YR
    out = (V[None, :, :] * (c[None, None, :] * np.exp(np.outer(t, w))[:, None, :])).sum(axis=2)
    return np.clip(out.real, 0.0, None)


def run_model_b(data, profile, R, temperature_k=None, loss_scale=1.0, dx=DX_CENTRES, times_yr=OUTPUT_TIMES_YR):
    """Bin fractions [loc, Z, t, bin, size] of model b at every Figshare location.

    ``data``: ``murga_data.read_model_2022``; ``profile``: the
    murga2022_fig6_profile table; ``R``: ``photo_loss_table``.
    ``loss_scale`` multiplies the photo loss (a sensitivity knob, not a fit).
    """
    a_cm, zs = data["a_cm"], data["z"]
    out = np.zeros((len(profile), zs.size, len(times_yr), X_BINS.size, a_cm.size))
    for loc in range(len(profile)):
        chi, T, n_h = data["chi"][loc], data["t_gas"][loc], data["n_h"][loc]
        Rl = photo_loss_at(R, chi) * loss_scale                       # size, bin, class
        for iz, z in enumerate(zs):
            c = CLASSES.index(charge_class(z))
            for m, a in enumerate(a_cm):
                nc = nc_of(a)
                loss = Rl[m, :, c].copy()
                loss[3:] += K_ER * np.sqrt(T / 100.0) * n_h
                add = np.array([h_addition(nc, j, z, T, n_h) for j in range(X_BINS.size)])
                out[loc, iz, :, :, m] = evolve(generator(loss, add, shiva.dl01_nh(nc), dx), times_yr)
    return out
