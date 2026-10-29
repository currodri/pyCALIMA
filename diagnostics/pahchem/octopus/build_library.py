#!/usr/bin/env python3
"""The charge-resolved PAH cross-section library, 0.02 eV to 10 keV, assembled from the finished DIaL3 states.

Per state whose three kicks have reached T = 20 (``plot_dial3_library.states``), in order
(docs/physics/pahchem-uv-absorption.md §4):

1. **TD-LDA** sigma(E) from the kicks (``plot_progress.octopus_sigma``) to 150 eV, with a Gaussian window of
   s0 = 0.15 eV. That is the narrowest width T = 20 gives cleanly (exp(-4.5) is left at the cut), and the warm-vapour
   width of the pyrene calibration. Cold molecules (s ~ 0.08 eV) would need T = 40; on coronene that changes
   pahchem's rates by <= 2.4 % (§4).
2. **Width** s(E):
   - s0 up to 7.5 eV;
   - rising to 0.8 eV at 14 eV, which removes the box artefacts of the discretised continuum. The width added to s0,
     sqrt(s^2 - s0^2), grows linearly from 7.5 eV;
   - flat to 30 eV, then rising to 4 eV at 40 eV, against the box ripples at 30-54 eV (compare_box_euv.py);
   - flat above.
   The width beyond s0 comes from diffusion in energy, d sigma / d tau = d/dE (D d sigma / dE), to tau = 1 with
   D = (s^2 - s0^2) / 2. At constant width that is the Gaussian convolution. It conserves the oscillator strength and
   keeps sigma >= 0. Unlike a convolution whose kernel width follows the source energy, it does not drift strength
   down the width gradient. That drift (w w' sigma) piled strength into the grid point where the width starts, and
   pushed it below 13.6 eV. The profile is empirical and labelled as such (the no-ad-hoc-factors rule).
3. **Positions:** ``--shift D`` (default 0.4) moves the neutrals' bands below 6 eV up by D eV, fading out by 10 eV (ad hoc,
   labelled as such; ions are not shifted), because TD-LDA puts them 0.3-0.65 eV too low (docs §5). ``--shift 0`` keeps TD-LDA's;
   the npz carries both variants.
4. **EUV and X-rays:** the atoms' photoionisation cross sections, Verner phfit2 C 1s + 2s + 2p (``wdb06_atomic``)
   and hydrogenic H 1s, with the valence part normalised to TD-LDA.
   - TD-LDA is kept to 80 eV. Its excess over the atoms at 30-54.4 eV (x1.33) is real: measured benzene shows x1.30
     (Feng+ dipole (e,e), ``--benzene``).
   - Above ~80 eV both TD-LDA (x0.84-0.92) and benzene (x0.78-0.79, flat to 200 eV) lie below the atomic sum, by a
     factor that does not depend on energy. The atomic valence shape is therefore scaled by k, TD-LDA's integral over
     the atoms' at 80-130 eV (0.88-0.89 for every state), and blended in with a cos^2 weight over the same range. The
     result is continuous with Octopus by construction. The unscaled atoms would put 100-290 eV about 13 % above
     TD-LDA and 27 % above benzene.
   - C 1s is added unscaled from Verner's 291 eV step; with ``--ck-edge`` the measured gas-phase C 1s oscillator strength
     per carbon (Gordon+ 2003, ISEELS, pyrene: the 1s -> pi* resonances at 285 eV, the Rydberg and sigma* structure to 296 eV)
     replaces it there, handing over to the atomic value by 310 eV.

Checks per state go to stdout and ``checks.csv``:
- the oscillator strength to 13.6 and 30 eV, final over raw (the §4 guard asks < 0.5 % at 30 eV);
- the most negative sigma over the maximum (guard >= -1e-3);
- f to 54.4 eV, 130 eV and 10 keV, against the valence electrons and the C 1s electrons;
- the valence scale k;
- rates final over raw in three fields: Mathis+83 (0.1-13.6 eV), Draine 1978 (5-13.6 eV) and a Kurucz 40 000 K star
  (13.6-136 eV). Also the mean absorbed Mathis photon energy. Raw here is Octopus's own spectrum (polynomial window,
  T = 20).

Output, in ``--out`` (no figures: the documentation figures come from ``make_library_figures.py``):
- ``pah_sigma_library.npz``: ``energy_ev``, ``label``, ``nc``, ``nh``, ``z``, ``sigma_cm2`` [state, energy] as shipped (neutrals
  shifted by ``shift_ev``), ``sigma_cm2_tdlda`` (TD-LDA positions throughout), ``valence_scale_k`` and ``params``;
- one ``sigma_<state>.dat`` per state (E, sigma as shipped, sigma with TD-LDA positions) and ``checks.csv``.

    build_library.py [--root ~/Documents/RAMSES_dev/pahspec/results/dial3_library] [--out <results>/pahchem/...]
                     [--small-root <megatron runs>] [--shift 0.4] [--ck-edge <gordon03_pyrene.dat>]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np  # noqa: E402
from scipy.linalg import solve_banded  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_dial3_library import ROOT, isrf_stats, sigma_atomic, states  # noqa: E402
from plot_progress import ROOT as MEGATRON_ROOT  # noqa: E402
from plot_progress import octopus_sigma  # noqa: E402

from pycalima import _paths  # noqa: E402
from pycalima.models.dust_charge import wdb06_atomic as W  # noqa: E402
from pycalima.pahchem import fields  # noqa: E402
from pycalima.pahchem import malloci2007 as M  # noqa: E402

S0, S_MID, S_TOP = 0.15, 0.8, 4.0          # eV: the width below 7.5 eV, at 14-30 eV, and above 40 eV
E_JOIN = (80.0, 130.0)                      # eV: the cos^2 hand-over to the scaled atoms, and where k is set
E_TD = np.round(np.arange(0.02, 150.0 + 1e-9, 0.02), 10)
E_1S = W.shell_threshold_eV("C", "1s")
SHIFT_TAPER = (6.0, 10.0)                   # eV: ``--shift`` applies fully below the first value and fades out (cos^2) by the second
K_BLEND_END = 310.0                         # eV: the measured C K edge hands over to the atomic C 1s by here
E_HI = np.unique(np.concatenate([np.geomspace(E_JOIN[1], 1e4, 401)[1:], [E_1S * (1 - 1e-6), E_1S],
                                 np.arange(280.0, 320.0 + 1e-9, 0.05)]))
E_OUT = np.concatenate([E_TD[E_TD <= E_JOIN[1] + 1e-9], E_HI])


#: smaller compact PAHs run on megatron with the same settings (0.25 A, 4 + 1 A box, T = 20), charges -1, 0, +1 only:
#: label -> (N_C, N_H, {charge: run directory under ``--small-root``})
SMALL = {
    "pyrene": (16, 10, {z: f"malloci_pyrene_{z}" for z in (-1, 0, 1)}),
    "coronene": (24, 12, {-1: "coronene_-1_h0.25", 0: "coronene_0_h0.25_T20", 1: "coronene_1_h0.25"}),
    "ovalene": (32, 14, {z: f"malloci_ovalene_{z}" for z in (-1, 0, 1)}),
}


def s_of_e(E):
    """The width s(E) [eV] of the final library (stage 2)."""
    added = np.sqrt(S_MID ** 2 - S0 ** 2) * np.clip((E - 7.5) / 6.5, 0.0, 1.0)
    return np.sqrt(S0 ** 2 + added ** 2) + (S_TOP - S_MID) * np.clip((E - 30.0) / 10.0, 0.0, 1.0)


def widen(E, s, steps=400):
    """``s`` (on the uniform grid ``E``, already at width S0) widened to s(E): conservative diffusion with
    D = (s(E)^2 - S0^2) / 2 to tau = 1, implicit Euler in ``steps`` steps, zero flux at the grid ends. The matrix is
    symmetric with unit row sums, so the integral is kept exactly; it is an M-matrix, so sigma stays >= 0."""
    dE = E[1] - E[0]
    D = 0.5 * np.maximum(s_of_e(E) ** 2 - S0 ** 2, 0.0)
    r = 0.5 * (D[1:] + D[:-1]) / dE ** 2 / steps
    ab = np.zeros((3, len(E)))
    ab[0, 1:], ab[2, :-1], ab[1] = -r, -r, 1.0
    ab[1, :-1] += r
    ab[1, 1:] += r
    out = np.asarray(s, dtype=float).copy()
    for _ in range(steps):
        out = solve_banded((1, 1), ab, out)
    return out


def sigma_atoms(nc, nh, E):
    """Valence (C 2s + 2p, H 1s; ``plot_dial3_library.sigma_atomic``) plus C 1s photoionisation [cm^2]."""
    return sigma_atomic(nc, nh, E) + sigma_1s(nc, E)


def sigma_1s(nc, E, kedge=None):
    """C 1s absorption [cm^2]: Verner's atomic photoionisation, or with ``kedge`` = (E [eV], f [1e-2 per eV per C]) the measured
    gas-phase C 1s oscillator strength spectrum (ISEELS, Gordon+ 2003) per carbon atom up to its last point, then the excess over
    the atomic value at that point fading to zero by ``K_BLEND_END`` (an interpolation: the measured 299-306 eV sigma*(C-C)
    structure is not represented)."""
    atom = nc * W.photoionisation_cross_section("C", "1s", E)
    if kedge is None:
        return atom
    Em, f = kedge
    to_cm2 = nc * 1e-2 * M.MB_EV_PER_F * M.MB_CM2
    e1 = Em[-1]
    end = to_cm2 * max(float(np.mean(f[Em > e1 - 0.3])), 0.0)
    excess = end - nc * W.photoionisation_cross_section("C", "1s", np.array([e1]))[0]
    w = np.cos(0.5 * np.pi * np.clip((E - e1) / (K_BLEND_END - e1), 0.0, 1.0)) ** 2
    meas = to_cm2 * np.clip(np.interp(E, Em, f), 0.0, None)
    return np.where(E < Em[0], 0.0, np.where(E <= e1, meas, atom + excess * w))


def shift_bands(E, s, d):
    """``s`` (on the uniform grid ``E``) moved up by ``d`` eV below SHIFT_TAPER[0], the shift fading out by SHIFT_TAPER[1]: the
    transport x = E + d t(E), sigma'(x) = sigma(E) / (dx/dE). It conserves f exactly and keeps sigma >= 0; between the two
    energies the spectrum is stretched by up to d x (pi/2) / (taper width)."""
    if d == 0.0:
        return s
    lo, hi = SHIFT_TAPER
    u = np.clip((E - lo) / (hi - lo), 0.0, 1.0)
    t = np.cos(0.5 * np.pi * u) ** 2
    dt = -0.5 * np.pi / (hi - lo) * np.sin(np.pi * u) * ((E > lo) & (E < hi))
    return np.interp(E, E + d * t, s / (1.0 + d * dt))


def join_weight(E):
    """TD-LDA's weight in the hand-over: 1 below E_JOIN[0], cos^2 down to 0 at E_JOIN[1]."""
    x = np.clip((np.asarray(E) - E_JOIN[0]) / (E_JOIN[1] - E_JOIN[0]), 0.0, 1.0)
    return np.cos(0.5 * np.pi * x) ** 2


def build(run, nc, nh, kedge=None, shift=0.0):
    """Stages 1-4 of one state: {stage: (E, sigma [cm^2])}, and the valence scale k."""
    _, raw, _ = octopus_sigma(run, energies=E_TD)
    _, gauss, _ = octopus_sigma(run, damp=lambda t, T: np.exp(-0.5 * (S0 * t) ** 2), energies=E_TD)
    wide = shift_bands(E_TD, widen(E_TD, gauss), shift)
    k = (E_TD >= E_JOIN[0]) & (E_TD <= E_JOIN[1])
    scale = np.trapezoid(wide[k], E_TD[k]) / np.trapezoid(sigma_atomic(nc, nh, E_TD[k]), E_TD[k])
    low = E_OUT <= E_JOIN[1] + 1e-9
    wt = np.where(low, join_weight(E_OUT), 0.0)
    td = np.interp(E_OUT, E_TD, wide, right=0.0)
    scaled = scale * sigma_atomic(nc, nh, E_OUT) + sigma_1s(nc, E_OUT, kedge)
    final = wt * td + (1 - wt) * scaled
    return {"raw": (E_TD, raw), "gauss": (E_TD, gauss), "wide": (E_TD, wide),
            "atoms": (E_OUT, sigma_atoms(nc, nh, E_OUT)), "scaled": (E_OUT, scaled), "final": (E_OUT, final)}, scale


def rate(field, E, s):
    """Absorbed photons [s^-1] of ``field`` over its own grid, negative ringing clipped."""
    e = field.energy_ev
    return np.trapezoid(np.clip(np.interp(e, E, s, left=0.0, right=0.0), 0.0, None) * field.flux_at(e), e)


def checks(label, nc, nh, z, st, scale):
    E, s = st["final"]
    Er, r = st["raw"]
    lo = E <= E_JOIN[1]
    mathis = fields.mathis1983(e_min=0.1, n=2000)
    d78 = fields.draine1978()
    star = fields.kurucz(40000, 1.0, e_max=136.0, e_min=13.6)
    n_val = M.valence_electrons(nc, nh, z)
    return {
        "state": f"{label}_{z:+d}", "nc": nc, "nh": nh, "z": z,
        "f13.6_final_over_raw": M.oscillator_strength(E, s, 0, 13.6) / M.oscillator_strength(Er, r, 0, 13.6),
        "f30_final_over_raw": M.oscillator_strength(E, s, 0, 30) / M.oscillator_strength(Er, r, 0, 30),
        "min_over_max": s[lo].min() / s[lo].max(),
        "f54_over_valence": M.oscillator_strength(E, s, 0, 54.4) / n_val,
        "f130_over_valence": M.oscillator_strength(E, s, 0, 130) / n_val,
        "f10keV_over_valence_plus_1s": M.oscillator_strength(E, s) / (n_val + 2 * nc),
        "valence_scale_k": scale,
        "mathis_final_over_raw": rate(mathis, E, s) / rate(mathis, Er, r),
        "draine78_final_over_raw": rate(d78, E, s) / rate(d78, Er, r),
        "kurucz40k_final_over_raw": rate(star, E, s) / rate(star, Er, r),
        "mathis_mean_photon_ev": isrf_stats(E, s)[0],
    }


def write(out, rows, rows0, table, shift, kedge):
    """``rows`` hold the library as shipped (neutrals shifted by ``shift`` eV, if any); ``rows0`` the same with TD-LDA positions."""
    note = (f"s0 {S0} eV to 7.5 eV, {S_MID} eV at 14-30 eV, {S_TOP} eV above 40 eV; hand-over cos^2 {E_JOIN[0]}-{E_JOIN[1]} eV to the "
            f"atoms (valence x k); C 1s: " + ("measured gas-phase pyrene ISEELS (Gordon+ 2003) to 296 eV, atomic from 310 eV"
                                              if kedge is not None else "atomic (Verner) from 291 eV")
            + f"; neutrals' bands below 6 eV shifted up by {shift} eV (ad hoc, tapering out by 10 eV), ions TD-LDA's; "
            f"sigma_cm2_tdlda has TD-LDA positions throughout")
    np.savez_compressed(
        out / "pah_sigma_library.npz", energy_ev=E_OUT, label=np.array([r[0] for r in rows]),
        nc=np.array([r[1] for r in rows]), nh=np.array([r[2] for r in rows]), z=np.array([r[3] for r in rows]),
        sigma_cm2=np.array([r[4]["final"][1] for r in rows]), sigma_cm2_tdlda=np.array([r[4]["final"][1] for r in rows0]),
        valence_scale_k=np.array([r[5] for r in rows]), shift_ev=shift, params=np.array(note))
    for (label, nc, nh, z, st, k), (*_, st0, _) in zip(rows, rows0):
        E, s = st["final"]
        np.savetxt(out / f"sigma_{label}_{z:+d}.dat", np.column_stack([E, s, st0["final"][1]]), fmt="%.6e",
                   header=f"{label} charge {z:+d}: N_C {nc}, N_H {nh}. Octopus TD-LDA (DIaL3, T = 20); build_library.py. {note}; "
                          f"valence scale k = {k:.3f}\nE [eV]  sigma [cm^2] (as shipped)  sigma [cm^2] (TD-LDA positions)")
    with open(out / "checks.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)


COMPACT = ("pyrene", "coronene", "ovalene", "C54H18", "C96H24", "C150H30", "C216H36")   # the circumcoronene family
ANCHOR_NC = (16, 24, 32, 54, 96, 150, 216)
ANCHOR_CHARGES = (-1, 0, 1, 2)


def anchors(rows, rows0):
    """Per-carbon spectra [cm^2 per C] for ``pycalima.pahchem.uvlib``: [positions (shifted, TD-LDA), structure (compact, mixed), size,
    charge, energy]. ``compact`` is the circumcoronene family; ``mixed`` the equal-weight mean over every library molecule of that size
    (the same as compact below C54, where only compact molecules were run). A charge missing at a size takes the next lower one (the
    dication of the small molecules takes the cation's: ``proxy`` marks them)."""
    out = np.zeros((2, 2, len(ANCHOR_NC), len(ANCHOR_CHARGES), len(E_OUT)), dtype=np.float32)
    proxy = np.zeros((len(ANCHOR_NC), len(ANCHOR_CHARGES)), dtype=bool)
    for ip, rr in enumerate((rows, rows0)):
        for isz, n in enumerate(ANCHOR_NC):
            for iz, q in enumerate(ANCHOR_CHARGES):
                for ist, structure in enumerate(("compact", "mixed")):
                    pick = [r for r in rr if r[1] == n and r[3] == q and (structure == "mixed" or r[0] in COMPACT)]
                    if not pick:
                        out[ip, ist, isz, iz] = out[ip, ist, isz, iz - 1]
                        proxy[isz, iz] = True
                        continue
                    out[ip, ist, isz, iz] = np.mean([r[4]["final"][1] / r[1] for r in pick], axis=0)
    return out, proxy


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="local mirror of DIaL3's pahspec/library")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/pahchem/octopus_library")
    ap.add_argument("--small-root", type=Path, default=MEGATRON_ROOT,
                    help="local mirror of megatron's ~/pahspec runs: adds pyrene, coronene and ovalene (C16, C24, C32; -1, 0, +1); "
                         "give an empty path to leave them out")
    ap.add_argument("--anchors", type=Path, default=None,
                    help="also write the per-carbon anchor spectra for pycalima.pahchem.uvlib (pah_uv_library.npz)")
    ap.add_argument("--shift", type=float, default=0.4,
                    help="ad hoc uniform upward shift [eV] of the NEUTRALS' bands below 6 eV (tapering out by 10 eV); 0 keeps "
                         "TD-LDA's positions. Default 0.4 (docs/physics/pahchem-uv-absorption.md §5)")
    ap.add_argument("--ck-edge", type=Path, default=None,
                    help="measured gas-phase C 1s oscillator strength per carbon (E [eV], f [1e-2 per eV]), e.g. the pyrene curve "
                         "digitised from Gordon+ 2003; default: the atomic step at 291 eV")
    a = ap.parse_args()
    out = a.out or _paths.get_results_dir("pahchem/octopus_library")
    out.mkdir(parents=True, exist_ok=True)
    kedge = tuple(np.loadtxt(a.ck_edge, unpack=True)) if a.ck_edge else None
    found = [(label, nc, nh, z, a.root / f"{label}_{z:+d}") for label, nc, nh, z, *_, done in states(a.root, 20.0 - 1e-6) if done]
    if a.small_root:
        found += [(label, nc, nh, z, a.small_root / d) for label, (nc, nh, runs) in SMALL.items() for z, d in runs.items()]
    rows0 = [(label, nc, nh, z, *build(run, nc, nh, kedge)) for label, nc, nh, z, run in found]
    runs = {(label, z): run for label, nc, nh, z, run in found}
    rows = [r if (r[3] != 0 or a.shift == 0.0) else (*r[:3], 0, *build(runs[(r[0], 0)], r[1], r[2], kedge, a.shift))
            for r in rows0]
    table = [checks(*r) for r in rows]
    write(out, rows, rows0, table, a.shift, kedge)
    if a.anchors:
        sig, proxy = anchors(rows, rows0)
        a.anchors.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.anchors, energy_ev=E_OUT, nc=np.array(ANCHOR_NC), charges=np.array(ANCHOR_CHARGES), sigma_per_c_cm2=sig,
                            proxy=proxy, shift_ev=a.shift, k_edge=kedge is not None,
                            axes=np.array("positions (shifted, tdlda), structure (compact, mixed), N_C, charge, energy"))
    keys = list(table[0])[4:]
    print(f"{len(rows)} finished states; columns: " + ", ".join(keys))
    for t in table:
        print(f"  {t['state']:18s} " + " ".join(f"{t[k]:7.4f}" if abs(t[k]) < 100 else f"{t[k]:7.1f}" for k in keys))
    print(f"library in {out}")


if __name__ == "__main__":
    main()
