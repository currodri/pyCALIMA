#!/usr/bin/env python3
"""The charge ladder in EUV fields (gas-coupling section 2.2c, stage 2 step 4): pahchem against WDB06 grains of the same carbon.

For C24H12, C54H18, C96H24, C150H30 and C216H36 (as ``check_charge_ladder``: PAHdb v4.00, the calibrated IPs, the defaults) in the
Kurucz 40 kK field at G0 = 1e4 Habing, cut at 13.6 eV (neutral gas) and extended to 100 eV (ionised gas; the star emits next to
nothing above the He II edge at 54 eV, so 100 eV holds its EUV), T = 1e4 K, n(H+) = n_e, atomic H 1e-3 n_e (only the hydrogenation
sees it), psi = G0 sqrt(T) / n_e = 1e2-1e6:
- pahchem: <Z>, its spread, the highest charge holding >= 1e-3 of the molecules, Gamma_PE, the share of the photoionisations made by
  photons above 13.6 eV, the Coulomb-explosion rate (``chain_overflow``) and the IR share of stand-in states;
- WDB06 (pyCALIMA's ``yield_model="WDB06"``: band, inner-shell, Auger and secondary yields, Draine 2003 absorption to 20 keV) for
  one graphite sphere of the same carbon, a = 10 A (N_C/468)^(1/3), in the same field (the same photons per G0), with H+ capture;
- the photon side alone, per charge: every state's photoionisation rate at G0 = 1 in the 100 eV field, pahchem's k_up(Z) against
  WDB06's R_pe(Z), and the share of each from above 13.6 eV. The charge balance also carries the electron sticking, where the two
  models differ most (section 2.2b, ``check_wd01_ingredients``); this comparison does not;
- the photon swap: pahchem's chain with WDB06's photoionisation rates for Z >= 0 (its detachment, ion rates and capture kept; on
  pahchem's own rates it reproduces ``charge_matrix`` exactly).
Nothing is fitted.

Result (2026-11-03):
- At 100 eV every state to the cold limit is live. EUV photons raise <Z> in both models: at psi = 1e6 pahchem C54 2.4 -> 6.9,
  C216 4.7 -> 12.7; WDB06 2.3 -> 5.7, 4.0 -> 9.6. They make 43-57 % of the photoionisations at psi = 1e2 (C24 70 %), all of them at
  psi >= 3e5. For N_C >= 54 pahchem stays below WDB06 up to psi ~ 1e5 (the cation sticking, as in the 13.6 eV field) and rises
  above it beyond. C24, with few anions in pahchem (attachment f ~ 0.003), is above WDB06 at psi <= 1e3 and is held at its cold
  limit (+4) at psi = 1e6.
- Per state, pahchem's photoionisation rate is 1.2-1.9x WDB06's for Z <= +1 (absorption, the library against graphite), 1.6-9x at
  +2..+4 for N_C >= 54 (C24 3.7-25x), 4-35x at +5 and +6, and 50 up to 3e7x from +10 up. Two causes:
  - thresholds: pahchem's IP(Z) climbs more slowly, the DFT step continued (1.54 eV per charge for C216) against the sphere's
    (e^2/a)(1 + 0.3 A/a) (1.94 eV): 0.6-2 eV lower at +3..+5, 3-8.5 eV at +10..+20;
  - yields, the larger: pahchem's mean yield above threshold is 0.39-0.74 while IP < 20 eV and 1 above (``Y_UNITY_EV``); WDB06's
    falls from 0.7 at Z = 0 to 0.08-0.32 at +3..+5 and <= 0.07 from +10 up (the electron's escape over (Z+1) e^2/a, its y2).
- With WDB06's photoionisation rates pahchem's <Z> at 100 eV, psi = 1e6, falls to 4.1 (C54) and 7.1 (C216), below WDB06 at every
  psi for N_C >= 54: its charge above WDB06's comes from the photo side alone, and outweighs its higher cation sticking.
- Gamma_PE pahchem/WDB06 at 100 eV: 0.9-1.9 at psi <= 1e3, 5-13 at psi >= 1e4 (C24 3.5-21).
- Stand-ins emit nearly all the IR at psi >= 1e5 for N_C >= 150, and >= 92 % at psi >= 3e5 for all (where the molecule is not
  stripped below its emitting states: nan); 24-86 % at psi = 3e4 for N_C >= 96.
- Coulomb explosions: C24 reaches its cold limit (+4) and explodes at 1.6e-8 s^-1 (psi = 1e4) to 1.2e-5 s^-1 (1e6) at G0 = 1e4;
  C54 at 2e-9 s^-1 (1e6); the larger ones never.
So above +3 pahchem's charge in EUV fields rests on unit yields above max(20 eV, IP) and the continued DFT step, neither measured;
WDB06's bulk-grain yield is not measured for molecules either.

    check_charge_ladder_euv.py [--workers 4] [--out DIR]
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import check_wd01_per_size as WS
import fit_pdrs4all_population as FP

from pycalima import _paths
from pycalima.models.dust_charge import dust_charging as DC
from pycalima.models.dust_charge import wdb06_yields
from pycalima.pahchem import fields
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment
from pycalima.plotting_style import use_calima_style

TEFF = 40000
G0 = 1e4
T = 1e4
PSI = np.logspace(2.0, 6.0, 9)
E_MAX = (13.6, 100.0)
M_H = 1.67262192e-24
ZS_PHOTO = np.arange(-1, 44)              # the photon side: -1 to C216's cold limit
F_MIN = 1e-3                              # "reached": the highest charge holding this share


def env(psi, e_max):
    ne = G0 * np.sqrt(T) / psi
    return Environment("euv", G0, 1e-3 * ne, ne, T, TEFF, field_e_max_ev=e_max, ions=(("H+", ne),))


def rad_field(e_max):
    """pahchem's Kurucz field at G0 = 1 as pyCALIMA's [E_eV, wavelength_nm, I_E (erg s^-1 cm^-2 eV^-1)], I_E = E F(E)."""
    f = fields.kurucz(TEFF, 1.0, e_max=e_max)
    return np.column_stack([f.energy_ev, fields.HC_EV_NM / f.energy_ev, f.energy_ev * f.photon_flux * fields.EV_ERG])


def wdb06_inputs(a, e_max):
    """(nu [Hz], J_nu [photons s^-1 cm^-2 Hz^-1], C_abs [cm^2], yield parameters) at G0 = 1: what ``equilibrium_charge_for_grain``
    gives its WDB06 branch."""
    from pycalima.models.dust_charge.dust_photoelectric_heating import read_dielectric_file

    rad = rad_field(e_max)
    E, wav_cm = rad[:, 0], rad[:, 1] * 1e-7
    nu = DC.c_cgs / wav_cm
    J_nu = rad[:, 2] / (nu * DC.eV2erg)                              # photons s^-1 cm^-2 Hz^-1
    C_abs = np.pi * a**2 * wdb06_yields.absorption_efficiency(E, a, "graphite")
    perp = read_dielectric_file(f"{DC.PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10")
    par = read_dielectric_file(f"{DC.PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10")
    wav_um = wav_cm * 1e4
    im_perp = np.interp(wav_um, perp["table"]["wavelength_um"][::-1], perp["table"]["Im_n"][::-1])
    im_par = np.interp(wav_um, par["table"]["wavelength_um"][::-1], par["table"]["Im_n"][::-1])
    yp = {"W": DC.graphite_work_function, "le": DC.electron_escape_length, "Imperp": im_perp, "Impar": im_par, "wav": wav_cm,
          "la": DC.photon_attenuation_length_graphite_vec(wav_cm, im_perp, im_par), "E_eV": E, "material": "graphite",
          "yield_model": "WDB06"}
    return nu, J_nu, C_abs, yp


def wdb06_photo_rates(a, e_max):
    """WDB06's R_pe(Z) [s^-1] on ZS_PHOTO at G0 = 1."""
    nu, J_nu, C_abs, yp = wdb06_inputs(a, e_max)
    return DC.compute_Rpe_vectorized(nu, J_nu, C_abs, a, ZS_PHOTO, wdb06_yields.yield_wdb06_vectorized, yp)


def thresholds_and_yields(pop, mol, z):
    """(IP pahchem, IP WDB06 [eV], mean yield above each IP, pahchem's and WDB06's) of charge z in the 100 eV field."""
    from pycalima.pahchem import photo

    f = fields.kurucz(TEFF, 1.0, e_max=100.0)
    E, F = f.energy_ev, f.photon_flux
    a = float(WS.W.radius_cm(mol.nc))
    ip_p = float(P.state_threshold(mol, z))
    sig = photo.sigma_uv(mol.nc, z, E, pop.p.uv_cross_section) * pop.p.sigma_scale
    y_p = P._yield(mol, z, pop.p, energy_ev=E)
    ip_w = float(wdb06_yields.ionisation_potential(DC.graphite_work_function, z, a))
    nu, _, C, yp = wdb06_inputs(a, 100.0)
    y_w = wdb06_yields.yield_wdb06_vectorized(nu, np.array([z]), a, yp)[:, 0]
    mean = lambda s, y, ip: np.trapezoid(s * F * y, E) / np.trapezoid(np.where(E >= ip, s * F, 0.0), E)   # noqa: E731
    return ip_p, ip_w, mean(sig, y_p, ip_p), mean(C, y_w, ip_w)


def wdb06_job(job):
    """(<Z>, sigma_Z, highest charge holding F_MIN, Gamma [erg s^-1]) of one WDB06 grain."""
    nc, psi, e_max = job
    a = float(WS.W.radius_cm(nc))
    ne = G0 * np.sqrt(T) / psi
    Zs, Pz, rates, zbar, zsig = DC.equilibrium_charge_for_grain(G0, ne, T, "graphite", a, rad_field=rad_field(e_max),
                                                                yield_params={"yield_model": "WDB06"},
                                                                ion_species=[{"n": ne, "T": T, "m": M_H, "z": 1.0}])
    return zbar, zsig, int(Zs[Pz >= F_MIN].max()), rates["Gamma_total"]


def charge_with_photo(pop, e, k_photo):
    """f[s, slot] of ``Population.charge_matrix`` with the photoionisation rates of Z >= 0 at G0 = 1 replaced by
    ``k_photo[s, slot]`` (pahchem's detachment, ion rates and capture kept): with pahchem's own k_up it is charge_matrix."""
    zs = np.array(pop.zs)
    k_up = pop._stack(P.field_shape(e))[0].copy()
    k_up[:, zs >= 0] = k_photo[:, zs >= 0]
    up = e.g0 * k_up
    up[:, 0] += P.anion_associative_detachment(pop.p) * e.n_h
    for s, mol in enumerate(pop.library):
        for z in (-1, 0):
            up[s, z - pop.zmin] += P.ion_up_rate(mol, z, e, pop.p)
    down = pop._capture_rates(e.temperature_k) * e.n_e
    step = pop.present[:, :-1] & pop.present[:, 1:] & (up[:, :-1] > 0) & (down[:, 1:] > 0)
    logr = np.zeros_like(up[:, :-1])
    logr[step] = np.log(up[:, :-1][step]) - np.log(down[:, 1:][step])
    # a state holds molecules only if an open step leads to it from the molecule's lowest charge
    n = len(pop.library)
    reach = np.zeros_like(pop.present)
    reach[np.arange(n), pop.present.argmax(axis=1)] = True
    for i in range(zs.size - 1):
        reach[:, i + 1] |= reach[:, i] & step[:, i]
    logf = np.where(reach, np.concatenate([np.zeros((n, 1)), np.cumsum(logr, axis=1)], axis=1), -np.inf)
    f = np.exp(logf - logf.max(axis=1, keepdims=True))
    return f / f.sum(axis=1, keepdims=True)


def pahchem(pop):
    """{quantity: (molecule, field, psi)} and the photo rates k_up(Z) at G0 = 1 per field."""
    n = len(pop.library)
    q = {k: np.zeros((n, len(E_MAX), PSI.size)) for k in ("z", "sigma", "top", "gamma", "euv_share", "explode", "ir_stand_in")}
    zs = np.array(pop.zs, dtype=float)
    k_up = {e_max: pop._stack(P.field_shape(env(PSI[0], e_max)))[0] for e_max in E_MAX}
    for i, e_max in enumerate(E_MAX):
        for j, psi in enumerate(PSI):
            e = env(psi, e_max)
            f = pop.charge_matrix(e)
            q["z"][:, i, j] = f @ zs
            q["sigma"][:, i, j] = np.sqrt(np.maximum(f @ zs**2 - (f @ zs) ** 2, 0.0))
            q["top"][:, i, j] = [zs[f[s] >= F_MIN].max() for s in range(n)]
            q["gamma"][:, i, j] = pop.photoelectric_heating(e)[0]
            q["explode"][:, i, j] = pop.chain_overflow(e)
            # photoionisations by photons above 13.6 eV: the 100 eV field equals the cut one below it (same G0 normalisation)
            up = (f * k_up[e_max]).sum(axis=1)
            q["euv_share"][:, i, j] = 1.0 - (f * k_up[13.6]).sum(axis=1) / up if e_max > 13.6 else 0.0
            w, E = pop.emission_weights(e), pop.state_powers(P.field_shape(e))
            power = (w * E).sum(axis=2)
            for s, m in enumerate(pop.library):
                q["ir_stand_in"][s, i, j] = sum(power[s, z - pop.zmin] for z in m.stand_ins) / power[s].sum()
    return q, k_up


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    pop = P.Population(L.library(), L.WINDOWS, cache_dir=FP.BASIS_CACHE)
    for e_max in E_MAX:
        pop.prepare(P.field_shape(env(PSI[0], e_max)), workers=args.workers)
    q, k_up = pahchem(pop)
    ncs = [m.nc for m in pop.library]
    jobs = [(nc, psi, e_max) for nc in ncs for e_max in E_MAX for psi in PSI]
    with ProcessPoolExecutor(args.workers) as ex:
        ref = np.array(list(ex.map(wdb06_job, jobs))).reshape(len(ncs), len(E_MAX), PSI.size, 4)
    rpe = {e_max: np.array([wdb06_photo_rates(float(WS.W.radius_cm(nc)), e_max) for nc in ncs]) for e_max in E_MAX}
    # the photon swap: pahchem's chain on WDB06's photoionisation rates, the chain first checked on pahchem's own
    zs = np.array(pop.zs)
    swap, worst = np.zeros((len(ncs), len(E_MAX), PSI.size)), 0.0
    for i, e_max in enumerate(E_MAX):
        k_w = np.zeros((len(ncs), zs.size))
        inside = (zs >= ZS_PHOTO[0]) & (zs <= ZS_PHOTO[-1])
        k_w[:, inside] = rpe[e_max][:, zs[inside] - ZS_PHOTO[0]]
        for j, psi in enumerate(PSI):
            e = env(psi, e_max)
            own = charge_with_photo(pop, e, pop._stack(P.field_shape(e))[0])
            worst = max(worst, float(np.abs(own - pop.charge_matrix(e)).max()))
            swap[:, i, j] = charge_with_photo(pop, e, k_w) @ zs
    q["z_swap"] = swap
    print(f"the swap's chain on pahchem's own rates reproduces charge_matrix to {worst:.1e}")
    for s, m in enumerate(pop.library):
        live = P.field_shape(env(PSI[0], 100.0))
        print(f"\n{m.name} (a = {WS.W.radius_cm(m.nc) * 1e8:.1f} A; live to +{max(pop._live_charges(m, live))} of the cold limit "
              f"+{max(m.charges)} at 100 eV):  psi " + " ".join(f"{p:8.0e}" for p in PSI))
        for i, e_max in enumerate(E_MAX):
            rows = ((f"<Z> pahchem, {e_max:g} eV", q["z"][s, i]), (f"<Z> WDB06, {e_max:g} eV", ref[s, i, :, 0]),
                    ("<Z> pahchem, WDB06 photo", q["z_swap"][s, i]),
                    ("sigma_Z pahchem", q["sigma"][s, i]), ("sigma_Z WDB06", ref[s, i, :, 1]),
                    (f"Z(f >= {F_MIN:g}) pahchem", q["top"][s, i]), (f"Z(f >= {F_MIN:g}) WDB06", ref[s, i, :, 2]),
                    ("Gamma pahchem/WDB06", q["gamma"][s, i] / ref[s, i, :, 3]))
            if e_max > 13.6:
                rows += (("photo-ion. above 13.6 eV", q["euv_share"][s, i]), ("IR from stand-ins", q["ir_stand_in"][s, i]),
                         ("explosions [s^-1]", q["explode"][s, i]))
            for label, a in rows:
                fmt = "{:8.2e}" if label.startswith("explosions") else "{:8.3f}"
                print(f"  {label:26s}" + " ".join(fmt.format(v) for v in a))
        print("  100 eV field, G0 = 1:    Z  photo rate [s^-1] (EUV share)    pahchem/WDB06   IP [eV]          mean yield above IP\n"
              "                              pahchem          WDB06                          pahchem WDB06    pahchem WDB06")
        for z in range(-1, max(m.charges) + 1):
            kp = k_up[100.0][s, z - pop.zmin]
            k13 = k_up[13.6][s, z - pop.zmin]
            r = rpe[100.0][s, z - ZS_PHOTO[0]]
            r13 = rpe[13.6][s, z - ZS_PHOTO[0]]
            if z <= 6 or z % 5 == 0 or z == max(m.charges):
                ip = (f"{'':16s}{'':16s}" if z < 0 else "{:7.2f} {:7.2f}    {:5.3f}  {:5.3f}".format(*thresholds_and_yields(pop, m, z)))
                print(f"  {'':22s}{z:+3d}  {kp:9.2e} ({1 - k13 / kp if kp > 0 else 0:4.2f})  {r:9.2e} ({1 - r13 / r if r > 0 else 0:4.2f})"
                      f"  {kp / r if r > 0 else np.nan:10.2f}    {ip}")
    plot(pop, q, ref, k_up, rpe, (args.out or _paths.get_plots_dir("pahchem")) / "pahchem_charge_ladder_euv.png")


def plot(pop, q, ref, k_up, rpe, path):
    use_calima_style()
    n = len(pop.library)
    fig, axes = plt.subplots(3, n, figsize=(3.3 * n, 9.5), gridspec_kw={"hspace": 0.32, "wspace": 0.3})
    styles = {13.6: "--", 100.0: "-"}
    for s, m in enumerate(pop.library):
        for i, e_max in enumerate(E_MAX):
            first = s == 0
            axes[0, s].semilogx(PSI, q["z"][s, i], color="#2a78d6", ls=styles[e_max], lw=2.0,
                                label=f"pahchem, to {e_max:g} eV" if first else None)
            axes[0, s].semilogx(PSI, ref[s, i, :, 0], color="#eb6834", ls=styles[e_max], lw=1.5,
                                label=f"WDB06 graphite, to {e_max:g} eV" if first else None)
            if e_max > 13.6:
                axes[0, s].semilogx(PSI, q["z_swap"][s, i], color="#2a78d6", ls=":", lw=1.5,
                                    label="pahchem with WDB06's photoionisation, to 100 eV" if first else None)
            axes[2, s].loglog(PSI, q["gamma"][s, i] / ref[s, i, :, 3], color="0.2", ls=styles[e_max], lw=1.5,
                              label=(f"to {e_max:g} eV" if s == 0 else None))
            zz = np.arange(-1, max(m.charges) + 1)
            kp = k_up[e_max][s, zz - pop.zmin]
            ok = kp > 0
            axes[1, s].semilogy(zz[ok], kp[ok], color="#2a78d6", ls=styles[e_max], lw=2.0)
            r = rpe[e_max][s, zz - ZS_PHOTO[0]]
            axes[1, s].semilogy(zz[r > 0], r[r > 0], color="#eb6834", ls=styles[e_max], lw=1.5)
        axes[0, s].set_title(m.name.replace(" (", ", UID ").rstrip(")"), fontsize=11)
        axes[0, s].set_xlabel(r"$\psi = G_0\sqrt{T}/n_e$")
        axes[1, s].set_xlabel(r"$Z$")
        axes[2, s].set_xlabel(r"$\psi$")
        axes[2, s].axhline(1.0, color="0.7", lw=1, zorder=0)
        for a in axes[:, s]:
            a.minorticks_on()
            a.tick_params(which="both", direction="in", top=True, right=True)
    axes[0, 0].set_ylabel(r"$\langle Z\rangle$")
    axes[1, 0].set_ylabel(r"photoionisation rate at $G_0 = 1$ [s$^{-1}$]")
    axes[2, 0].set_ylabel(r"$\Gamma_{\rm PE}$ pahchem / WDB06")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0))
    axes[2, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(rf"Kurucz {TEFF // 1000} kK, $G_0 = 10^{{4}}$, $T = 10^{{4}}$ K, $n({{\rm H^+}}) = n_e$", fontsize=13, y=1.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
