#!/usr/bin/env python3
"""pahchem's PAH photoelectric heating against Berné+22 (A&A 667, A159), one ingredient at a time.

A four-to-six-state charge chain with swappable inputs. Step 0 is the Berné+22 model (their code,
``four_levels_model.py``) rebuilt from their own data files. It must reproduce the output of their code
(``--berne-json``), which validates the kernel ``photo.photoelectron_power`` and the chain. Each later step
replaces one input by pahchem's, cumulatively. The last step must equal ``Population.photoelectric_heating``
for PAHdb v4.00 circumcoronene (C54H18, calibrated IPs).

Berné+22 inputs (their code):
- the field: their ``ISRF/draine1978.txt`` (one-sided, 2 pi), G0 from 91.2-240 nm;
- sigma: the per-C average of 9-10 DFT molecules (C24-C66) times N_C;
- IPs: 3.9 eV + the WD01-like charge terms (5.74, 8.88 eV for N_C = 54); detachment at a fixed 6 eV, yield 1;
- yields: Jochims+96 (neutral), Wenzel+20 (cation); the dication is the top;
- recombination: V90/BT94 Spitzer form 1.28e-10 N_C T^1/2 (1 + phi (1 + Z)); attachment: coronene,
  2.74e-9 (T/300)^0.11 exp(1.12/T);
- partition 1 for detachment and 0.46 otherwise; eps over the power absorbed below 13.6 eV (down to 0.6 eV).

The steps (cumulative): absorbed power counted on 5-13.6 eV only (eps); pahchem's Draine field on E_PH;
Li & Draine sigma; pahchem's thresholds, yields and chain (EA 1.52, IPs 6.51, 9.27, 12.07 eV, detachment
Jochims at the EA, up to +3); pahchem's cation recombination (DS87, 0.82 disc); pahchem's neutral attachment
(DS87 capture x statistical sticking, against their coronene law).

    check_berne22_pe_heating.py --berne-dir ~/Documents/GitHub/photoelectric-heating [--berne-json run.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pycalima.pahchem import chemistry, fields, photo, population
from pycalima.pahchem.baseline import E_PH, Environment

H_EV_S, C_CM, EV_ERG = 4.135667696e-15, 2.99792458e10, 1.602176634e-12
NC, T, PARTITION = 54, 100.0, photo.PHOTOELECTRON_PARTITION
BERNE_SIGMA = {"anion": ["coronene", "ovalene", "tetrabenzocoronene", "circumbiphenyl", "circumanthracene",
                         "circumpyrene", "hexabenzocoronene", "dicoronylene", "circumcoronene", "circumovalene"],
               "rest": ["ovalene", "tetrabenzocoronene", "circumbiphenyl", "circumanthracene", "circumpyrene",
                        "hexabenzocoronene", "dicoronylene", "circumcoronene", "circumovalene"]}
NCS = {"coronene": 24, "ovalene": 32, "tetrabenzocoronene": 36, "circumbiphenyl": 38, "circumanthracene": 40,
       "circumpyrene": 42, "hexabenzocoronene": 42, "dicoronylene": 48, "circumcoronene": 54, "circumovalene": 66}
STATE_DIR = {-1: ("anions", "anion"), 0: ("neutrals", "neutral"), 1: ("cations", "cation"), 2: ("dications", "dication")}


def berne_field(berne_dir):
    """(E [eV] ascending, photon flux [cm^-2 s^-1 eV^-1], G0) of their draine1978.txt, as their radiation_field()."""
    lam_nm, i_lam = np.loadtxt(berne_dir / "ISRF" / "draine1978.txt", unpack=True, skiprows=3)  # astropy ascii header
    E = (H_EV_S * C_CM / (lam_nm * 1e-7))[::-1]
    i_e = i_lam[::-1] * H_EV_S * C_CM * 1e7 / E**2                         # erg s^-1 cm^-2 sr^-1 eV^-1
    sel = (lam_nm >= 91.2) & (lam_nm <= 240)
    g0 = np.trapezoid(2 * np.pi * i_lam[sel], lam_nm[sel]) / (5.6e-14 * C_CM)
    return E, 2 * np.pi * i_e / (E * EV_ERG), g0


def berne_sigma(berne_dir, z, E):
    names = BERNE_SIGMA["anion" if z == -1 else "rest"]
    acc = None
    for n in names:
        sub, suffix = STATE_DIR[z]
        e, s = np.loadtxt(berne_dir / sub / f"{n}_{suffix}.txt", unpack=True)
        keep = e < 13.6
        v = np.interp(E, e[keep], (s / NCS[n])[keep])
        acc = v if acc is None else acc + v
    return acc / len(names) * NC * 1e-18


class Flux:
    """A tabulated photon flux with the ``photon_rate`` interface of ``fields.RadiationField``."""

    def __init__(self, E, F):
        self.E, self.F = E, F

    def photon_rate(self, s, energy_ev=None):
        return float(np.trapezoid(s * self.F, self.E))


def chain(E, flux, sigma, yields, thresholds, partitions, down, absorb_min_ev):
    """(f[z], Gamma [erg s^-1 per molecule], eps) of the chain; ``down[z]`` [s^-1] is the capture into z from z+1."""
    zs = sorted(sigma)
    f, up = {zs[0]: 1.0}, {}
    for z in zs[:-1]:
        up[z] = flux.photon_rate(sigma[z] * yields[z])
        f[z + 1] = f[z] * up[z] / down[z]
    tot = sum(f.values())
    f = {z: v / tot for z, v in f.items()}
    gamma = sum(f[z] * photo.photoelectron_power(flux, sigma[z], yields[z], E, thresholds[z], partitions[z])
                for z in zs[:-1]) * EV_ERG
    m = E >= absorb_min_ev
    absorbed = sum(f[z] * np.trapezoid((sigma[z] * flux.F * E)[m], E[m]) for z in zs) * EV_ERG
    return f, gamma, gamma / absorbed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--berne-dir", type=Path, required=True)
    ap.add_argument("--berne-json", type=Path, help="output of their code over the same n_e scan, for step 0")
    args = ap.parse_args()
    import fit_34um_sidegroups as S

    mol = next(m for m in S.library() if m.name == "C54H18 (37)")
    Eb, Fb, g0 = berne_field(args.berne_dir)
    Eb, Fb = Eb[Eb <= 13.6], Fb[Eb <= 13.6]
    a = (NC / 468.0) ** (1 / 3)
    e2_nm = 1.439964                                                     # e^2 / (4 pi eps0) [eV nm]
    ip0 = 3.9 + 0.5 * e2_nm / a + 2 * e2_nm / a * 0.03 / a
    ip1 = 3.9 + 1.5 * e2_nm / a + 3 * e2_nm / a * 0.03 / a
    zs_b = (-1, 0, 1, 2)
    pop = population.Population([mol], {"all": (1.0, 1e4)})
    ref = json.loads(args.berne_json.read_text()) if args.berne_json else None
    n_es = [r["n_e"] for r in ref] if ref else list(np.geomspace(3e-4, 3e1, 13))
    print(f"Berne G0 = {g0:.3f} (91.2-240 nm), N_C = {NC}, T = {T} K; IPs {ip0:.3f}, {ip1:.3f}; detachment 6 eV\n")

    def berne_inputs(E, sig_fn):
        sigma = {z: sig_fn(z, E) for z in zs_b}
        yields = {-1: (E >= 6.0).astype(float), 0: photo.yield_jochims1996(E, ip0),
                  1: photo.yield_wenzel2020(E, ip1, NC)}
        return sigma, yields, {-1: 6.0, 0: ip0, 1: ip1}, {-1: 1.0, 0: PARTITION, 1: PARTITION}

    def berne_down(ne):
        phi = 1.85e5 / (T * np.sqrt(NC))
        return {-1: ne * 2.74e-9 * (T / 300) ** 0.11 * np.exp(1.12 / T),
                0: ne * 1.28e-10 * NC * np.sqrt(T) * (1 + phi), 1: ne * 1.28e-10 * NC * np.sqrt(T) * (1 + 2 * phi)}

    zs_p = mol.charges

    def pahchem_inputs(E, sig_fn):
        sigma = {z: sig_fn(z, E) for z in zs_p}
        yields = {z: np.interp(E, E_PH, population._yield(mol, z, pop.p)) for z in zs_p[:-1]}
        th = {z: mol.ionisation_ev[z] for z in zs_p[:-1]}
        return sigma, yields, th, {z: (1.0 if z < 0 else PARTITION) for z in zs_p[:-1]}

    def pahchem_down(ne, capture):
        if capture == "cation":                                          # pahchem recombination, their attachment
            cap = pop._capture_rates(T)[0]
            d = {z: ne * cap[z + 1 - pop.zmin] for z in zs_p[1:-1]}
            d[-1] = ne * 2.74e-9 * (T / 300) ** 0.11 * np.exp(1.12 / T)
            return d
        if capture == "berne":
            phi = 1.85e5 / (T * np.sqrt(NC))
            d = {z: ne * 1.28e-10 * NC * np.sqrt(T) * (1 + phi * (1 + z)) for z in zs_p[1:-1]}
            d[-1] = ne * 2.74e-9 * (T / 300) ** 0.11 * np.exp(1.12 / T)
            return d
        cap = pop._capture_rates(T)[0]
        return {z: ne * cap[z + 1 - pop.zmin] for z in zs_p[:-1]}

    ld = lambda z, E: photo.sigma_li_draine(NC, z, E)                    # noqa: E731
    bs = lambda z, E: berne_sigma(args.berne_dir, z, E)                  # noqa: E731
    dr = fields.draine1978(g0=g0)
    Ep = E_PH
    Fp = dr.flux_at(Ep)
    steps = [
        ("0 Berne+22 (their inputs)", Eb, Flux(Eb, Fb), lambda: berne_inputs(Eb, bs), lambda ne: berne_down(ne), 0.0),
        ("1 eps over 5-13.6 eV", Eb, Flux(Eb, Fb), lambda: berne_inputs(Eb, bs), lambda ne: berne_down(ne), 5.0),
        ("2 pahchem Draine field on E_PH", Ep, Flux(Ep, Fp), lambda: berne_inputs(Ep, bs), lambda ne: berne_down(ne), 5.0),
        ("3 Li & Draine sigma", Ep, Flux(Ep, Fp), lambda: berne_inputs(Ep, ld), lambda ne: berne_down(ne), 5.0),
        ("4 pahchem thresholds, yields, -1..+3", Ep, Flux(Ep, Fp), lambda: pahchem_inputs(Ep, ld),
         lambda ne: pahchem_down(ne, "berne"), 5.0),
        ("5a pahchem cation recombination (DS87)", Ep, Flux(Ep, Fp), lambda: pahchem_inputs(Ep, ld),
         lambda ne: pahchem_down(ne, "cation"), 5.0),
        ("5 + statistical neutral attachment (= pahchem)", Ep, Flux(Ep, Fp), lambda: pahchem_inputs(Ep, ld),
         lambda ne: pahchem_down(ne, "pahchem"), 5.0),
    ]
    rows = {}
    for name, E, flux, inputs, down, amin in steps:
        sigma, yields, th, part = inputs()
        rows[name] = [chain(E, flux, sigma, yields, th, part, down(ne), amin) for ne in n_es]
    per_h = 0.1 / NC * 2.7e-4                                           # their f_C = 0.1, C/H = 2.7e-4
    for i, ne in enumerate(n_es):
        gam = g0 * np.sqrt(T) / ne
        print(f"gamma {gam:9.3g}  n_e {ne:.3g}")
        if ref:
            r = ref[i]
            print(f"   their code                                   eps {r['eps']:.4f}  Gamma/H {r['heat_per_H']:.3e}  "
                  f"f(-,0,+,++) {np.round(r['f'], 3)}")
        for name, res in rows.items():
            f, g, eps = res[i]
            fs = [f.get(z, 0.0) for z in (-1, 0, 1, 2)] + [f.get(3, 0.0)]
            print(f"   {name:44s} eps {eps:.4f}  Gamma/H {g * per_h:.3e}  f(-,0,+,++,+3) {np.round(fs, 3)}")
        env = Environment("berne", g0, 0.0, ne, T, None)
        gp, _, ep = pop.photoelectric_heating(env)
        print(f"   {'Population.photoelectric_heating':44s} eps {ep[0]:.4f}  Gamma/H {gp[0] * per_h:.3e}")


if __name__ == "__main__":
    main()
