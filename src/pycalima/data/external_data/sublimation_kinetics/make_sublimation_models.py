#!/usr/bin/env python3
"""Build sublimation_models.toml from compilation.csv.

The compilation (compilation.csv, COMPILATION_NOTES.md) lists vapour-pressure
fits, evaporation coefficients and surface energies per material and source.
This script selects, per material, the rows that make each named dataset
(``fiducial``, ``GD89`` and alternatives), copying the numbers straight from
the CSV so that nothing is retyped. The choices and their reasons are in
SELECTION below; COMPILATION_NOTES.md section 6 gives the per-material
caveats behind them.

Run from anywhere:  python make_sublimation_models.py
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMU = 1.66053906660e-24

# Standard atomic weights [amu] of key species.
ATOMIC = {"C": 12.011, "Mg": 24.305, "Si": 28.086, "Fe": 55.845, "Al": 26.982,
          "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Na": 22.990, "K": 39.098,
          "Ca": 40.078, "O": 15.999}

# Hypothetical radius per key species [Angstrom] of Nozawa et al. (2003)
# Table 2, used only to fill a missing bulk density (rho = m_key_unit / V).
N03_A0 = {"Si": (1.684, 28.086), "Fe3O4": (1.805, 231.531 / 3)}

# GD89 constants as in models/dust_radiation/dust_sublimation.py GD89_PARAMS
# (Guhathakurta & Draine 1989): J = alpha * P0 * exp(-B/T) per monomer.
GD89 = {
    "graphite": dict(P0=4.6e30, B=81200.0, theta=20000.0, alpha=0.1,
                     unit_mass=12.011, rho=2.24),
    "silicate": dict(P0=7.0e30, B=68100.0, theta=20000.0, alpha=0.1,
                     unit_mass=140.69, rho=3.5),
}
GD89_REF = "Guhathakurta & Draine 1989, ApJ 345, 230 (eqs. 3.12-3.17)"

# ---------------------------------------------------------------------------
# Selection. Each dataset is a list of channels taken from CSV rows
# (material, source_id); "n" is the number of evaporating units (the material's
# monomer, e.g. C atoms or formula units) carried by one vapour molecule of
# the channel; "alpha" is (alpha0, alpha_T_K) for alpha = alpha0 exp(-alpha_T_K/T),
# capped at 1. "sigma" is the surface energy [erg cm^-2] for the small-grain
# correction and nucleation; None leaves it unset.
# ---------------------------------------------------------------------------
W = "W18-D2+FE"


def eff(row, alpha, sigma=None, note=""):
    return dict(channels=[dict(row=row, n=1, alpha=alpha)], sigma=sigma, note=note)


SELECTION = {
    "graphite": dict(
        name="graphite", formula="C", unit="C atom", key=("C", 1), heat="graphite",
        datasets={
            "fiducial": dict(channels=[
                dict(row=("graphite", "Z70-C1"), n=1, alpha=(0.24, 0.0)),
                dict(row=("graphite", "Z70-C2"), n=2, alpha=(0.50, 0.0)),
                dict(row=("graphite", "Z70-C3"), n=3, alpha=(0.023, 0.0))],
                sigma=1400.0,
                note="Zavitsanos 1970 C1/C2/C3 pressures with Zavitsanos 1966 species "
                     "alpha (same laboratory). 1.4x GD89 at 2000 K, 2.8x at 2500 K. "
                     "sigma from Nozawa et al. 2003 (Tabak et al. 1975)."),
            "Z70_TW57": dict(channels=[
                dict(row=("graphite", "Z70-C1"), n=1, alpha=(0.37, 0.0)),
                dict(row=("graphite", "Z70-C2"), n=2, alpha=(0.34, 0.0)),
                dict(row=("graphite", "Z70-C3"), n=3, alpha=(0.08, 0.0))],
                sigma=1400.0,
                note="Zavitsanos 1970 pressures with Thorn & Winslow 1957 species alpha "
                     "(2.5x GD89 at 2000 K)."),
            "W18": eff(("graphite", W), (0.1, 0.0), 1400.0,
                       "Woitke et al. 2018 free-evaporation total vapour, alpha adopted."),
            "N03": eff(("graphite", "N03+conv"), (0.1, 0.0), 1400.0,
                       "Nozawa et al. 2003 Gibbs fit (C1 only), alpha adopted."),
        }),
    "amorphous_carbon": dict(
        name="amorphous carbon", formula="C", unit="C atom", key=("C", 1), heat="graphite",
        proxy_of="graphite",
        note="No sublimation data separate from graphite (COMPILATION_NOTES s.6); "
             "graphite datasets are used as a proxy. Amorphous C is metastable, so its "
             "true vapour pressure is >= graphite's by an unknown amount."),
    "silicate": dict(
        name="astronomical silicate (forsterite kinetics)", formula="Mg2SiO4",
        unit="Mg2SiO4 formula unit", key=("Si", 1), heat="silicate",
        density=3.5,
        note="No astrosilicate sublimation data exist; forsterite is the proxy "
             "(COMPILATION_NOTES s.6). Key species Si as in Baskin & Laor 2018.",
        datasets={
            "fiducial": eff(("forsterite", "vL14a"), (0.1, 0.0), 436.0,
                            "Nagahara et al. 1994 Knudsen-cell forsterite via van Lieshout "
                            "et al. 2014a Table 3; alpha 0.1 (Hashimoto 1989; Tsuchiyama "
                            "et al. 1998). 7-40x faster than GD89 at 1500-2000 K."),
            "W18": eff(("forsterite", W), (0.1, 0.0), 436.0,
                       "Woitke et al. 2018 free evaporation into vacuum."),
            "N03": eff(("forsterite", "N03+conv"), (0.1, 0.0), 436.0,
                       "Nozawa et al. 2003 Gibbs fit, congruent vacuum evaporation."),
        }),
    "forsterite": dict(
        name="forsterite", formula="Mg2SiO4", unit="Mg2SiO4 formula unit",
        key=("Mg", 2), heat="silicate", gd89_as="silicate",
        datasets={
            "fiducial": eff(("forsterite", "vL14a"), (0.1, 0.0), 436.0,
                            "Nagahara et al. 1994 via van Lieshout et al. 2014a; alpha 0.1."),
            "W18": eff(("forsterite", W), (0.1, 0.0), 436.0, "Woitke et al. 2018."),
            "N03": eff(("forsterite", "N03+conv"), (0.1, 0.0), 436.0, "Nozawa et al. 2003."),
        }),
    "enstatite": dict(
        name="enstatite", formula="MgSiO3", unit="MgSiO3 formula unit", key=("Mg", 1),
        heat="silicate", congruent=False,
        note="Evaporates incongruently (SiO2 lost, forsterite residue; Tachibana et "
             "al. 2002); congruent rows are formal.",
        datasets={
            "fiducial": eff(("enstatite", W), (0.1, 0.0), 400.0,
                            "Woitke et al. 2018 free evaporation; alpha adopted (none measured)."),
            "vL14a": eff(("enstatite", "vL14a"), (0.1, 0.0), 400.0,
                         "van Lieshout et al. 2014a (0.6 dex above W18 at 1700 K)."),
            "N03": eff(("enstatite", "N03+conv"), (0.1, 0.0), 400.0, "Nozawa et al. 2003."),
        }),
    "fayalite": dict(
        name="fayalite", formula="Fe2SiO4", unit="Fe2SiO4 formula unit", key=("Fe", 2),
        heat="silicate",
        datasets={
            "fiducial": eff(("fayalite", W), (0.1, 0.0), None,
                            "Woitke et al. 2018 free evaporation; alpha adopted. vL14a lies "
                            "1.6 dex higher (unresolved)."),
            "vL14a": eff(("fayalite", "vL14a"), (0.1, 0.0), None,
                         "Nagahara et al. 1994 via van Lieshout et al. 2014a [doubtful]."),
        }),
    "ferrosilite": dict(
        name="ferrosilite", formula="FeSiO3", unit="FeSiO3 formula unit", key=("Fe", 1),
        heat="silicate",
        datasets={"fiducial": eff(("ferrosilite", "W18-D4+FE"), (0.1, 0.0), None,
                                  "Woitke et al. 2018 Table D.4 thermochemistry; alpha adopted.")}),
    "silica": dict(
        name="silica", formula="SiO2", unit="SiO2 formula unit", key=("Si", 1),
        datasets={
            "fiducial": eff(("silica", W), (0.013, 0.0), 605.0,
                            "Woitke et al. 2018 total vapour with alpha 0.013 measured against "
                            "the total vapour (Hashimoto 1989)."),
            "vL14a": eff(("silica", "vL14a"), (1.0, 0.0), 605.0,
                         "van Lieshout et al. 2014a, SiO2(g) only, alpha ~1 against it."),
        }),
    "silicon_monoxide": dict(
        name="silicon monoxide (solid)", formula="SiO", unit="SiO", key=("Si", 1),
        datasets={"fiducial": eff(("silicon monoxide", "G13"), (0.521, 3686.0), None,
                                  "Gail et al. 2013 vapour pressure and alpha(T).")}),
    "iron": dict(
        name="metallic iron", formula="Fe", unit="Fe atom", key=("Fe", 1),
        datasets={
            "fiducial": eff(("iron", "JANAF-Fe"), (math.exp(1.44), 2595.0), 1800.0,
                            "JANAF vapour pressure with alpha = exp(1.44 - 2595/T) "
                            "(Tachibana et al. 2011)."),
            "W18": eff(("iron", W), (math.exp(1.44), 2595.0), 1800.0, "Woitke et al. 2018."),
            "N03": eff(("iron", "N03+conv"), (math.exp(1.44), 2595.0), 1800.0,
                       "Nozawa et al. 2003."),
            "vL14a": eff(("iron", "vL14a"), (1.0, 0.0), 1800.0,
                         "Ferguson et al. 2004 via van Lieshout et al. 2014a (0.5 dex low)."),
        }),
    "silicon": dict(
        name="silicon", formula="Si", unit="Si atom", key=("Si", 1),
        datasets={"fiducial": eff(("silicon", "JANAF-Si"), (0.1, 0.0), 800.0,
                                  "JANAF; alpha adopted (none found).")}),
    "chromium": dict(
        name="chromium", formula="Cr", unit="Cr atom", key=("Cr", 1),
        datasets={"fiducial": eff(("chromium", "JANAF-Cr"), (0.9, 0.0), 1880.0,
                                  "JANAF; alpha 0.9 (Pound 1972).")}),
    "troilite": dict(
        name="troilite", formula="FeS", unit="FeS formula unit", key=("Fe", 1),
        congruent=False,
        note="Evaporates incongruently: S leaves (S2 in vacuum, H2S in H2) and a "
             "porous Fe residue remains (Tsuchiyama et al. 1997; Tachibana & "
             "Tsuchiyama 1998). The congruent row is a formal upper bound; the "
             "S-loss channel is recorded but not modelled yet.",
        datasets={"fiducial": eff(("troilite", W), (0.1, 0.0), 380.0,
                                  "Woitke et al. 2018 formal congruent free evaporation.")}),
    "corundum": dict(
        name="corundum", formula="Al2O3", unit="Al2O3 formula unit", key=("Al", 2),
        datasets={
            "fiducial": eff(("corundum", W), (0.1, 0.0), 690.0,
                            "Woitke et al. 2018 with alpha 0.1 (0.02-0.2, face dependent; "
                            "Takigawa et al. 2015)."),
            "N03": eff(("corundum", "N03+conv"), (0.1, 0.0), 690.0, "Nozawa et al. 2003."),
        }),
    "spinel": dict(
        name="spinel", formula="MgAl2O4", unit="MgAl2O4 formula unit", key=("Al", 2),
        datasets={"fiducial": eff(("spinel", W), (0.1, 0.0), None,
                                  "Woitke et al. 2018; alpha adopted.")}),
    "perovskite": dict(
        name="perovskite", formula="CaTiO3", unit="CaTiO3 formula unit", key=("Ti", 1),
        datasets={"fiducial": eff(("perovskite", W), (0.1, 0.0), None,
                                  "Woitke et al. 2018; alpha adopted.")}),
    "rutile": dict(
        name="rutile", formula="TiO2", unit="TiO2 formula unit", key=("Ti", 1),
        datasets={"fiducial": eff(("rutile", W), (0.1, 0.0), 480.6,
                                  "Woitke et al. 2018; alpha adopted; sigma Lee et al. 2018.")}),
    "periclase": dict(
        name="periclase", formula="MgO", unit="MgO formula unit", key=("Mg", 1),
        datasets={"fiducial": eff(("periclase", W), (0.17, 0.0), 1100.0,
                                  "Woitke et al. 2018 with alpha 0.17 (0.12-0.22, Hashimoto 1989).")}),
    "wustite": dict(
        name="wustite", formula="FeO", unit="FeO formula unit", key=("Fe", 1),
        datasets={"fiducial": eff(("wustite", W), (0.1, 0.0), 580.0,
                                  "Woitke et al. 2018 (nominal FeO); alpha adopted.")}),
    "magnetite": dict(
        name="magnetite", formula="Fe3O4", unit="Fe3O4 formula unit", key=("Fe", 3),
        congruent=False, note="Loses O2 before congruent sublimation; formal row.",
        datasets={"fiducial": eff(("magnetite", "W18-D4+FE"), (0.1, 0.0), 400.0,
                                  "Woitke et al. 2018 Table D.4; alpha adopted.")}),
    "hematite": dict(
        name="hematite", formula="Fe2O3", unit="Fe2O3 formula unit", key=("Fe", 2),
        congruent=False, note="Loses O2 (to Fe3O4) before congruent sublimation; formal row.",
        datasets={"fiducial": eff(("hematite", "W18-D4+FE"), (0.1, 0.0), None,
                                  "Woitke et al. 2018 Table D.4; alpha adopted.")}),
    "silicon_carbide": dict(
        name="silicon carbide", formula="SiC", unit="SiC formula unit", key=("Si", 1),
        congruent=False, note="Loses Si and leaves a carbon residue (Yazdi et al. 2016).",
        datasets={
            "fiducial": eff(("silicon carbide", W), (0.1, 0.0), 1800.0,
                            "Woitke et al. 2018 total vapour; alpha adopted."),
            "vL14a": eff(("silicon carbide", "vL14a"), (0.1, 0.0), 1800.0,
                         "van Lieshout et al. 2014a (Lilov 1993, SiC2 only)."),
        }),
    "titanium_carbide": dict(
        name="titanium carbide", formula="TiC", unit="TiC formula unit", key=("Ti", 1),
        datasets={"fiducial": eff(("titanium carbide", W), (0.1, 0.0), 1242.0,
                                  "Woitke et al. 2018; alpha adopted.")}),
    "halite": dict(
        name="halite", formula="NaCl", unit="NaCl formula unit", key=("Na", 1),
        datasets={"fiducial": eff(("halite", "JANAF-NaCl"), (0.4, 0.0), 113.3,
                                  "JANAF; alpha 0.4 (Pound 1972).")}),
    "sylvite": dict(
        name="sylvite", formula="KCl", unit="KCl formula unit", key=("K", 1),
        datasets={"fiducial": eff(("sylvite", "JANAF-KCl"), (0.67, 0.0), 100.3,
                                  "JANAF; alpha 0.63-0.72 (Pound 1972).")}),
    "alabandite": dict(
        name="alabandite", formula="MnS", unit="MnS formula unit", key=("Mn", 1),
        datasets={"fiducial": eff(("alabandite", W), (0.1, 0.0), None,
                                  "Woitke et al. 2018 vacuum free evaporation; alpha adopted.")}),
    "sodium_sulfide": dict(
        name="sodium sulfide", formula="Na2S", unit="Na2S formula unit", key=("Na", 2),
        datasets={"fiducial": eff(("sodium sulfide", W), (0.1, 0.0), None,
                                  "Woitke et al. 2018 vacuum free evaporation; alpha adopted.")}),
    "water_ice": dict(
        name="water ice", formula="H2O", unit="H2O molecule", key=("O", 1),
        datasets={"fiducial": eff(("water ice", "FS84/G11"), (1.0, 0.0), 109.0,
                                  "Fanale & Salvail 1984 vapour pressure; alpha ~1 below "
                                  "~200 K, falling to ~0.15 above (Gundlach et al. 2011).")}),
}


def _csv_rows():
    with open(HERE / "compilation.csv", newline="", encoding="utf-8") as fh:
        return {(r["material"], r["source_id"]): r for r in csv.DictReader(fh)}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _density(mat, rows, sel):
    if mat.get("density"):
        return mat["density"]
    for ds in sel.values():
        for ch in ds["channels"]:
            rho = _num(rows[ch["row"]]["rho_g_cm3"])
            if rho:
                return rho
    formula = mat["formula"]
    if formula in N03_A0:
        a0, m = N03_A0[formula]
        return m * AMU / (4.0 / 3.0 * math.pi * (a0 * 1e-8) ** 3)
    return None


def build():
    rows = _csv_rows()
    out = ["# Sublimation kinetics by material and dataset. GENERATED by",
           "# make_sublimation_models.py from compilation.csv -- do not edit by hand.",
           "#",
           "# Evaporation flux of the material's units [cm^-2 s^-1] from a bulk solid:",
           "#   form = \"hkl\":  J = sum_c n_c alpha_c(T) P_c(T) / sqrt(2 pi m_c k T),",
           "#                  log10 P_c[dyn cm^-2] = A10 - B10_K / T,",
           "#                  alpha_c = min(1, alpha0 exp(-alpha_T_K / T));",
           "#   form = \"gd89\": J = alpha P0 exp(-B_K / T) (Guhathakurta & Draine 1989).",
           "# n_c: units carried by one vapour molecule of channel c. sigma_erg_cm2 sets",
           "# the small-grain term theta = (32 pi / 3)^(1/3) (m_unit/rho)^(2/3) sigma / k",
           "# of B(N) = B - theta N^(-1/3) unless theta_K is given.",
           ""]
    for key, mat in SELECTION.items():
        src = SELECTION[mat["proxy_of"]] if "proxy_of" in mat else mat
        sel = src["datasets"]
        first = next(iter(sel.values()))["channels"][0]["row"]
        unit_mass = _num(rows[first]["mu_unit_amu"]) if mat["formula"] != "C" else 12.011
        rho = _density(mat, rows, sel) if "proxy_of" not in mat else _density(src, rows, sel)
        out.append(f"[material.{key}]")
        out.append(f"name = {_q(mat['name'])}")
        out.append(f"formula = {_q(mat['formula'])}")
        out.append(f"unit = {_q(mat['unit'])}")
        out.append(f"unit_mass_amu = {unit_mass!r}")
        if rho is not None:
            out.append(f"density_gcm3 = {round(rho, 4)!r}")
        out.append(f"key_species = {_q(mat['key'][0])}")
        out.append(f"key_mass_amu = {ATOMIC[mat['key'][0]]!r}")
        out.append(f"key_per_unit = {mat['key'][1]}")
        if mat.get("heat") or src.get("heat"):
            out.append(f"heat_capacity = {_q(mat.get('heat') or src.get('heat'))}")
        out.append(f"congruent = {'false' if mat.get('congruent') is False else 'true'}")
        if "proxy_of" in mat:
            out.append(f"proxy_of = {_q(mat['proxy_of'])}")
        if mat.get("note"):
            out.append(f"note = {_q(mat['note'])}")
        out.append("")
        gd = GD89.get(mat.get("gd89_as", src.get("gd89_as", key if key in GD89 else
                                               ("graphite" if mat["formula"] == "C" else None))))
        if gd:
            out.append(f"[material.{key}.dataset.GD89]")
            out.append('form = "gd89"')
            out.append(f"P0_cm2s = {gd['P0']!r}")
            out.append(f"B_K = {gd['B']!r}")
            out.append(f"alpha = {gd['alpha']!r}")
            out.append(f"theta_K = {gd['theta']!r}")
            out.append(f"unit_mass_amu = {gd['unit_mass']!r}")
            out.append(f"density_gcm3 = {gd['rho']!r}")
            out.append(f"reference = {_q(GD89_REF)}")
            out.append("")
        for dname, ds in sel.items():
            out.append(f"[material.{key}.dataset.{dname}]")
            out.append('form = "hkl"')
            if ds.get("sigma") is not None:
                out.append(f"sigma_erg_cm2 = {ds['sigma']!r}")
            out.append(f"note = {_q(ds['note'])}")
            for ch in ds["channels"]:
                r = rows[ch["row"]]
                a0, aT = ch["alpha"]
                out.append(f"[[material.{key}.dataset.{dname}.channel]]")
                out.append(f"source_id = {_q(r['source_id'])}")
                out.append(f"vapour = {_q(r['formula'])}")
                out.append(f"A10 = {_num(r['A10'])!r}")
                out.append(f"B10_K = {_num(r['B10_K'])!r}")
                out.append(f"mass_amu = {_num(r['mu_unit_amu'])!r}")
                out.append(f"units_per_molecule = {ch['n']}")
                out.append(f"alpha0 = {a0!r}")
                out.append(f"alpha_T_K = {aT!r}")
                out.append(f"T_range_K = {_q(r['T_range_K'])}")
                out.append(f"verification = {_q(r['verification'])}")
            out.append("")
    (HERE / "sublimation_models.toml").write_text("\n".join(out), encoding="utf-8")
    print("wrote", HERE / "sublimation_models.toml")


if __name__ == "__main__":
    build()
