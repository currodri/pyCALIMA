"""Shared pieces of the PAHrt study: the truth that a per-cell PAH solver for RAMSES must reproduce.

The study (plan of 2026-11-24, "carrying pahchem's PAH physics into future RAMSES versions") asks how far pahchem's
model can be reduced to bin-averaged kernels and a small equilibrium solver run per cell. Its reference ("the truth")
is each molecule's full joint (Z, N_H) chain from ``pahchem.population.Population`` (``Params.hydrogen_chain =
"joint"``, read-only use), with the per-state observables of ``Population.joint_state_observables`` (pahchem PR #17,
merged into this study branch).

Here:
- the molecules of the GSD's molecular bins (``CarbonLayout``: N_C 20-42-89-189-400), ``per_bin`` of each, evenly in
  N_C rank (as ``hydrogen_tables --per-bin``); the top bin has 5 complete molecules in PAHdb v4.00;
- base phases, **illustrative inputs** (labelled: the study varies every input around them; they are not predictions
  of any region): WNM and CNM (Wolfire et al. 2003-like), an Orion-Bar-like atomic PDR layer (pahchem's
  ``environments/orion_bar.toml`` ranges), translucent gas, a dense core;
- ``Truth``: one molecule's ``Population``, and ``outputs(env)`` with everything RAMSES would take from the molecular
  bins (charge fractions, heating, cooling, absorbed power, carbon loss, H2, H uptake, the net electron source, the
  ion recombination per ion, the hydrogenation shares).

The electron source counts the electrons each process frees or captures: photoionisation and detachment steps (the
"ionisation" process minus its ion-driven part, ``population.ion_up_rate``), associative detachment and the dZ +1
deposits free one; electron captures and the dZ -1 deposits (with or without a proton) take one; proton loss frees
none (the charge leaves with the proton). The ion recombination on PAHs is alpha[ion] = f(-1) k_mn + f(0) k_ct per
molecule per unit n(ion) [cm^3 s^-1], with ``ion_up_rate`` (both are the same at every N_H in pahchem).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

os.environ.setdefault("CALIMA_WORKERS", "1")

OUTPUTS = ("f_m1", "f_0", "f_p1", "f_p2", "z_mean", "heating", "cooling", "absorbed", "carbon", "h2", "h_uptake",
           "e_source", "alpha_Cp", "alpha_Hp", "bare", "dehydrogenated", "intact", "superh", "dnh")
#: outputs that are rates per molecule (compared in ratio); the rest are shares or charges (compared in difference)
RATES = ("heating", "cooling", "absorbed", "carbon", "h2", "h_uptake", "alpha_Cp", "alpha_Hp")


def bin_edges():
    from pycalima.gsd.carbon import M_C, CarbonLayout

    lay = CarbonLayout()
    return lay.grid().m_edges[:lay.n_pah_bins + 1] / M_C


def molecules(per_bin=5):
    """{bin index: [molecules]}: ``per_bin`` per bin evenly in N_C rank (all of a bin with fewer), the PAHrt kernels'
    choice (``pahrt_kernels.representatives``)."""
    from pycalima.pahchem.pahrt_kernels import representatives

    return dict(enumerate(representatives(bin_edges(), per_bin)))


@dataclass(frozen=True)
class Phase:
    """A cell's inputs: G0 [Habing, in the field's own 6-13.6 eV], n(H I) [cm^-3], T [K], n_e [cm^-3], the ions
    {name: n [cm^-3]}, the field: 'draine' (Draine 1978 FUV), 'draine+old' (with the solar-circle old-star light),
    'kurucz<T_eff>', or 'powerlaw<alpha>' (to 13.6 eV)."""

    g0: float
    n_hi: float
    T: float
    n_e: float
    ions: tuple
    field: str = "draine"

    def environment(self, name="cell"):
        from pycalima.pahchem.baseline import Environment

        kw = {"ions": tuple(self.ions), "optical_isrf": 0.0, "teff_k": None}
        if self.field == "draine+old":
            kw["optical_isrf"] = 1.0
        elif self.field.startswith("kurucz"):
            kw["teff_k"] = int(self.field[6:])
        elif self.field.startswith("powerlaw"):
            kw["field_powerlaw"] = float(self.field[8:])
        elif self.field != "draine":
            raise ValueError(f"unknown field {self.field!r}")
        return Environment(name, self.g0, self.n_hi, self.n_e, self.T, **kw)


#: illustrative base phases (labelled; see the module doc)
PHASES = {
    "WNM": Phase(1.7, 0.5, 8000.0, 0.012, (("C+", 7e-5), ("H+", 0.0119))),
    "CNM": Phase(1.7, 40.0, 80.0, 0.024, (("C+", 5.6e-3), ("H+", 0.018))),
    "translucent": Phase(10.0, 300.0, 50.0, 0.14, (("C+", 0.14),)),
    "PDR": Phase(2e4, 5e4, 600.0, 7.0, (("C+", 7.0),)),
    "dense": Phase(0.01, 5.0, 15.0, 1e-3, (("S+", 1e-3),)),
}


def scaled_ions(phase, factor):
    return tuple((n, x * factor) for n, x in phase.ions)


class Truth:
    """One molecule's full joint chain (``Population`` with the joint (Z, N_H) chain)."""

    def __init__(self, mol, params_kw=None):
        from pycalima import _paths
        from pycalima.pahchem import population as P

        self.mol = mol
        self.params = P.Params(hydrogen_chain="joint", **(params_kw or {}))
        self.pop = P.Population([mol], {"all": (1.0, 1e4)}, params=self.params,
                                cache_dir=_paths.get_model_data_dir() / "pahchem_population_basis")

    def network(self, env):
        """(network, weights {name: W[Z, N_H]}) at ``env``."""
        from pycalima.pahchem import hydrogen_observables as HO

        nets = self.pop.joint_networks(env)
        return nets[0], HO.state_weights(self.pop, env, nets)[0]

    def ion_rates(self, env):
        """{ion: (k_mn, k_ct)} [cm^3 s^-1]: the ion-driven steps out of the anion and the neutral per unit n(ion)."""
        from pycalima.pahchem import population as P

        out = {}
        for ion, _ in env.ions:
            one = replace(env, ions=((ion, 1.0),))
            out[ion] = (P.ion_up_rate(self.mol, -1, one, self.params), P.ion_up_rate(self.mol, 0, one, self.params))
        return out

    def outputs(self, env, net=None, W=None, F=None):
        if net is None:
            net, W = self.network(env)
        return outputs(net, W, self.mol.nh, self.ion_rates(env), env, F)


def electrons_per_event(name, dz):
    """Free electrons a process makes per event (negative: captured); ion-driven steps are handled by the caller."""
    if name.startswith("ionisation"):
        return dz if "proton" not in name else dz + int(name.split("emitted")[0].split()[-2])
    if name == "associative detachment":
        return 1
    if name.startswith("deposit dZ +1"):
        return 1
    if name.startswith("deposit dZ -1") or name.startswith("electron capture"):
        return -1
    return 0                 # H, H2, proton loss, H addition, Eley-Rideal


def outputs(net, W, nh0, ion_rates, env, F=None):
    """Every output (``OUTPUTS``) of one molecule's network ``net`` at its state F (the steady state if None)."""
    F = net.steady_state_closed() if F is None else F
    z = np.asarray(net.charges)
    nh = np.asarray(net.nh_values)
    fz = F.sum(axis=1)
    frac = {k: float(fz[list(z).index(c)]) if c in z else 0.0 for k, c in (("f_m1", -1), ("f_0", 0), ("f_p1", 1),
                                                                         ("f_p2", 2))}
    out = dict(frac)
    out["z_mean"] = float(fz @ z)
    for k in ("heating", "cooling", "absorbed", "carbon", "h2", "h_uptake"):
        out[k] = float((W[k] * F).sum())
    ion_up = np.zeros(z.size)                      # ion-driven up rates out of each charge [s^-1]
    for ion, n in env.ions:
        k_mn, k_ct = ion_rates[ion]
        ion_up += n * np.where(z == -1, k_mn, 0.0) + n * np.where(z == 0, k_ct, 0.0)
    e = 0.0
    for p in net.processes:
        per = electrons_per_event(p.name, p.dz)
        if per == 0:
            continue
        r = p.rate
        if p.name == "ionisation (dZ +1)":
            r = np.maximum(r - ion_up[:, None], 0.0)
        e += per * float((r * F).sum())
    out["e_source"] = e
    for ion, key in (("C+", "alpha_Cp"), ("H+", "alpha_Hp")):
        if ion in ion_rates:
            k_mn, k_ct = ion_rates[ion]
            out[key] = frac["f_m1"] * k_mn + frac["f_0"] * k_ct
        else:
            out[key] = 0.0
    fn = F.sum(axis=0)
    out["bare"] = float(fn[nh == 0].sum())
    out["dehydrogenated"] = float(fn[(nh > 0) & (nh < nh0)].sum())
    out["intact"] = float(fn[nh == nh0].sum())
    out["superh"] = float(fn[nh > nh0].sum())
    out["dnh"] = float(fn @ (nh - nh0))
    return out
