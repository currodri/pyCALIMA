"""Golden outputs of the GSD operators at their defaults.

The per-cell GSD operator adds opt-in features to these operators (a collision pair rule, per-moment sinks, ...).
With every feature off, the outputs must stay what they were before that work. ``data/gsd_golden.npz`` holds them for fixed inputs, which cover:
- collisions, on both the fused table path and the generic path;
- sputtering, and growth through the remap;
- the gas exchange;
- the group charge solve, for graphite and for 'ac' (which needs the DustEM THEMIS data).

Regenerate the file only on purpose (a new reference machine, at the commit that introduced it):

    python tests/test_gsd_golden.py --write

Another machine's libm moves the collisions and sputtering by ~1e-10 (megatron, 2026-11-26). Such a machine keeps its
own file, written at the reference commit, outside the checkout: ``$CALIMA_GSD_GOLDEN`` names it, for the test and for
``--write``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gsd_quick

GOLDEN = Path(os.environ.get("CALIMA_GSD_GOLDEN", Path(__file__).parent / "data" / "gsd_golden.npz"))
# Tight enough to catch any change of the physics or the numerics; loose enough for a different libm. If another
# machine fails only by round-off, regenerate there at the commit that introduced the file.
RTOL = 1e-11
C_LIGHT = 2.99792458e10
M_P = 1.67262192e-24
M_H = 1.6735575e-24
YR = 3.15576e7
PC = 3.0857e18


def _carbon_state(grid, n_H, dust_to_gas=0.005):
    """A realisable MRN-like carbon state on ``grid``: N ∝ a^-2.5 per bin, each bin's mean mass at the geometric mean
    of its edges, normalised to ``dust_to_gas`` of the gas mass."""
    a = grid.a_centres
    m = np.sqrt(grid.m_edges[:-1] * grid.m_edges[1:])
    N = a ** -2.5
    N *= dust_to_gas * 1.4 * M_H * n_H / np.sum(N * m)
    return N, N * m


def case_collide(table_ppd):
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    grid = CarbonLayout().grid()
    n_H = 30.0
    N, M = _carbon_state(grid, n_H)
    rv = RelativeVelocity(T=100.0, n_H=n_H, mu=1.4, v_turb=2.6e5, L=10.0 * PC)
    kw = {} if table_ppd == "default" else {"table_ppd": table_ppd}
    model = GrainCollisions(get_material("themis_ac").shatter, rv, law="KT10", coagulation=True,
                            a_frag_min=float(grid.a_edges[0]), collector="remove", n_speed=4, threads=1, **kw)
    c = collide(grid, N, M, model, 30e6 * YR, errmax=1e-3)
    return {"N": c["N"], "M": c["M"], "sink_M": c["sink_M"], "overflow_M": c["overflow_M"],
            "n_steps": c["n_steps"]}


def case_sputter():
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.sputtering import erosion_rate, sputter

    grid = CarbonLayout().grid()
    N, M = _carbon_state(grid, 1.0)
    adot = erosion_rate("graphite", 1e6, 1.0, rho=grid.rho)
    N1, M1, dM = sputter(grid, N, M, adot, 1e5 * YR)
    return {"N": N1, "M": M1, "dM": dM}


def case_growth():
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.remap import remap

    grid = CarbonLayout().grid()
    N, M = _carbon_state(grid, 1.0)
    N1, M1, lost, oN, oM = remap(grid, N, M, lambda a: a + 2e-8)
    return {"N": N1, "M": M1, "lost": lost, "oN": oN, "oM": oM}


def case_exchange():
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    model = get_sublimation_model("graphite", "GD89")
    pop = Population.empty(model, 3e-8, 1e-3, 8, T_d=1700.0, nucleation=False)
    n_C = 10 ** -3.44 * 1e10
    a = pop.grid.a_centres
    w = np.exp(-np.log(a / 1e-5) ** 2 / (2 * 0.15 ** 2))
    pop.M = w / w.sum() * 0.99 * n_C * 12.011 * 1.66053906660e-24
    pop.N = pop.M / pop.grid.mass(a)
    gas = {"C": 0.01 * n_C}
    for _ in range(3):
        exchange_step([pop], gas, 1e4, 1e7)
    return {"N": pop.N, "M": pop.M, "gas_C": gas["C"]}


def _bb_groups(Tsrc, G0, L0, L1):
    """Group photon densities [cm^-3] and mean energies [eV] of a blackbody with G0 Habing units in 6-13.6 eV."""
    ev, kb = 1.602176634e-12, 8.617333262e-5
    E = np.geomspace(0.1, 500.0, 6000)
    uE = E ** 3 / np.expm1(E / (kb * Tsrc))
    b = (E >= 6) & (E <= 13.6)
    uE *= G0 * 5.33e-14 / np.trapezoid(uE[b], E[b])
    N, egy = [], []
    for l0, l1 in zip(L0, L1, strict=True):
        m = (E >= l0) & (E < l1)
        n = np.trapezoid(uE[m] / (E[m] * ev), E[m])
        N.append(n)
        egy.append(np.trapezoid(uE[m], E[m]) / ev / n)
    return np.array(N), np.array(egy)


def case_charging(material):
    from pycalima.models.dust_charge import rt_group_charging as rg

    out = {}
    for a in (1.06e-7, 3e-7, 5e-6):
        tab = rg.build_bin_table(material, a, U=rg.u_grid(n=241), Ts=np.geomspace(3e3, 3e6, 13),
                                 T_hot=np.geomspace(1e3, 1e9, 13), label=f"{material}_{a:g}")
        N, egy = _bb_groups(3e4, 10.0, tab["L0"], tab["L1"])
        gt = rg.GroupChargingTable(tab, egy=egy)
        for T, ne in ((100.0, 3e-2), (8e3, 0.03)):
            r = gt.solve(N, C_LIGHT, T, ne, [(ne, 1.0, M_P)])
            out[f"a{a:g}_T{T:g}"] = np.array([r["Zmean"], r["Zsigma"], r["Gamma"], r["Lambda"]])
    return out


CASES = {"collide_table": lambda: case_collide("default"), "collide_generic": lambda: case_collide(None),
         "sputter": case_sputter, "growth": case_growth, "exchange": case_exchange,
         "charging_graphite": lambda: case_charging("graphite"), "charging_ac": lambda: case_charging("ac")}
NEEDS_DUSTEM = {"charging_ac"}


def _flatten(name, d):
    return {f"{name}.{k}": np.asarray(v, float) for k, v in d.items()}


def compute_all():
    out = {}
    for name, f in CASES.items():
        out.update(_flatten(name, f()))
    return out


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN.name} missing: run `python tests/test_gsd_golden.py --write` at the reference commit")
    with np.load(GOLDEN) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.parametrize("name", list(CASES))
def test_defaults_reproduce_the_golden_outputs(golden, name, request):
    if name in NEEDS_DUSTEM:
        request.getfixturevalue("themis_cm20")
    got = _flatten(name, CASES[name]())
    want = {k: v for k, v in golden.items() if k.startswith(name + ".")}
    assert set(got) == set(want)
    for k, v in want.items():
        np.testing.assert_allclose(got[k], v, rtol=RTOL, atol=0.0, err_msg=k)


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit(__doc__)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    res = compute_all()
    np.savez_compressed(GOLDEN, **res)
    for k, v in res.items():
        print(f"{k:28s} {np.atleast_1d(v)[:4]}")
    print(GOLDEN)
