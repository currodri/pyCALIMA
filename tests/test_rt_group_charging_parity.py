"""GroupChargingTable.solve against RAMSES, state by state.

tests/data/rtgroup_parity_lg5.npz holds 652 states of the RAMSES 22-bin charging bench (13 cases x 3000 cells x 22
bins), from six bins (graphite 8.5, 63, 470 nm; silicate 2.6, 32, 390 nm), chosen to cover every branch of the solve
of calima/dust_charging.f90 at RAMSES staging + PR #86 + calima/charge-lg5 7f842bb4: discrete and wide P(Z), the tail
toward Z = 0, LG5's 5- and 7-charge windows, wide_moments used or not, small grains in very cold gas. With them, the
outputs of the Fortran bench driver (lgx_bench) for WDB06rt and LG5 (rtg_lattice), rtg_wide_moments on and off, and
the six bins' tables trimmed to what these states use. Built by studies/pycalima_port_20261118/scripts/
build_fixture.py; the full 858,000-state comparison is diagnostics/dust_charge/check_rtgroup_charging_parity.py.

The criterion is round-off: |rel| <= 1e-10 on every output and the same branch flags (discrete; D the solve's own;
alpha from the tail) in every state. Two outputs are compared in the forms where round-off is small: a charge (<Z>,
Z*) to 1e-10 of max(|Z|, 1e-3) (a mean near zero cancels terms of order one), and an alpha below 1e-100 by its ln,
to 1e-10 of |ln alpha| (the far tail of the Gaussian estimate, ln alpha ~ -270 .. -690).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parent / "data" / "rtgroup_parity_lg5.npz"
CONFIGS = {"c6wdb": {}, "c6wdb_off": {"wide_moments": False}, "c6lg": {"lattice": True},
           "c6lg_off": {"lattice": True, "wide_moments": False}}
M_HION, M_HEION = 1.67262192e-24, 6.6446573e-24
COLS = (["Zmean", "Zsigma", "discrete", "Gamma", "Lambda", "D-1", "D+1", "D+2", "D+3"]
        + [f"alpha{j}" for j in range(10)] + ["Zstar", "flags"])
REL = 1e-10


@pytest.fixture(scope="module")
def fixture():
    z = np.load(FIXTURE)
    tabs = {}
    for b in z["bins"]:
        t = {}
        for key in z.files:
            if key.startswith(f"bin{b}:"):
                v = z[key]
                t[key.split(":", 1)[1]] = v.item() if v.ndim == 0 else v
        tabs[int(b)] = t
    return z, tabs


def _solve(gt, x, zg, opts):
    r = gt.solve(x[0:8], x[16], x[17], x[18], [(x[19], 1.0, M_HION), (x[20], 1.0, M_HEION), (x[21], 2.0, M_HEION)],
                 G0_bg=x[22], Z_guess=zg, recomb="gauss", dlnT=1e-5, coulomb=True, **opts)
    fl = 1.0 if r["D_abs"] else 0.0
    if not opts.get("lattice", False) and r["alpha_tail"]:
        fl += 2.0
    return np.concatenate([[r["Zmean"], r["Zsigma"], float(r["discrete"]), r["Gamma"], r["Lambda"] + r["Lambda_auto"]],
                           r["D"], r["alpha_rec"], [r["Zstar"], fl]])


@pytest.mark.parametrize("config", CONFIGS)
def test_solve_reproduces_ramses_to_round_off(fixture, config):
    from pycalima.models.dust_charge import rt_group_charging as rg

    z, tabs = fixture
    X, S, F, G = z["inputs"], z["states"], z[f"fortran:{config}"], z[f"guess:{config}"]
    gts = {}
    P = np.empty_like(F)
    for n, (_, _, b) in enumerate(S):
        key = (int(b), tuple(X[n, 8:16]))
        if key not in gts:
            gts[key] = rg.GroupChargingTable(tabs[int(b)], egy=X[n, 8:16])
        P[n] = _solve(gts[key], X[n], G[n], CONFIGS[config])
    for j, name in enumerate(COLS):
        p, f = P[:, j], F[:, j]
        if name in ("discrete", "flags"):
            bad = np.nonzero(p != f)[0]
        elif name in ("Zmean", "Zstar"):
            bad = np.nonzero(np.abs(p - f) > REL * np.maximum(np.abs(f), 1e-3))[0]
        elif name.startswith("alpha"):
            tiny = np.abs(f) < 1e-100
            lnf = np.log(np.maximum(f, 1e-320))
            err = np.where(tiny, np.abs(np.log(np.maximum(p, 1e-320)) - lnf) / np.abs(lnf),
                           np.abs(p - f) / np.maximum(np.abs(f), 1e-300))
            bad = np.nonzero(np.where(p == f, 0.0, err) > REL)[0]
        else:
            bad = np.nonzero(np.where(p == f, 0.0, np.abs(p - f) / np.maximum(np.abs(f), 1e-300)) > REL)[0]
        assert bad.size == 0, (config, name, [(tuple(S[i]), p[i], f[i]) for i in bad[:3]])
