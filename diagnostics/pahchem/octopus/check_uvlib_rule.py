#!/usr/bin/env python3
"""How good is the size and charge rule of ``pycalima.pahchem.uvlib``? Per-carbon absorption rates (0.1-13.6 eV) in four fields:

A. structure: the mixed anchors over the compact ones at the sizes that have isomers (C24, C32, C42, C54, C96, C150);
B. size: each interior anchor predicted by log-linear interpolation between its two neighbours (compact anchors, whose C42 is itself
   interpolated and so left out), over the actual one;
C. charge: the dication over the cation (what the cation would cost as a stand-in for the dication; every anchor is now computed).

    PYTHONPATH=diagnostics/pahchem python diagnostics/pahchem/octopus/check_uvlib_rule.py
"""
import numpy as np

from pycalima.pahchem import fields, uvlib

t = uvlib.library()
E = t["energy_ev"]
k = E <= 13.6
FIELDS = {"Mathis": fields.mathis1983(1.0, e_min=0.1, n=2000), "Draine 1978": fields.draine1978(1.0),
          "Kurucz 10 kK": fields.kurucz(10000, 1.0), "blackbody 6000 K": fields.blackbody(6000, 1.0, e_min=0.1)}
SIZES = list(t["nc"].astype(int))
COMPUTED = [n for n in SIZES if n != 42]      # compact C42 is an interpolation, not a calculation
CHARGES = list(t["charges"])


def anchor(size, z, structure="compact", positions="shifted"):
    return t["sigma"][uvlib.POSITIONS.index(positions), uvlib.STRUCTURES.index(structure), SIZES.index(size), CHARGES.index(z)].astype(float)


def rate(f, s):
    e = f.energy_ev
    return np.trapezoid(np.clip(np.interp(e, E[k], s[k], left=0.0, right=0.0), 0.0, None) * f.flux_at(e), e)


def line(label, a, b):
    print(f"  {label:34s}" + "  ".join(f"{n} {rate(f, a) / rate(f, b):.3f}" for n, f in FIELDS.items()))


print("A. mixed over compact anchors, per-carbon rates")
for n in (24, 32, 42, 54, 96, 150):
    for z in (-1, 0, 1, 2):
        line(f"N_C {n} z {z:+d}", anchor(n, z, "mixed"), anchor(n, z, "compact"))
print("B. interpolated from the two neighbours over the actual anchor (compact)")
for i in range(1, len(COMPUTED) - 1):
    lo, mid, hi = COMPUTED[i - 1], COMPUTED[i], COMPUTED[i + 1]
    w = (np.log(mid) - np.log(lo)) / (np.log(hi) - np.log(lo))
    for z in (-1, 0, 1):
        line(f"N_C {mid} from {lo}/{hi} z {z:+d}", (1 - w) * anchor(lo, z) + w * anchor(hi, z), anchor(mid, z))
print("C. dication over cation, mixed anchors (the cost the cation would have as a stand-in; the library no longer uses it)")
for n in (54, 96, 150):
    line(f"N_C {n}", anchor(n, 2, "mixed"), anchor(n, 1, "mixed"))
