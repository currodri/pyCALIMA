"""Cross-checks of the cell's optics (A3c phase 4): ``gsd.optics.material_optics`` against independent references.

1. UV absorption of the covalent a-C(:H) bins: the background's 6.8-13.6 eV photons absorbed per C atom
   (``PseudoBinOptics``, at s = 0.5, on every band-gap node) against ``hydrogen.BinHydrogen.abs_per_c`` (its own
   profile weights and Mathis field, U = 1). The two fields differ by one normalisation (Mathis U = 1 against
   G0 = 1 Habing in 6-13.6 eV), so the ratio must be one constant across band gaps and bins.
2. Dust temperatures in the UV background at G0 = 1.7 Habing (about the Mathis ISRF) per bin at s = 0, 0.5, 1:
   large grains should sit near 15-20 K (THEMIS CM20 ~17-18 K, astrosilicate ~15-17 K at U = 1).
3. The covalent bins' equilibrium band gap in a CNM cell (G0 1.7, n(H I) 30 cm^-3, 100 K) from ``cell.precool``'s
   recipe against ``BinHydrogen.equilibrium`` in the same field (U from item 1's ratio).

    python diagnostics/gsd/check_cell_optics.py [--out results/gsd_a3c/cell_optics_check.txt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

M_C = 12.011 * 1.66053906660e-24


def main(argv=None):
    import run_cell as R

    from pycalima.gsd.hydrogen import Y_CH, Conditions, equilibrium_gap, k_h

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/gsd_a3c/cell_optics_check.txt"))
    a = ap.parse_args(argv)
    c, s_comp, lay = R.demo_compositions(1.0, optics=True)
    po, bh = c.optics, c.hydrogen
    lines = ["Cell optics cross-checks (gsd.optics.material_optics; diagnostics/gsd/check_cell_optics.py)", "",
             "1. covalent bins: background UV photons absorbed per C (G0 = 1 Habing) / BinHydrogen.abs_per_c (U = 1)"]
    half = np.full(c.grid.n_bins, 0.5)
    uv = po.uv_photons(half, None, 1.0) * M_C                       # [gap bin, node], per C at G0 = 1
    ratios = []
    for j, b in enumerate(bh):
        r = uv[j] / np.interp(po.gap_nodes, b.e_grid, b.abs_per_c)
        ratios.append(r)
        lines.append(f"   bin {po.gap_bins[j]} ({c.grid.a_edges[po.gap_bins[j]] * 1e7:.2f}-"
                     f"{c.grid.a_edges[po.gap_bins[j] + 1] * 1e7:.2f} nm): ratio {r.min():.4f}-{r.max():.4f}")
    allr = np.concatenate(ratios)
    g_per_u = 1.0 / float(np.median(allr))
    lines.append(f"   all: {allr.min():.4f}-{allr.max():.4f} (spread {allr.max() / allr.min() - 1:.2%}); "
                 f"G0 per Mathis U = {g_per_u:.3f}")
    lines += ["", "2. T_d [K] in the UV background, G0 = 1.7 (bins smallest first; molecular and overflow bins left out)"]
    mb = lay.n_pah_bins
    for name, comp, sl in (("carbon", c, slice(mb, c.grid.n_bins - 1)), ("silicate", s_comp, slice(0, -1))):
        for sv in (0.0, 0.5, 1.0):
            s = np.full(comp.grid.n_bins, sv)
            gap = None
            if comp.optics.gap is not None:
                gap = np.full(comp.optics.gap_bins.size, 0.1)
            _, power, _ = comp.optics.absorption(s, None, 1.7, gap)
            T = comp.optics.dust_temperature(s, power, gap)
            lines.append(f"   {name:8s} s = {sv:.1f}: " + " ".join(f"{t:5.1f}" for t in T[sl])
                         + ("   (covalent bins at E_g 0.1 eV)" if gap is not None else ""))
    lines += ["", "3. CNM band gap (G0 1.7, n(H I) 30, 100 K): precool recipe against BinHydrogen.equilibrium"]
    G0, n_hi, T = 1.7, 30.0, 100.0
    kh = k_h(n_hi, T)
    uvc = po.uv_photons(half, None, G0) * M_C
    for j, b in enumerate(bh):
        e_cell = equilibrium_gap(b, Y_CH * np.interp(b.e_grid, po.gap_nodes, uvc[j]), kh)[1]
        x_ref = b.equilibrium(Conditions(G0 / g_per_u, n_hi, T))
        lines.append(f"   bin {po.gap_bins[j]}: cell {e_cell:.3f} eV, BinHydrogen {float(b.e_gap(x_ref)):.3f} eV")
    txt = "\n".join(lines)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
