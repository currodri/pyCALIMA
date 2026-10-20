"""Murga+2022 model b (H loss and addition) through pahchem.murga2022, against Figshare.

Each gate isolates one input:
- the output times, from the files alone;
- the photo-loss rate law, on the deep-MC loss chains where H addition is
  negligible;
- the small-PAH dehydrogenation, region by region;
- the super-H balance of the larger PAHs (strict xfail).

Needs the murga2022-figshare dataset. The photo-loss table is cached next
to it, so the first run takes ~3 min.
"""

import numpy as np
import pandas as pd
import pytest

from pycalima._paths import get_external_data_path
from pycalima.pahchem import murga2022 as N
from pycalima.pahchem import murga_data as M
from pycalima.pahchem import shiva

DEEP_MC = range(134, 152)          # 3.3 A neutral loss chains (H addition negligible at T = 13 K)
_STATE = {}


@pytest.fixture(scope="module")
def data():
    from pycalima._datasets import DatasetUnavailable, ensure_dataset

    try:
        ensure_dataset("murga2022-figshare")
    except DatasetUnavailable:
        pytest.skip("murga2022-figshare not fetched (calima-fetch-data fetch murga2022-figshare)")
    return M.read_model_2022("b")


def _profile():
    return pd.read_csv(get_external_data_path("pahchem/murga2022_fig6_profile.csv"), comment="#")


def _chain(d, loc):
    """(k1, k2) of the normal -> mid -> lowest loss chain of 3.3 A neutrals at ``loc``."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics" / "pahchem"))
    from check_murga2022 import fit_chain

    nn = d["n"][loc, 0, :, :, 0]
    return fit_chain(nn / nn.sum(axis=1, keepdims=True))


def test_output_times_are_linear(data):
    """The 3.3 A neutral normal fraction in the deep MC falls by the same factor
    at every output: an exponential sampled at equal intervals. With the age of
    1e5 yr (Sect. 4.1) the outputs are 0, 1e4, ..., 1e5 yr (measured spread of the
    per-output ratio < 1 % at locations 140-147)."""
    for loc in range(140, 148):
        nn = data["n"][loc, 0, :, 2, 0] / data["n"][loc, 0, :, :, 0].sum(axis=1)
        ratio = nn[2:] / nn[1:-1]
        assert np.ptp(ratio[ratio > 0]) / ratio.mean() < 0.02, (loc, np.round(ratio, 4))


@pytest.mark.parametrize("temperature, ok", [("dl01", True), ("tielens", False)])
def test_photo_loss_hydrogen_dependence_picks_the_dl01_temperature(data, temperature, ok):
    """k(mid -> lowest)/k(normal -> mid) of the deep-MC chains is 1.250 at every chi;
    it is the N_H dependence of the photo-loss rate, free of any rate-to-flux
    convention. Ours: 1.257 with the DL01 T(E), 0.787 with Tielens' (1.278 and
    0.835 without the C2H2 hydrogen)."""
    k = np.array([_chain(data, loc)[0] for loc in DEEP_MC])
    theirs = k[:, 1] / k[:, 0]
    assert np.allclose(theirs, 1.25, atol=0.01)
    nc = N.nc_of(data["a_cm"][0])
    R = [N.photo_loss_on_grid(nc, j, 0, temperature, chi_grid=[1e-3])[0] for j in (1, 2)]
    assert (abs(R[0] / R[1] - 1.25) < 0.05) == ok, R[0] / R[1]


@pytest.mark.parametrize("acetylene_h, lo, hi", [(True, 0.6, 1.3), (False, 1.1, 2.3)])
def test_deep_mc_step_rate_and_its_rise_with_depth(data, acetylene_h, lo, hi):
    """Their absolute step rate over ours (DL01, dN_H = the 0.5 N_H0 flux step of
    the files, ours single-photon, so proportional to chi). With R_H counting
    the H of the C2H2 channel (M22 Sect. 3.1: R_H is "the net rate of detachment
    of all hydrogen atoms which might be detached ... both separately and also
    inside hydrogen or acetylene molecules"): 0.70 at chi = 1.9e-3 rising to
    1.16 at chi = 1.2e-5. Counting H and H2 only: 1.22 to 2.02. (Before the
    2026-10-20 audit fixes C2 and C3: 0.87-1.44 and 1.52-2.51.) Their rate per
    unit chi grows with depth, which a field shape that changes with depth
    (not stated in M22) would give."""
    nc = N.nc_of(data["a_cm"][0])
    step = N.DX_FLUX * shiva.dl01_nh(nc)
    r1 = N.photo_loss_on_grid(nc, 2, 0, "dl01", chi_grid=[1e-3], acetylene_h=acetylene_h)[0] / 1e-3 / step
    ratio = np.array([_chain(data, loc)[0][0] / (r1 * data["chi"][loc]) for loc in DEEP_MC])
    assert np.all((ratio > lo) & (ratio < hi)), np.round(ratio, 2)
    assert np.corrcoef(np.arange(ratio.size), ratio)[0, 1] > 0.95          # rises with depth


@pytest.fixture(scope="module")
def model_b(data):
    if "b" not in _STATE:
        R = N.photo_loss_table(data["a_cm"], "dl01")
        _STATE["b"] = N.run_model_b(data, _profile(), R)
    return _STATE["b"]


def _regional(data, ours, size, region, t=-1):
    from pycalima.pahchem.murga_charge import region_of

    reg = np.array([region_of(x) for x in _profile().distance_arcsec])
    m = reg == region
    n = data["n"][m][:, :, t, :, size]                     # loc, Z, bin
    tot = n.sum(axis=2, keepdims=True)
    th = np.where(tot > 0, n / np.where(tot > 0, tot, 1), 0.0)
    w = (data["f"][m][:, :, 0, 0, size] * (data["n"][m][:, :, 0, :, size].sum(axis=2) > 0))[:, :, None]
    return (ours[m][:, :, t, :, size] * w).sum(axis=(0, 1)) / w.sum(), (th * w).sum(axis=(0, 1)) / w.sum()


@pytest.mark.parametrize("region", ["PDR 1", "PDR 2", "DF", "MC"])
def test_small_pahs_are_dehydrogenated_as_in_figshare(data, model_b, region):
    """3.3 A after 1e5 yr, f-weighted lowest-bin fraction, ours / Figshare:
    PDR 1 0.98 / 0.95, PDR 2 0.92 / 0.93, DF 1.00 / 1.00, MC 0.85 / 0.86
    (without the C2H2 hydrogen in R_H: 0.96, 0.86, 1.00, 0.83)."""
    ours, theirs = _regional(data, model_b, 0, region)
    assert ours[0] == pytest.approx(theirs[0], abs=0.05)


@pytest.mark.parametrize("region", ["PDR 1", "DF", "MC", pytest.param("PDR 2", marks=pytest.mark.xfail(strict=True, reason=(
    "PDR 2 keeps <X_H/X_H0> +0.12 to +0.17 at 6-8.7 A (+0.05 to +0.08 at 11.7-14.2 A): the top bin of the "
    "neutrals fills where Figshare's stays empty, so their loss from bin 4 is >= 1e3x ours there; the files "
    "do not show why (pahchem-audit C3).")))])
def test_larger_pahs_stay_normal_as_in_figshare(data, model_b, region):
    """>= 6 A stay normal as in the Figshare model b. Holds with the addition and
    loss laws read from the files (``murga2022.h_addition(law="figshare")``,
    DX_FLUX, 2-H Eley-Rideal; pahchem-audit C3). With M20 as printed
    (``law="m20"``) they went +0.3 to +0.5 super-hydrogenated."""
    for size in range(2, 9):                              # 6 - 19.9 A
        ours, theirs = _regional(data, model_b, size, region)
        assert abs(((ours - theirs) * N.X_BINS).sum()) < 0.1, (region, size)
