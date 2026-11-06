"""pahchem's pyrene cascades against measured IR emission of UV-excited pyrene (Saykally's laboratory, as replotted by
de Bentzmann+26): gates at the 2026-10-21 level, rms over the lab points / lab peak with a free scale and offset.
Needs PAHdb v4.00, the CosmicPAH-IRDB laws and the Mackie+18 supplement; skipped without them."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pycalima import _paths
from pycalima._datasets import get_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def check():
    for name in ("pahdb-theoretical-v4-00", "cosmicpah-irdb", "mackie2018-supplement"):
        if get_dataset(name).locate() is None:
            pytest.skip(f"{name} not available")
    sys.path.insert(0, str(ROOT / "diagnostics" / "pahchem"))
    import check_saykally_pyrene as C

    grids = {b: np.linspace(v[0], v[1], 1200) for b, v in C.BANDS.items()}
    return C, grids, C.models(grids)


@pytest.mark.parametrize("band,model,limit", [("3.3", "lab laws", 0.09), ("12", "lab laws", 0.06),
                                              ("3.3", "VPT2 + polyads", 0.07)])
def test_pyrene_emission_matches_saykally(check, band, model, limit):
    C, grids, mods = check
    d = pd.read_csv(_paths.get_external_data_path("pahchem", "saykally_pyrene_emission.csv"), comment="#")
    lab = d[np.isclose(d.band_um, float(band)) & (d.kind == "lab")]
    rms, _ = C.scored(mods[band][model], grids[band], lab.nu_cm.to_numpy(), lab.intensity.to_numpy())
    assert rms < limit
