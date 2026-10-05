"""Laboratory data the pahchem rates are validated against.

- ``west2018_table1`` / ``jochims94_table(n)``: hand-transcribed tables bundled
  in ``data/external_data/pahchem/`` (see each file's header).
- ``west2018_esi``: the frequencies and RRKM k(E) of West et al. (2018) ESI
  Table S1. It is a manual dataset (``calima-fetch-data import west2018-esi``),
  read here with the standard library so that no Excel reader is needed.
"""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pycalima import _paths

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def west2018_table1() -> pd.DataFrame:
    """West+18 Table 1 (iPEPICO): E0 [eV] and dS(1000 K) [J/K/mol] for R1-R25."""
    return pd.read_csv(_paths.get_external_data_path("pahchem", "west2018_table1.csv"),
                       comment="#").set_index("reaction")


def jochims94_table(n: int) -> pd.DataFrame:
    """Jochims+94 Table 1 (appearance potentials) or Table 2 (RRK thresholds)."""
    if n not in (1, 2):
        raise ValueError("Jochims et al. (1994) has Tables 1 and 2")
    return pd.read_csv(_paths.get_external_data_path("pahchem", f"jochims94_table{n}.csv"),
                       comment="#").set_index("molecule")


def electron_recombination() -> pd.DataFrame:
    """Measured PAH-cation electron recombination at ~300 K (C6-C16)."""
    return pd.read_csv(_paths.get_external_data_path("pahchem", "pah_electron_recombination.csv"),
                       comment="#")


@dataclass(frozen=True)
class WestReaction:
    """One reaction of West+18 ESI Table S1."""

    number: int
    parent: str
    ts_label: str
    reactant_freqs_cm: np.ndarray
    ts_freqs_cm: np.ndarray
    energy_cm: np.ndarray        # ion internal energy grid
    k_s: np.ndarray              # their RRKM k(E) [s^-1]


def _xlsx_sheets(path):
    """{sheet name: list of rows (lists of cell strings)}."""
    def col(ref):
        n = 0
        for ch in re.match(r"[A-Z]+", ref).group():
            n = n * 26 + ord(ch) - 64
        return n - 1

    sheets = {}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in rels}
        for sh in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", _NS):
            rid = sh.get(f"{{{_NS['r']}}}id")
            root = ET.fromstring(z.read("xl/" + target[rid].lstrip("/").removeprefix("xl/")))
            rows = []
            for row in root.iter(f"{{{_NS['m']}}}row"):
                cells = {}
                for c in row.findall("m:c", _NS):
                    v = c.find("m:v", _NS)
                    if v is not None:
                        cells[col(c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else v.text
                rows.append([cells.get(i, "") for i in range(max(cells, default=-1) + 1)])
            sheets[sh.get("name")] = rows
    return sheets


def _numbers(rows, j):
    out = []
    for r in rows:
        try:
            out.append(float(r[j]))
        except (IndexError, ValueError):
            pass
    return np.array(out)


def west2018_esi(path=None) -> dict[int, WestReaction]:
    """West+18 ESI Table S1, one ``WestReaction`` per reaction number.

    The "rovib frequencies" sheet has one column per species (M, M+, M-H+ and
    one per transition state), with the reaction number above each TS column;
    the reactant is M-H+ for TS labels starting "TS-(M-H)" and M+ otherwise.
    The "rates" sheet has an (energy [cm^-1], k [s^-1]) column pair per reaction.
    """
    if path is None:
        from pycalima._datasets import ensure_dataset

        path = ensure_dataset("west2018-esi") / "c7cp07369k2_suppl.xlsx"
    sheets = _xlsx_sheets(path)
    fr = sheets["rovib frequencies"]
    mol_row, rxn_row, lab_row = fr[0], fr[1], fr[2]
    first = next(i for i, r in enumerate(fr) if r and r[0].startswith("vibrational"))
    freq_rows = fr[first:]
    width = max(len(r) for r in fr)

    def cell(row, j):
        return row[j] if j < len(row) else ""

    ts_cols, parent, species = {}, "", {}
    for j in range(1, width):
        if cell(mol_row, j):
            parent, species = cell(mol_row, j), {}
        species[cell(lab_row, j)] = j
        if cell(rxn_row, j):
            label = cell(lab_row, j)
            reactant = "M-H+" if label.startswith("TS-(M-H)") else "M+·"
            ts_cols[int(float(cell(rxn_row, j)))] = (parent, label, j, species[reactant])

    rates_rows = sheets["rates"][3:]
    out = {}
    for n, (par, label, j_ts, j_r) in sorted(ts_cols.items()):
        jj = 1 + 2 * (n - 1)
        pairs = [(float(r[jj]), float(r[jj + 1])) for r in rates_rows
                 if len(r) > jj + 1 and r[jj] and r[jj + 1]]
        E, k = np.array(pairs).T
        out[n] = WestReaction(n, par, label, _numbers(freq_rows, j_r), _numbers(freq_rows, j_ts), E, k)
    return out
