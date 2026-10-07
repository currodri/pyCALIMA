"""Species from a NASA Ames PAHdb theoretical ASCII export (v3.20).

For benchmarks defined on a PAHdb sample, such as the 308 species of
Maragkoudakis, Peeters & Ricca (2020). PAHdb needs a registration and is never
bundled; the file is the manual dataset ``pahdb-theoretical-v3-20``, found
through ``$CALIMA_PAHDB_DIR``.

As in ``PAHSpecies.from_states_file``, FREQUENCY is used as written: PAHdb
stores frequencies already scaled, and SCALE records the factor applied.
"""

from __future__ import annotations

import re

import numpy as np

from pycalima.pahchem.molecule import _EINSTEIN_A_PER_KM_MOL, PAHSpecies

_FORMULA = re.compile(r"^C(\d+)(?:H(\d+))?$")


def ascii_path():
    """Path of the v3.20 ASCII export (raises with instructions if it is not installed)."""
    from pycalima._datasets import ensure_dataset

    return ensure_dataset("pahdb-theoretical-v3-20") / "pahdb-complete-theoretical-v3.20_pIN3f5.ascii"


def read_ascii(path, uids) -> dict:
    """{uid: PAHSpecies} for ``uids`` from a PAHdb ASCII export.

    Only pure C/H species are supported; a requested UID that is missing, or
    that holds other elements, raises.
    """
    wanted, out = set(int(u) for u in uids), {}
    uid = formula = charge = None
    freq, inten, in_trans = [], [], False

    def flush():
        if uid in wanted:
            m = _FORMULA.match((formula or "").rstrip("+-"))
            if not m:
                raise ValueError(f"PAHdb UID {uid}: {formula} is not a pure C/H species")
            f = np.array(freq)
            out[uid] = PAHSpecies(int(m[1]), int(m[2] or 0), int(charge), f,
                                  _EINSTEIN_A_PER_KM_MOL * f**2 * np.array(inten),
                                  f"PAHdb v3.20 ASCII, UID {uid}")

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("UID:"):
                flush()
                uid, formula, charge, freq, inten, in_trans = int(line.split()[1]), None, None, [], [], False
            elif uid not in wanted:
                continue
            elif line.startswith("FORMULA:"):
                formula = line.split()[1]
            elif line.startswith("CHARGE:"):
                charge = int(line.split()[1])
            elif line.startswith("# TRANSITIONS:"):
                in_trans = True
            elif in_trans and line.startswith("#"):
                if not line.startswith("#    "):
                    in_trans = False
            elif in_trans and line.strip():
                t = line.split()
                freq.append(float(t[0]))
                inten.append(float(t[1]))
        flush()
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"PAHdb UIDs not found: {sorted(missing)[:10]}")
    return out


#: header fields of the ASCII export read by ``read_metadata``
_META_FIELDS = {"FORMULA": str, "CHARGE": int, "METHOD": str, "TOTAL_E": float, "N_SOLO": int, "N_DUO": int,
                "N_TRIO": int, "N_QUARTET": int, "N_QUINTET": int, "N_CH2": int, "N_CHX": int}


def read_metadata(path, uids) -> dict:
    """{uid: {field: value}} of the header fields in ``_META_FIELDS`` (TOTAL_E in hartree).

    The edge-hydrogen counts (N_SOLO ... N_QUINTET) count H atoms, as in PAHdb.
    BASIS is the basis set named in the first Gaussian route line of COMMENTS
    ("4-31G", "6-31G*", ...), or "?" where none is named (e.g. "chkbas").
    Total energies are comparable only at one functional (METHOD without its
    R/U prefix) and basis.
    """
    wanted, out, uid, routed = set(int(u) for u in uids), {}, None, set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("UID:"):
                uid = int(line.split()[1])
                if uid in wanted:
                    out[uid] = {"BASIS": "?"}
            elif uid in wanted and line.startswith("#    # ") and uid not in routed:
                routed.add(uid)
                basis = line[7:].split()[0].partition("/")[2] if line[7:].strip() else ""
                if re.fullmatch(r"\d-\d+\+*G[*()dp,]*", basis):
                    out[uid]["BASIS"] = basis
            elif uid in wanted and ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                if key.strip() in _META_FIELDS and value.strip():
                    out[uid][key.strip()] = _META_FIELDS[key.strip()](value.split()[0])
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"PAHdb UIDs not found: {sorted(missing)[:10]}")
    return out
