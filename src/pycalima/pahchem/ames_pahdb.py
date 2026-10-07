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
#: a Pople basis as written in a Gaussian route line, any case ("4-31G", "6-31g*", "6-31+G(d)")
_BASIS = re.compile(r"\d-\d+\+*G[*()dp,]*", re.IGNORECASE)


def _normal_basis(basis):
    return basis.upper().replace("(D)", "(d)").replace("(D,P)", "(d,p)")


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
                if _BASIS.fullmatch(basis):
                    out[uid]["BASIS"] = _normal_basis(basis)
            elif uid in wanted and ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                if key.strip() in _META_FIELDS and value.strip():
                    out[uid][key.strip()] = _META_FIELDS[key.strip()](value.split()[0])
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"PAHdb UIDs not found: {sorted(missing)[:10]}")
    return out


# -- XML libraries (theoretical v4.00, clusters, anharmonic, experimental) --------------------------
#: integer header fields of an XML specie
_XML_INT = ("charge", "n_solo", "n_duo", "n_trio", "n_quartet", "n_quintet", "n_ch", "n_ch2", "n_ch3")
_ELEMENT = {1: "H", 6: "C", 7: "N", 8: "O", 12: "Mg", 14: "Si", 26: "Fe"}


def xml_path(dataset="pahdb-theoretical-v4-00"):
    """Path of a PAHdb XML library (raises with instructions if it is not installed)."""
    from pycalima._datasets import ensure_dataset, get_dataset

    return ensure_dataset(dataset) / get_dataset(dataset).files[0]


def _basis_from_route(text):
    basis = text.lstrip("# ").split()[0].partition("/")[2] if text.strip("# ") else ""
    return _normal_basis(basis) if _BASIS.fullmatch(basis) else "?"


def bond_graph_hash(types, xyz, rounds=6, cc_max=1.75, ch_max=1.25):
    """A Weisfeiler-Lehman hash of the bond graph of one structure.

    Atoms are bonded below ``cc_max`` (heavy atoms) or ``ch_max`` (to H) [A].
    The graph, not the geometry, is hashed, so every charge state of one
    molecule shares it while isomers (same formula and edge counts, other
    rings) do not. Molecules of a van der Waals cluster are not bonded to each
    other.
    """
    import hashlib

    t = np.asarray(types, dtype=int)
    x = np.asarray(xyz, dtype=float)
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    either_h = (t[:, None] == 1) | (t[None, :] == 1)
    bonded = (d < np.where(either_h, ch_max, cc_max)) & ~np.eye(len(t), dtype=bool) & ~((t[:, None] == 1) & (t[None, :] == 1))
    labels = [str(v) for v in t]
    for _ in range(rounds):
        labels = [hashlib.sha1((labels[i] + "|" + ",".join(sorted(labels[j] for j in np.flatnonzero(bonded[i])))
                                ).encode()).hexdigest()[:16] for i in range(len(t))]
    return hashlib.sha1(",".join(sorted(labels)).encode()).hexdigest()[:20]


def read_xml(path, select=None, geometry_hash=True):
    """[record] of every specie in a PAHdb XML library (v4.00, clusters, anharmonic).

    A record is a dict: ``uid`` (None for clusters), the cluster attributes
    (``monomers``, ``type``, ``conformation``), ``formula`` (from the file, else
    from the geometry), the integers of ``_XML_INT``, ``method``, ``basis``,
    ``total_e`` [hartree], ``nc``, ``nh``, ``pure_ch``, ``freqs_cm`` and
    ``intensity`` (km/mol), and, with ``geometry_hash``, ``graph`` (the
    ``bond_graph_hash``). Frequencies are as written: already scaled, as in
    the ASCII export. ``select(record)`` (without modes) keeps a specie.
    Laboratory spectra (``<laboratory>``) are not read here.
    """
    import xml.etree.ElementTree as ET

    out = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag.rsplit("}", 1)[-1] != "specie":
            continue
        rec = {"uid": int(el.get("uid")) if el.get("uid") else None,
               "monomers": el.get("monomers"), "type": el.get("type"), "conformation": el.get("conformation"),
               "basis": "?", "method": None, "total_e": None, "formula": None}
        for k in _XML_INT:
            rec[k] = 0
        types, xyz, freqs, inten = [], [], [], []
        for c in el:
            tag = c.tag.rsplit("}", 1)[-1]
            text = (c.text or "").strip()
            if tag == "comments":
                for cm in c:
                    if (cm.text or "").lstrip().startswith("#") and rec["basis"] == "?":
                        rec["basis"] = _basis_from_route(cm.text.strip())
            elif tag in _XML_INT and text:
                rec[tag] = int(text)
            elif tag == "formula":
                rec["formula"] = text
            elif tag == "method":
                rec["method"] = text
            elif tag == "total_e" and text:
                rec["total_e"] = float(text)
            elif tag == "geometry":
                for atom in c:
                    v = {a.tag.rsplit("}", 1)[-1]: a.text for a in atom}
                    types.append(int(v["type"]))
                    xyz.append((float(v["x"]), float(v["y"]), float(v["z"])))
            elif tag == "transitions":
                for mode in c:
                    v = {a.tag.rsplit("}", 1)[-1]: a.text for a in mode}
                    freqs.append(float(v["frequency"]))
                    inten.append(float(v["intensity"]))
        counts = {}
        for t in types:
            counts[_ELEMENT.get(t, str(t))] = counts.get(_ELEMENT.get(t, str(t)), 0) + 1
        rec["nc"], rec["nh"] = counts.get("C", 0), counts.get("H", 0)
        rec["pure_ch"] = bool(types) and set(counts) <= {"C", "H"}
        if rec["formula"] is None and types:
            rec["formula"] = "".join(f"{e}{counts[e]}" for e in ("C", "H") if e in counts) + "".join(
                f"{e}{n}" for e, n in sorted(counts.items()) if e not in ("C", "H"))
        if geometry_hash and types:
            rec["graph"] = bond_graph_hash(types, xyz)
        if select is None or select(rec):
            rec["freqs_cm"], rec["intensity"] = np.array(freqs), np.array(inten)
            out.append(rec)
        el.clear()
    return out


#: bump when read_xml changes what it returns, so cached indexes are rebuilt
_XML_INDEX_VERSION = 2


def xml_index(path, cache_dir=None):
    """``read_xml`` of the whole file, cached as a pickle in ``cache_dir`` (keyed by size, mtime and reader version)."""
    import os
    import pickle
    from pathlib import Path

    path = Path(path)
    st = os.stat(path)
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{path.stem}-{st.st_size}-{int(st.st_mtime)}-r{_XML_INDEX_VERSION}.pkl"
        if cache.exists():
            with open(cache, "rb") as fh:
                return pickle.load(fh)
    recs = read_xml(path)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(recs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return recs


def species_from_record(rec, source):
    """``PAHSpecies`` of a pure C/H ``read_xml`` record."""
    if not rec["pure_ch"]:
        raise ValueError(f"{source}: {rec['formula']} is not a pure C/H species")
    f = rec["freqs_cm"]
    return PAHSpecies(rec["nc"], rec["nh"], int(rec["charge"]), f, _EINSTEIN_A_PER_KM_MOL * f**2 * rec["intensity"],
                      source)
