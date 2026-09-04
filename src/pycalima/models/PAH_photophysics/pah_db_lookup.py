"""
pah_db_lookup.py — PAHdb species lookup and vibrational mode file manager.

Implements the Andrews (2016) fallback strategy:
  1. If vibrational modes for (Nc, Nh, Z) exist on disk, return them.
  2. Otherwise find the closest Nh within the same (Nc, Z) from the catalog,
     extract transitions directly from the XML, write a .dat file, and return it.

Main entry point
----------------
    path, exact, note = get_modes_file(Nc, Nh, Z, states_dir, catalog, xml_path)

Catalog management
------------------
    build_pahdb_catalog(xml_path, catalog_path)   — build once from XML
    load_pahdb_catalog(catalog_path)              — load the JSON catalog

Low-level helpers
-----------------
    find_best_species(catalog, Nc, Nh, Z)         — lookup with Nh fallback
    extract_modes_by_uid(xml_path, uid, ...)       — write .dat from XML
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Regex helpers for XML parsing
# ---------------------------------------------------------------------------
_UID_RE      = re.compile(r'<specie uid="(\d+)">')
_FORMULA_RE  = re.compile(r'<formula>(C\d+(?:H\d+)?[+\-]?(?:\d+)?)</formula>')
_CHARGE_RE   = re.compile(r'<charge>(-?\d+)</charge>')
_SYM_RE      = re.compile(r'<symmetry>([^<]+)</symmetry>')
_SOLO_RE     = re.compile(r'<n_solo>(\d+)</n_solo>')
_DUO_RE      = re.compile(r'<n_duo>(\d+)</n_duo>')
_MULT_RE     = re.compile(r'^(\d+)-')
_FORMULA_PARSE = re.compile(r'^C(\d+)(H(\d+))?([+\-]?\d*)?$')

# XML transition fields
_FREQ_RE     = re.compile(r'<frequency scale="([^"]+)">([^<]+)</frequency>')
_INTENS_RE   = re.compile(r'<intensity>([^<]+)</intensity>')
_TSYM_RE     = re.compile(r'<symmetry>([^<]+)</symmetry>')


# ---------------------------------------------------------------------------
# Catalog building
# ---------------------------------------------------------------------------

def build_pahdb_catalog(xml_path: str, catalog_path: str,
                        nc_values: list = None,
                        max_multiplicity: int = 2) -> list:
    """
    Parse the PAHdb XML and write a JSON catalog of CnHm species
    (optionally filtered to nc_values).

    By default includes singlets (mult=1) and doublets (mult=2).
    Doublets are odd-H radicals computed with UB3LYP in PAHdb — they
    are physically correct species needed for dehydrogenation sequences
    (e.g. C96H23, C96H25 neutral).  Set max_multiplicity=1 to revert
    to the old singlet-only behaviour.

    The catalog entry includes a ``mult`` field (1 or 2) so callers can
    distinguish closed-shell from open-shell species.

    Parameters
    ----------
    xml_path         : path to pahdb-complete-theoretical-*.xml
    catalog_path     : output JSON path (e.g. model_data/PAH_states/pahdb_catalog.json)
    nc_values        : list of Nc to include, e.g. [24, 54, 96]. None = all PAHs.
    max_multiplicity : maximum spin multiplicity to include (default 2 = singlets + doublets).

    Returns
    -------
    List of catalog entry dicts (same as what load_pahdb_catalog returns).
    """
    nc_set = set(nc_values) if nc_values else None

    entries = []
    buffer  = []
    cur_uid = None

    with open(xml_path, 'r', errors='replace') as f:
        for line in f:
            m = _UID_RE.search(line)
            if m:
                cur_uid = m.group(1)
                buffer  = [line]
                continue
            if '</specie>' in line:
                text = ''.join(buffer)
                entry = _parse_specie_block(text, cur_uid, nc_set, max_multiplicity)
                if entry is not None:
                    entries.append(entry)
                buffer = []
                continue
            buffer.append(line)

    os.makedirs(os.path.dirname(os.path.abspath(catalog_path)), exist_ok=True)
    with open(catalog_path, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"Catalog built: {len(entries)} usable species → {catalog_path}")
    return entries


def _parse_specie_block(text: str, uid: str, nc_set,
                        max_multiplicity: int = 2) -> dict | None:
    """Return a catalog entry dict, or None if the species should be excluded."""
    fm = _FORMULA_RE.search(text)
    if not fm:
        return None
    formula = fm.group(1)

    pm = _FORMULA_PARSE.match(formula)
    if not pm:
        return None
    Nc = int(pm.group(1))
    if nc_set is not None and Nc not in nc_set:
        return None
    Nh = int(pm.group(3)) if pm.group(3) else 0

    cm = _CHARGE_RE.search(text)
    sm = _SYM_RE.search(text)
    if not cm or not sm:
        return None
    charge   = int(cm.group(1))
    symmetry = sm.group(1).strip()

    mm = _MULT_RE.match(symmetry)
    if not mm:
        return None
    mult = int(mm.group(1))
    if mult > max_multiplicity:
        return None  # triplet or higher — exclude

    som = _SOLO_RE.search(text)
    dum = _DUO_RE.search(text)
    n_solo = int(som.group(1)) if som else None
    n_duo  = int(dum.group(1)) if dum else None

    return {
        "uid":      int(uid),
        "Nc":       Nc,
        "Nh":       Nh,
        "Z":        charge,
        "formula":  formula,
        "symmetry": symmetry,
        "mult":     mult,
        "n_solo":   n_solo,
        "n_duo":    n_duo,
    }


def load_pahdb_catalog(catalog_path: str) -> list:
    """Load the JSON catalog built by build_pahdb_catalog."""
    with open(catalog_path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Species lookup
# ---------------------------------------------------------------------------

def find_best_species(catalog: list, Nc: int, Nh: int, Z: int) -> tuple:
    """
    Find the best available PAHdb species for a requested (Nc, Nh, Z).

    Strategy (Andrews 2016):
      1. Exact match on (Nc, Nh, Z) — use the highest-symmetry isomer (first
         entry with the most duo H, as proxy for the most compact structure).
      2. No exact Nh: nearest Nh within same (Nc, Z), preferring the isomer
         whose solo+duo topology is closest to the fully-hydrogenated parent.

    Parameters
    ----------
    catalog : list of dicts from load_pahdb_catalog
    Nc, Nh, Z : requested species

    Returns
    -------
    (entry, is_exact, note) where note explains any fallback applied.
    Returns (None, False, reason) if no usable species found.
    """
    pool = [e for e in catalog if e["Nc"] == Nc and e["Z"] == Z]
    if not pool:
        return None, False, f"No species with Nc={Nc} Z={Z:+d} in catalog"

    # Exact Nh match
    exact = [e for e in pool if e["Nh"] == Nh]
    if exact:
        chosen = _pick_canonical(exact)
        return chosen, True, f"exact match uid={chosen['uid']}"

    # Fallback: nearest Nh, prefer lower (dehydrogenated side) on tie
    pool_sorted = sorted(pool, key=lambda e: (abs(e["Nh"] - Nh), -e["Nh"]))
    chosen = _pick_canonical([e for e in pool_sorted
                               if abs(e["Nh"] - pool_sorted[0]["Nh"]) == 0])
    delta = chosen["Nh"] - Nh
    note  = (f"fallback: using C{Nc}H{chosen['Nh']} (uid={chosen['uid']}, "
             f"ΔNh={delta:+d}) for requested Nh={Nh}")
    return chosen, False, note


def _pick_canonical(entries: list) -> dict:
    """
    Among a set of isomers, prefer the one with the largest n_duo
    (most compact edge topology, closest to the symmetric parent).
    """
    return max(entries, key=lambda e: (e["n_duo"] or 0, e["n_solo"] or 0))


# ---------------------------------------------------------------------------
# Mode extraction from XML
# ---------------------------------------------------------------------------

def extract_modes_by_uid(xml_path: str, uid: int, output_path: str,
                         n_solo: int = None, n_duo: int = None) -> str:
    """
    Extract vibrational transitions for a single UID from the PAHdb XML
    and write a .dat file compatible with load_pah_modes.

    Parameters
    ----------
    xml_path    : path to the PAHdb XML
    uid         : integer UID to extract
    output_path : file path to write (e.g. model_data/PAH_states/C54H18_0.dat)
    n_solo, n_duo : override metadata (optional; read from XML if not provided)

    Returns
    -------
    output_path (the written file).
    """
    target_uid = str(uid)
    modes       = []
    formula     = charge = symmetry = None
    n_solo_xml  = n_duo_xml = None

    in_target = False
    in_trans  = False
    cur_freq = cur_scale = cur_intens = cur_sym = None

    with open(xml_path, 'r', errors='replace') as f:
        for line in f:
            if f'uid="{target_uid}">' in line:
                in_target = True
                continue

            if not in_target:
                continue

            if '</specie>' in line:
                break

            # Collect metadata
            if formula is None:
                m = _FORMULA_RE.search(line)
                if m:
                    formula = m.group(1)
            if charge is None:
                m = _CHARGE_RE.search(line)
                if m:
                    charge = int(m.group(1))
            if symmetry is None:
                m = _SYM_RE.search(line)
                if m:
                    symmetry = m.group(1).strip()
            if n_solo_xml is None:
                m = _SOLO_RE.search(line)
                if m:
                    n_solo_xml = int(m.group(1))
            if n_duo_xml is None:
                m = _DUO_RE.search(line)
                if m:
                    n_duo_xml = int(m.group(1))

            # Transitions block
            if '<transitions>' in line:
                in_trans = True
                continue
            if '</transitions>' in line:
                in_trans = False
                continue

            if in_trans:
                if '<mode>' in line:
                    cur_freq = cur_scale = cur_intens = cur_sym = None
                    continue
                if '</mode>' in line:
                    if cur_freq is not None and cur_intens is not None:
                        modes.append((uid, cur_freq, cur_scale or 1.0,
                                      cur_intens, cur_sym or ''))
                    continue
                m = _FREQ_RE.search(line)
                if m:
                    cur_scale = float(m.group(1))
                    cur_freq  = float(m.group(2))
                    continue
                m = _INTENS_RE.search(line)
                if m:
                    cur_intens = float(m.group(1))
                    continue
                m = _TSYM_RE.search(line)
                if m:
                    cur_sym = m.group(1).strip()

    if not modes:
        raise ValueError(f"No transitions found for uid={uid} in {xml_path}")

    ns = n_solo if n_solo is not None else (n_solo_xml or 0)
    nd = n_duo  if n_duo  is not None else (n_duo_xml  or 0)

    _write_dat(output_path, uid, formula or f"uid{uid}", charge or 0,
               symmetry or '?', ns, nd, modes)
    return output_path


def _write_dat(output_path: str, uid: int, formula: str, charge: int,
               symmetry: str, n_solo: int, n_duo: int, modes: list):
    """Write a .dat file in the NASA PAHdb transitions format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    with open(output_path, 'w') as f:
        f.write(f"\\ DATE     = '{now}'\n")
        f.write( "\\ ORIGIN   = 'NASA Ames Research Center'\n")
        f.write( "\\ TYPE     = 'TRANSITIONS'\n")
        f.write( "\\ SPECIES\n")
        f.write(f"\\N_SOLO     : {n_solo} \n")
        f.write(f"\\N_DUO      : {n_duo} \n")
        f.write( "| UID|FREQUENCY|INTENSITY| SCALE|SYMMETRY|\n")
        f.write( "|long|   double|   double|double|    char|\n")
        f.write( "|    |   1 / cm| km / mol|      |        |\n")
        f.write( "|null|     null|     null|  null|    null|\n")
        for uid_col, freq, scale, intens, sym in modes:
            f.write(f"{uid_col:5d} {freq:10.4f} {intens:11.4f} {scale:6.4f} {sym:>10s} \n")


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def get_modes_file(Nc: int, Nh: int, Z: int,
                   states_dir: str,
                   catalog: list,
                   xml_path: str) -> tuple:
    """
    Return the path to a .dat transitions file for (Nc, Nh, Z), creating it
    from the PAHdb XML if necessary.

    Implements the Andrews (2016) fallback: if no exact (Nc, Nh, Z) file
    exists, use the nearest Nh within the same (Nc, Z).

    Parameters
    ----------
    Nc, Nh, Z  : requested species
    states_dir  : directory holding .dat files (e.g. model_data/PAH_states/)
    catalog     : list returned by load_pahdb_catalog
    xml_path    : path to the PAHdb XML (needed only if file must be extracted)

    Returns
    -------
    (file_path, is_exact, note)
        file_path : str   — path to the .dat file (guaranteed to exist)
        is_exact  : bool  — True if the file is the exact requested species
        note      : str   — description of any fallback applied
    """
    def _dat_path(nc, nh, z):
        h_part = f"H{nh}" if nh > 0 else ""
        return os.path.join(states_dir, f"C{nc}{h_part}_{z}.dat")

    # 1. Check if exact file already exists on disk
    exact_path = _dat_path(Nc, Nh, Z)
    if os.path.exists(exact_path):
        return exact_path, True, "exact file on disk"

    # 2. Look up best available species in catalog
    entry, is_exact, note = find_best_species(catalog, Nc, Nh, Z)
    if entry is None:
        raise FileNotFoundError(
            f"No usable species for C{Nc}H{Nh} Z={Z:+d} in catalog. {note}"
        )

    # 3. Check if the fallback species' file is already on disk
    fallback_path = _dat_path(entry["Nc"], entry["Nh"], entry["Z"])
    if os.path.exists(fallback_path):
        return fallback_path, is_exact, note

    # 4. Extract from XML
    print(f"  Extracting uid={entry['uid']} ({entry['formula']}) → {fallback_path}")
    extract_modes_by_uid(xml_path, entry["uid"], fallback_path,
                         n_solo=entry["n_solo"], n_duo=entry["n_duo"])
    return fallback_path, is_exact, note


# ---------------------------------------------------------------------------
# CLI: build catalog
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PAHdb species catalog")
    parser.add_argument("--xml",  default="external_data/pahdb-complete-theoretical-v4.00.xml")
    parser.add_argument("--out",  default="model_data/PAH_states/pahdb_catalog.json")
    parser.add_argument("--nc",   nargs="+", type=int, default=None,
                        help="Nc values to include (default: all)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    xml_abs = root / args.xml
    out_abs = root / args.out

    catalog = build_pahdb_catalog(str(xml_abs), str(out_abs), args.nc)

    # Quick summary
    from collections import Counter
    counts = Counter((e["Nc"], e["Z"]) for e in catalog)
    print("\nSpecies count by (Nc, Z):")
    for (nc, z), n in sorted(counts.items()):
        print(f"  C{nc}  Z={z:+d}  : {n} isomers")
