"""
download_cagliari_pah_database.py
==================================
Scrape and download the Cagliari Theoretical Spectral Database of PAHs:
    https://www.dsf.unica.it/~gmalloci/pahs/pahs.html

For each of the 40 PAHs in the database this script fetches the
'General properties' sub-page and extracts:

    - Formula, Nc (carbon atoms), Nh (hydrogen atoms)
    - Symmetry point group (neutral and ions)
    - Single ionization energy:  IP1_adiab, IP1_vert, IP1_exp   [eV]
    - Double ionization energy:
          IP2_singlet_adiab, IP2_singlet_vert                    [eV]
          IP2_triplet_adiab, IP2_triplet_vert                    [eV]
          IP2_exp                                                 [eV]
    - Electron affinity:         EA_adiab, EA_vert, EA_exp       [eV]
    - Zero-point energies:       ZPE_{anion,neutral,cation,dication_T} [kcal/mol]
    - Static dipole polarizabilities (alpha_xx, alpha_yy, alpha_zz) [Angstrom^3]
      for anion, neutral, cation, dication_S
    - Mean isotropic polarizability per charge state:
          alpha_mean = (alpha_xx + alpha_yy + alpha_zz) / 3       [Angstrom^3]
    - Rotational constants A, B, C [cm^-1]
      for anion, neutral, cation, dication_S, dication_T
    - van der Waals:  vdW_alpha0 [a.u.], vdW_omega1, vdW_C6, vdW_KAA

Output
------
    cagliari_pah_database.csv  in the same directory as this script.

Usage
-----
    python external_data/download_cagliari_pah_database.py

No external packages required (stdlib only: urllib, re, csv, ssl).
"""

from __future__ import annotations

import csv
import re
import ssl
import time
import urllib.request
from pathlib import Path
from html import unescape

# ── configuration ─────────────────────────────────────────────────────────────

BASE_URL    = "https://www.dsf.unica.it/~gmalloci/pahs"
INDEX_URL   = f"{BASE_URL}/frames/framepahs.html"
OUTPUT_NAME = "cagliari_pah_database.csv"
DELAY_S     = 0.3   # polite delay between requests

# Ignore self-signed certificate (the server has an outdated cert)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE


# ── helpers ───────────────────────────────────────────────────────────────────

def fetch(url: str) -> str:
    """Fetch URL, return decoded text; retries once on failure."""
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, context=SSL_CTX, timeout=15) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as exc:
            if attempt == 0:
                time.sleep(1.0)
            else:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def strip_tags(html: str) -> str:
    """Remove HTML tags, decode entities, collapse whitespace."""
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL)
    html = re.sub(r"<[^>]+>", " ", html)
    html = unescape(html)
    html = re.sub(r"[  ]+", " ", html)  # also remove non-breaking space (\xa0)
    return html


def find_float(pattern: str, text: str) -> float | None:
    """Return first float matching pattern, or None."""
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            pass
    return None


def parse_formula(formula: str) -> tuple[int, int]:
    """
    Parse e.g. 'C24H12' → (24, 12).
    Returns (Nc, Nh).
    """
    mc = re.search(r"C(\d+)", formula)
    mh = re.search(r"H(\d+)", formula)
    Nc = int(mc.group(1)) if mc else 0
    Nh = int(mh.group(1)) if mh else 0
    return Nc, Nh


# ── index parser ──────────────────────────────────────────────────────────────

def parse_index(html: str) -> list[dict]:
    """
    Returns list of {name, display_name, formula} dicts from framepahs.html.
    """
    # links look like: href="../azulene/azulene.html"
    molecules = []
    for m in re.finditer(
        r'href="\.\./([^/]+)/[^"]+\.html"[^>]*>\s*'
        r'([^<(]+)\s*\(([^)]+)\)',
        html,
    ):
        name         = m.group(1).strip()
        display_name = m.group(2).strip()
        formula_raw  = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        # Rebuild clean formula: 'C 24 H 12' → 'C24H12'
        formula = re.sub(r"\s+", "", formula_raw)
        molecules.append(dict(name=name, display_name=display_name, formula=formula))
    return molecules


# ── per-molecule parser ────────────────────────────────────────────────────────

def parse_gen_page(text: str) -> dict:
    """
    Parse plain text of a *_gen.html page (after strip_tags) into a dict of
    floats (or None for missing values).
    """
    d: dict[str, float | str | None] = {}

    # ── symmetry ──────────────────────────────────────────────────────────────
    m = re.search(r"Symmetry point group\s+(D\w+|C\w+)\s*\(neutral\)", text)
    d["symmetry_neutral"] = m.group(1) if m else None

    m = re.search(r"Symmetry point group\s+(D\w+|C\w+)\s*\(ions\)", text)
    d["symmetry_ions"] = m.group(1) if m else None

    # ── single ionisation energy ───────────────────────────────────────────────
    # Locate the "Single ionization energy" section
    m_sec = re.search(r"Single ionization energy.*?Double ionization energy", text, re.DOTALL)
    sec = m_sec.group(0) if m_sec else ""
    d["IP1_adiab_eV"]  = find_float(r"Adiab\.\s*=\s*([\d.]+)", sec)
    d["IP1_vert_eV"]   = find_float(r"Vert\.\s*=\s*([\d.]+)", sec)
    # experimental: ( Exp. 7.29 ± 0.03 ) or ( Exp. – )
    me = re.search(r"Exp\.\s+([\d.]+)\s*[±]", sec)
    d["IP1_exp_eV"] = float(me.group(1)) if me else None

    # ── double ionisation energy ───────────────────────────────────────────────
    m_sec = re.search(r"Double ionization energy.*?Electron affinity", text, re.DOTALL)
    sec = m_sec.group(0) if m_sec else ""

    # Singlet first, then Triplet
    m_sing = re.search(r"Singlet:(.*?)(?:Triplet:|$)", sec, re.DOTALL)
    m_trip = re.search(r"Triplet:(.*?)$", sec, re.DOTALL)
    sing = m_sing.group(1) if m_sing else ""
    trip = m_trip.group(1) if m_trip else ""

    d["IP2_singlet_adiab_eV"] = find_float(r"Adiab\.\s*=\s*([\d.]+)", sing)
    d["IP2_singlet_vert_eV"]  = find_float(r"Vert\.\s*=\s*([\d.]+)", sing)
    d["IP2_triplet_adiab_eV"] = find_float(r"Adiab\.\s*=\s*([\d.]+)", trip)
    d["IP2_triplet_vert_eV"]  = find_float(r"Vert\.\s*=\s*([\d.]+)", trip)
    me = re.search(r"Exp\.\s+([\d.]+)\s*[±]", sec)
    d["IP2_exp_eV"] = float(me.group(1)) if me else None

    # ── electron affinity ──────────────────────────────────────────────────────
    m_sec = re.search(r"Electron affinity.*?Zero Point Energies", text, re.DOTALL)
    sec = m_sec.group(0) if m_sec else ""
    d["EA_adiab_eV"] = find_float(r"Adiab\.\s*=\s*([\d.]+)", sec)
    d["EA_vert_eV"]  = find_float(r"Vert\.\s*=\s*([\d.]+)", sec)
    me = re.search(r"Exp\.\s+([\d.]+)\s*[±]", sec)
    d["EA_exp_eV"] = float(me.group(1)) if me else None

    # ── zero-point energies ────────────────────────────────────────────────────
    m_sec = re.search(r"Zero Point Energies.*?Static dipole", text, re.DOTALL)
    sec = m_sec.group(0) if m_sec else ""
    d["ZPE_anion_kcalmol"]     = find_float(r"Anion\s*=\s*([\d.]+)", sec)
    d["ZPE_neutral_kcalmol"]   = find_float(r"Neutral\s*=\s*([\d.]+)", sec)
    d["ZPE_cation_kcalmol"]    = find_float(r"Cation\s*=\s*([\d.]+)", sec)
    d["ZPE_dication_T_kcalmol"]= find_float(r"Dication\s+T\s*=\s*([\d.]+)", sec)

    # ── static dipole polarizabilities ────────────────────────────────────────
    m_sec = re.search(r"Static dipole polarizabilit.*?Rotational constants", text, re.DOTALL)
    sec = m_sec.group(0) if m_sec else ""

    def _parse_pol_block(label: str, src: str) -> tuple[float | None, float | None, float | None]:
        m_blk = re.search(rf"{label}\s*:(.*?)(?:Anion|Neutral|Cation|Dication|Rotational|$)",
                          src, re.DOTALL)
        blk = m_blk.group(1) if m_blk else ""
        xx = find_float(r"xx\s*=\s*([\d.]+)", blk)
        yy = find_float(r"yy\s*=\s*([\d.]+)", blk)
        zz = find_float(r"zz\s*=\s*([\d.]+)", blk)
        return xx, yy, zz

    for label, prefix in [("Anion", "pol_anion"),
                           ("Neutral", "pol_neutral"),
                           ("Cation", "pol_cation"),
                           ("Dication S", "pol_dication_S")]:
        xx, yy, zz = _parse_pol_block(label, sec)
        d[f"{prefix}_axx_Ang3"] = xx
        d[f"{prefix}_ayy_Ang3"] = yy
        d[f"{prefix}_azz_Ang3"] = zz
        if xx is not None and yy is not None and zz is not None:
            d[f"{prefix}_amean_Ang3"] = (xx + yy + zz) / 3.0
        else:
            d[f"{prefix}_amean_Ang3"] = None

    # ── rotational constants ───────────────────────────────────────────────────
    m_sec = re.search(r"Rotational constants.*?van der Waals", text, re.DOTALL)
    sec = m_sec.group(0) if m_sec else ""

    def _parse_rot_block(label: str, src: str) -> tuple[float | None, float | None, float | None]:
        m_blk = re.search(rf"{label}\s*:(.*?)(?:Anion|Neutral|Cation|Dication|van der|$)",
                          src, re.DOTALL)
        blk = m_blk.group(1) if m_blk else ""
        A = find_float(r"A\s*=\s*([\d.e+-]+)", blk)
        B = find_float(r"B\s*=\s*([\d.e+-]+)", blk)
        C = find_float(r"C\s*=\s*([\d.e+-]+)", blk)
        return A, B, C

    for label, prefix in [("Anion",      "rot_anion"),
                           ("Neutral",    "rot_neutral"),
                           ("Cation",     "rot_cation"),
                           ("Dication S", "rot_dication_S"),
                           ("Dication T", "rot_dication_T")]:
        A, B, C = _parse_rot_block(label, sec)
        d[f"{prefix}_A_cm-1"] = A
        d[f"{prefix}_B_cm-1"] = B
        d[f"{prefix}_C_cm-1"] = C

    # ── van der Waals coefficients ─────────────────────────────────────────────
    m_sec = re.search(r"van der Waals coefficients.*", text, re.DOTALL)
    sec = m_sec.group(0) if m_sec else ""
    d["vdW_alpha0_au"]  = find_float(r"alpha\s*\(0\)\s*=\s*([\d.]+)", sec)
    d["vdW_omega1"]     = find_float(r"omega\s*1\s*=\s*([\d.]+)", sec)
    # C6 is stored as value × 10^-3 on the page
    d["vdW_C6_x1e-3_au"] = find_float(r"C 6\s*=\s*([\d.]+)", sec)
    d["vdW_KAA_x1e-6_au"]= find_float(r"K AA\s*=\s*([\d.]+)", sec)

    return d


# ── main ──────────────────────────────────────────────────────────────────────

def main(output_dir=None):
    print(f"Fetching index from {INDEX_URL} …")
    index_html = fetch(INDEX_URL)
    molecules  = parse_index(index_html)
    print(f"  Found {len(molecules)} PAHs.\n")

    rows: list[dict] = []

    for i, mol in enumerate(molecules, 1):
        name     = mol["name"]
        formula  = mol["formula"]
        Nc, Nh   = parse_formula(formula)
        gen_url  = f"{BASE_URL}/{name}/{name}_gen.html"

        print(f"  [{i:02d}/{len(molecules)}]  {mol['display_name']:35s}  {formula:12s}  "
              f"Nc={Nc:3d}  Nh={Nh:2d}  … ", end="", flush=True)

        try:
            html = fetch(gen_url)
            text = strip_tags(html)
            props = parse_gen_page(text)
        except Exception as exc:
            print(f"ERROR: {exc}")
            props = {}

        row: dict = {
            "name":         mol["display_name"],
            "formula":      formula,
            "Nc":           Nc,
            "Nh":           Nh,
            "url_gen":      gen_url,
        }
        row.update(props)
        rows.append(row)
        print("ok")
        time.sleep(DELAY_S)

    # ── collect all column names in a stable order ─────────────────────────────
    # Fixed prefix first, then all property keys from first complete row
    base_cols = ["name", "formula", "Nc", "Nh", "url_gen"]
    prop_cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in base_cols and k not in prop_cols:
                prop_cols.append(k)
    all_cols = base_cols + prop_cols

    # ── write CSV ─────────────────────────────────────────────────────────────
    if output_dir is None:
        from pycalima._paths import get_dataset_cache_dir
        output_dir = get_dataset_cache_dir(create=True) / "external_data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in all_cols})

    print(f"\nSaved {len(rows)} rows × {len(all_cols)} columns")
    print(f"→ {output_path}")
    return output_path


if __name__ == "__main__":
    main()
