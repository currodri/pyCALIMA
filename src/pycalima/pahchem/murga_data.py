"""The public Murga et al. model outputs (Figshare, CC BY 4.0): fetchers and parsers.

- **Murga+2020** (MNRAS 497, 2327; doi:10.6084/m9.figshare.12597092):
  environment profiles (Fig. 2) and PAH C2H2 production rates (Fig. 6, A1)
  for the Orion Bar (isochoric ``goi1`` and isobaric ``goi2``) and the
  Horsehead. Dataset ``murga2020-figshare``.
- **Murga+2022** (MNRAS 509, 800; doi:10.6084/m9.figshare.14892612): the
  (location, Z, time, X_H, size) populations of models b, c and d along the
  MARION Orion Bar profile, and the band intensities. Dataset
  ``murga2022-figshare`` (~215 MB).

Both are ``kind = "fetch"`` in ``data/registry.toml``; ``calima-fetch-data
fetch <name>`` calls the fetchers below, which check the MD5 sums that
Figshare publishes. Layouts are those read on 2026-10-05 (method cards
murga2020.toml, murga2022.toml).
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np

FIGSHARE_2020 = {"Share_Acetylene.zip": ("https://ndownloader.figshare.com/files/23738321",
                                         "3348d6064527098523e805d4148aa431")}
FIGSHARE_2022 = {
    "result_marion1_av_nodestr.dat": ("https://ndownloader.figshare.com/files/28675968",
                                      "57a3e067dcdaf940123b32d044e0298d"),
    "result_marion1_av_nodestr60.dat": ("https://ndownloader.figshare.com/files/28675971",
                                        "94baba754da7a3afc3bc3d4c25b14e31"),
    "result_marion1_av.dat": ("https://ndownloader.figshare.com/files/28675974",
                              "4732ab23f398cb90ef3843f6e92223d4"),
    "bands_fix_marion1_av_nodestr.dat": ("https://ndownloader.figshare.com/files/28690398",
                                         "3ea5a96c687f889dc533bafe458333c1"),
    "bands_fix_marion1_av_nodestr60.dat": ("https://ndownloader.figshare.com/files/28690395",
                                           "a6a4f2cdf0d8d04dc1def43a81f4841c"),
    "bands_fix_marion1_av.dat": ("https://ndownloader.figshare.com/files/28690401",
                                 "f442aa9e7305c632b1df7425df99f031"),
}
#: Murga+2022 Table 4 model -> result file (a, no H or C loss, has no file)
MODELS_2022 = {"b": "result_marion1_av_nodestr.dat", "c": "result_marion1_av.dat",
               "d": "result_marion1_av_nodestr60.dat"}
#: Murga+2020 environments -> (profile, C2H2-rate file), relative to Share_Acetylene/
SITES_2020 = {"orion_isochoric": ("Orion-Bar-1/Orion_par_goi1.dat", "Orion-Bar-1/pahrates_orion_goi1.dat"),
              "orion_isobaric": ("Orion-Bar-2/Orion_par_goi2.dat", "Orion-Bar-2/pahrates_orion_goi2.dat"),
              "horsehead": ("Horsehead/Horse_par_fix.dat", "Horsehead/pahrates_horse.dat")}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(files, output_dir):
    from pycalima._datasets import DatasetUnavailable, _download

    out = Path(output_dir)
    for name, (url, md5) in files.items():
        dest = out / name
        if not (dest.exists() and _md5(dest) == md5):
            _download(url, dest)
        if _md5(dest) != md5:
            raise DatasetUnavailable(f"{name}: MD5 mismatch against Figshare ({md5})")
    return out


def fetch_murga2020(output_dir):
    """Download and unpack the Murga+2020 Figshare archive into ``output_dir``."""
    out = _fetch(FIGSHARE_2020, output_dir)
    with zipfile.ZipFile(out / "Share_Acetylene.zip") as z:
        for member in z.namelist():
            if member.startswith("__MACOSX") or member.endswith(".DS_Store"):
                continue
            z.extract(member, out)
    shutil.rmtree(out / "__MACOSX", ignore_errors=True)


def fetch_murga2022(output_dir):
    """Download the Murga+2022 Figshare model outputs into ``output_dir``."""
    _fetch(FIGSHARE_2022, output_dir)


# ---------------------------------------------------------------------------
# Murga+2020
# ---------------------------------------------------------------------------

def read_profile_2020(site, root=None):
    """Fig. 2 profile of ``site`` (``SITES_2020``) as a dict of arrays.

    Columns: A_V, T_gas [K], n_tot = n_H + n_H2 [cm^-3] as printed, then the
    abundances x(H), x(H2), x(e), x(C+) and chi (``U``, MMP83 units).
    The abundance normalisation differs between files (inferred): the Orion
    rows have x(H) + 2 x(H2) = 1 (per H nucleus), the Horsehead rows
    x(H) + x(H2) = 1. It is recorded here, not resolved.
    """
    root = _root("murga2020-figshare") if root is None else Path(root)
    t = np.loadtxt(root / "Share_Acetylene" / SITES_2020[site][0], comments="#")
    keys = ("av", "t_gas", "n_tot", "x_h", "x_h2", "x_e", "x_cplus", "chi")
    return {k: t[:, i] for i, k in enumerate(keys)}


def read_c2h2_rates_2020(site, root=None):
    """PAH C2H2 production rate per H nucleus [s^-1] behind Fig. 6.

    Returns (av, t_yr, rate) with rate[i_av, i_t] on the profile's A_V points
    and the 100-step 1-1e6 yr time grid. Multiply by ``n_tot`` of the profile
    for s^-1 cm^-3.
    """
    root = _root("murga2020-figshare") if root is None else Path(root)
    t = np.loadtxt(root / "Share_Acetylene" / SITES_2020[site][1])
    i_av, times = t[:, 0].astype(int), np.unique(t[:, 1])
    rate = np.full((i_av.max() + 1, times.size), np.nan)
    rate[i_av, np.searchsorted(times, t[:, 1])] = t[:, 2]
    return read_profile_2020(site, root)["av"], times, rate


# ---------------------------------------------------------------------------
# Murga+2022
# ---------------------------------------------------------------------------

N_LOC, N_Z, N_T, N_XH, N_SIZE = 158, 12, 11, 5, 10


def read_model_2022(model="b", root=None, cache=True):
    """Populations of Murga+2022 model ``model`` (b, c or d; ``MODELS_2022``).

    Returns a dict:

    - ``n`` [loc, Z, t, X_H, size]: number density [cm^-3];
    - ``f`` [loc, Z, t, X_H, size]: the WD01 charge fraction f(a, Z), the
      same for every t and X_H;
    - ``n_h``, ``t_gas``, ``chi`` [loc]: from the block headers;
    - ``z`` [Z], ``x_h`` [X_H] (X_H column; the normal state is 1.385), and
      ``a_cm`` [size].

    Each block is "counter n_H T chi Z", then 5 X_H sub-blocks of 10 rows
    "a X_H n Z f". The parse (~1 min) is cached as ``<file>.npz``.
    """
    root = _root("murga2022-figshare") if root is None else Path(root)
    path = root / MODELS_2022[model]
    npz = path.with_suffix(".npz")
    if cache and npz.exists():
        with np.load(npz) as d:
            return {k: d[k] for k in d.files}
    lines = path.read_text().split("\n")
    shape = (N_LOC, N_Z, N_T, N_XH, N_SIZE)
    n, f = np.zeros(shape), np.zeros(shape)
    env = np.zeros((N_LOC, 3))
    z, x_h, a_cm = np.zeros(N_Z), np.zeros(N_XH), np.zeros(N_SIZE)
    i = block = 0
    while i < len(lines):
        s = lines[i].split()
        if len(s) == 5 and not lines[i].startswith(" "):
            loc, rem = divmod(block, N_Z * N_T)
            iz, it = divmod(rem, N_T)
            env[loc] = [float(x) for x in s[1:4]]
            z[iz] = float(s[4])
            block += 1
            i += 1
            for e in range(N_XH):
                i += 1                                   # "e 0" sub-header
                rows = np.array([lines[i + m].split() for m in range(N_SIZE)], float)
                i += N_SIZE
                a_cm[:], x_h[e] = rows[:, 0], rows[0, 1]
                n[loc, iz, it, e], f[loc, iz, it, e] = rows[:, 2], rows[:, 4]
        else:
            i += 1
    if block != N_LOC * N_Z * N_T:
        raise ValueError(f"{path.name}: {block} blocks, expected {N_LOC * N_Z * N_T}")
    out = {"n": n, "f": f, "n_h": env[:, 0], "t_gas": env[:, 1], "chi": env[:, 2],
           "z": z, "x_h": x_h, "a_cm": a_cm}
    if cache:
        np.savez(npz, **out)
    return out


def _root(name):
    from pycalima._datasets import ensure_dataset

    return ensure_dataset(name)
