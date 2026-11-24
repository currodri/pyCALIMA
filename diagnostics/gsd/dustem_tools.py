"""Build and run DustEM (Compiegne et al. 2011) from Python, for the GSD's observational checks (A3c, Track O).

The source and data come from $CALIMA_DUSTEM_DIR, which is never written to: ``build`` copies the Fortran sources
into a work folder, points DustEM's data path at a run folder there and compiles it with gfortran; the run folder
has its own data/ (symlinks to every data file except the inputs written per run, GRAIN.DAT and SIZE_*.DAT),
links to oprop/ and hcap/, and its own out/. ``run`` writes a GRAIN.DAT (and tabulated size distributions), runs
DustEM and reads its SED, extinction and size distribution outputs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def dustem_root() -> Path:
    d = os.environ.get("CALIMA_DUSTEM_DIR")
    if not d or not (Path(d) / "src" / "DM_constants.f90").exists():
        raise SystemExit("set $CALIMA_DUSTEM_DIR to the DustEM 4.3 folder (with src/, data/, oprop/, hcap/)")
    return Path(d)


def build(work: Path) -> Path:
    """The DustEM executable in ``work`` (compiled once). Returns the run folder."""
    root = dustem_root()
    work = Path(work).resolve()
    run = work / "run"
    exe = work / "src" / "dustem"
    if not exe.exists():
        src = work / "src"
        shutil.copytree(root / "src", src, dirs_exist_ok=True)
        path = str(run) + "/"
        if len(path) > 100:
            raise SystemExit(f"DustEM's data path is limited to 100 characters: {path}")
        c = (src / "DM_constants.f90").read_text()
        c = re.sub(r"data_path='[^']*'", f"data_path='{path}'", c, count=1)
        (src / "DM_constants.f90").write_text(c)
        mk = (src / "Makefile").read_text()
        mk = re.sub(r"^ FFLAGS = -O3.*$", " FFLAGS = -O2 -fno-second-underscore", mk, flags=re.M)
        (src / "Makefile").write_text(mk)
        subprocess.run(["make", "clean"], cwd=src, check=True, capture_output=True)
        subprocess.run(["make"], cwd=src, check=True, capture_output=True)
    (run / "data").mkdir(parents=True, exist_ok=True)
    (run / "out").mkdir(exist_ok=True)
    for sub in ("oprop", "hcap"):
        if not (run / sub).exists():
            (run / sub).symlink_to(root / sub)
    for f in (root / "data").iterdir():
        dst = run / "data" / f.name
        if f.name == "GRAIN.DAT" or f.name.startswith("SIZE_") or dst.exists():
            continue
        dst.symlink_to(f)
    return run


def run(work: Path, keywords: str, lines: list[str], sizes: dict | None = None, g0: float = 1.0) -> dict:
    """Run DustEM with the grain lines ``lines`` (GRAIN.DAT format) and run keywords. ``sizes``: {grain type:
    (a [cm], a^4 dn/da, rho(a))} written as SIZE_<type>.DAT for types with the 'size' keyword. Returns {'lam' [um],
    'sed' [erg/s/H, 4 pi nu I_nu / N_H, total], 'ext' [cm^2/H, total], 'sed_types', 'ext_types'}."""
    run_dir = build(work)
    data = run_dir / "data"
    for f in data.glob("SIZE_*.DAT"):
        f.unlink()
    for t, (a, a4n, rho) in (sizes or {}).items():
        body = "\n".join(f" {x:.6e}  {y:.6e}  {r:.6e}" for x, y, r in zip(a, a4n, rho, strict=True))
        (data / f"SIZE_{t}.DAT").write_text(f"# size distribution written by dustem_tools\n# [a (cm), a^4 dn/da, rho]\n"
                                            f"{len(a)}\n{body}\n")
    (data / "GRAIN.DAT").write_text("# written by dustem_tools\n" + keywords + "\n" + f"{g0:.6f}\n"
                                    + "\n".join(lines) + "\n")
    for f in (run_dir / "out").glob("*.RES"):
        f.unlink()
    exe = Path(work).resolve() / "src" / "dustem"
    p = subprocess.run([str(exe)], cwd=run_dir, capture_output=True, text=True)
    if p.returncode != 0 or not (run_dir / "out" / "SED.RES").exists():
        raise RuntimeError("DustEM failed:\n" + p.stdout[-2000:] + p.stderr[-2000:])
    sed = _read_res(run_dir / "out" / "SED.RES")
    ext = _read_res(run_dir / "out" / "EXT.RES")
    out = {"lam": sed[:, 0], "sed": sed[:, -1], "sed_types": sed[:, 1:-1], "ext": ext[:, -1] * 1e-21,
           "ext_lam": ext[:, 0]}
    if (run_dir / "out" / "SDIST.RES").exists():
        out["sdist"] = read_sdist(run_dir / "out" / "SDIST.RES")
    return out


def _read_res(path):
    rows = []
    started = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not started:                      # the "n_types n_lambda" line
            started = True
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows)


def read_sdist(path):
    """[(type, a [cm], a^4 dn/da [cm^3/H], rho)] from SDIST.RES."""
    lines = [s for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith("#")]
    n_types, k, out = int(lines[0]), 1, []
    for _ in range(n_types):
        name, n = lines[k].split()[0], int(lines[k].split()[1])
        arr = np.array([[float(x) for x in s.split()] for s in lines[k + 1:k + 1 + n]])
        out.append((name, arr[:, 0], arr[:, 1], arr[:, 2]))
        k += 1 + n
    return out


def grain_lines(model: str) -> tuple[str, list[str]]:
    """(run keywords, grain lines) of DustEM's data/GRAIN_<model>.DAT."""
    text = (dustem_root() / "data" / f"GRAIN_{model}.DAT").read_text().splitlines()
    body = [s for s in text if s.strip() and not s.startswith("#")]
    return body[0], body[2:]
