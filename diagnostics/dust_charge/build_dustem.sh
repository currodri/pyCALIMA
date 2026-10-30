#!/usr/bin/env bash
# Build and run DustEM v4.3 for the CM20 charging check (check_cm20_dustem.py).
#
#   diagnostics/dust_charge/build_dustem.sh [--dump] DUSTEM_RELEASE BUILD_DIR
#
# DUSTEM_RELEASE: the extracted release ($CALIMA_DUSTEM_DIR: data/, oprop/, src/, ...).
# BUILD_DIR: where to copy and build it. DustEM keeps its data path in a 100-character
#   string (DM_constants.f90), so BUILD_DIR must be shorter than ~95 characters.
# --dump: also patch DM_zdist.f90 to write, per grain type, size index and charge, the
#   photoelectric current and heating, the ion and electron currents, l_e and Z_eq to
#   BUILD_DIR/src/fort.99 (columns: type ns a Z f(Z) Jpe Hpe J_ion J_e le[A] Zeq).
#
# The release's Makefile uses x86 flags (-msse2, -ffast-math); this builds with
# gfortran -O2. Its out/ZDIST.RES differs from the shipped one: above ~0.5 um the
# shipped CM20 charges (<Z> ~ 0) are an artefact, and at the smallest sizes the two
# differ by ~4% (DustEM's l_e in PE_CURRENT is not set when no photon exceeds the
# threshold, so it depends on the build).
#
# Then:  python diagnostics/dust_charge/check_cm20_dustem.py --zdist BUILD_DIR/out/ZDIST.RES
set -euo pipefail

dump=0
if [[ "${1:-}" == "--dump" ]]; then dump=1; shift; fi
if [[ $# -ne 2 ]]; then
    sed -n '2,20p' "$0"; exit 2
fi
release=$(cd "$1" && pwd)
build=$2
if [[ ${#build} -gt 94 ]]; then
    echo "BUILD_DIR is ${#build} characters; DustEM's data_path holds 100: use a shorter path" >&2
    exit 1
fi
[[ -f "$release/src/DM_zdist.f90" ]] || { echo "$release is not a DustEM release (no src/DM_zdist.f90)" >&2; exit 1; }

mkdir -p "$build"
build=$(cd "$build" && pwd)
rsync -a --exclude out "$release/" "$build/"
mkdir -p "$build/out"

python3 - "$build" "$dump" <<'EOF'
import sys
from pathlib import Path

build, dump = Path(sys.argv[1]), sys.argv[2] == "1"
src = build / "src"
p = src / "DM_constants.f90"
s = p.read_text()
old = s[s.index("data_path='"):].split("'")[1]
p.write_text(s.replace(f"data_path='{old}'", f"data_path='{build}/'"))
p = src / "Makefile"
lines = [("# " + l if l.lstrip().startswith("FFLAGS") and not l.lstrip().startswith("#") else l)
         for l in p.read_text().splitlines()]
i = next(k for k, l in enumerate(lines) if "FC = gfortran" in l)
lines.insert(i + 1, " FFLAGS = -O2 -fno-second-underscore")
p.write_text("\n".join(lines) + "\n")
if dump:
    p = src / "DM_zdist.f90"
    s = p.read_text()
    a = """          CALL GAS_CURRENTS (nt, 1, 1, nion, z1, aa, t_gas, le, jp, jm, cp, cm)
          jgg = jgg + (jp(1) + jm(1))*fz(i)"""
    b = """          CALL GAS_CURRENTS (nt, 1, 1, nion, z1, aa, t_gas, le, jp, jm, cp, cm)
          WRITE (99, '(A8,1X,I4,1X,1P,9(E14.6,1X))') gtype(nt), ns, a, zb(i), fz(i), jj, hh, jp(1), jm(1), le, zeq
          jgg = jgg + (jp(1) + jm(1))*fz(i)"""
    assert s.count(a) == 1, "DM_zdist.f90 is not the v4.3 source this patch expects"
    p.write_text(s.replace(a, b))
EOF

(cd "$build/src" && make clean >/dev/null && make >/dev/null && ./dustem > "$build/out/run.log" 2>&1)
tail -3 "$build/out/run.log"
echo "wrote $build/out/ZDIST.RES"
[[ $dump -eq 1 ]] && echo "per-charge rates in $build/src/fort.99"
exit 0
