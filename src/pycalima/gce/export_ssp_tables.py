"""Export cumulative SSP release tables to ``model_data/ssp_yields/``.

    python -m pycalima.gce.export_ssp_tables
    python -m pycalima.gce.export_ssp_tables --yield-set kobayashi11 --dust-preset parente23

The tables do not depend on the grain-size configuration, so they live
beside, not inside, the per-model directories.
"""

from __future__ import annotations

import argparse

from .ssp import build_ssp_table, default_table_path
from .stellar_dust import PRESETS
from .stellar_yields import _YIELD_SETS

DEFAULT_COMBINATIONS = (("lc18_karakas", "chabrier", "dwek98"),
                        ("kobayashi11", "chabrier", "dwek98"))


def export_ssp_tables(combinations=DEFAULT_COMBINATIONS) -> dict:
    """Build and save one table per (yield_set, imf, dust_preset)."""
    written = []
    for yield_set, imf, dust_preset in combinations:
        table = build_ssp_table(yield_set, imf=imf, dust_preset=dust_preset)
        path = table.save(default_table_path(yield_set, imf, dust_preset))
        print(f"  wrote {path}")
        written.append(str(path))
    return {"tables": written, "output_dir": str(default_table_path().parent)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yield-set", choices=sorted(_YIELD_SETS))
    parser.add_argument("--imf", default="chabrier", choices=["chabrier", "kroupa", "salpeter"])
    parser.add_argument("--dust-preset", default="dwek98", choices=sorted(PRESETS))
    args = parser.parse_args(argv)
    combos = (DEFAULT_COMBINATIONS if args.yield_set is None
              else ((args.yield_set, args.imf, args.dust_preset),))
    export_ssp_tables(combos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
