"""Write the Planck-mean absorption tables of the collapse materials.

    python -m pycalima.collapse.export_collapse_optics [--species Fe MgO ...] [--force]

One ``planck_qabs_<species>.npz`` per material in ``model_data/collapse/``
(``optics.planck_table_path``). Species whose optical constants are a
registered dataset that has not been fetched are skipped with the fetch
command to run.
"""

from __future__ import annotations

import argparse

from pycalima.collapse.materials import materials
from pycalima.collapse.optics import PlanckTable, planck_table_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--species", nargs="*", default=None, help="default: all")
    p.add_argument("--force", action="store_true", help="recompute existing tables")
    args = p.parse_args(argv)
    names = args.species or sorted(materials())
    for name in names:
        path = planck_table_path(name)
        if path.exists() and not args.force:
            print(f"{name:8s} exists: {path}")
            continue
        try:
            tab = PlanckTable.compute(name)
        except (FileNotFoundError, KeyError) as exc:   # DatasetUnavailable, or not registered
            print(f"{name:8s} skipped: {exc}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tab.save(path)
        print(f"{name:8s} wrote {path}")


if __name__ == "__main__":
    main()
