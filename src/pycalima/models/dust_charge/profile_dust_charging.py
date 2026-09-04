#!/usr/bin/env python3
"""
Profile wall time and function hotspots for representative dust charging runs.

Usage:
  python -m models.dust_charge.profile_dust_charging

Outputs:
  model_data/dust_charging_data/profiles/dust_charging_profile_report.json
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import time
from pathlib import Path

from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
from pycalima.models.grain_size_config import get_model_data_dir




def _output_path() -> Path:
    out_dir = get_model_data_dir() / "dust_charging_data" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "dust_charging_profile_report.json"


def _profile_case(case: dict, top_n: int = 25) -> dict:
    """Run one charging case under cProfile and return summarized stats."""
    params = dict(case)

    # Warm-up once to avoid import/JIT effects in the measured run.
    equilibrium_charge_for_grain(
        params["G0"],
        params["ne"],
        params["T"],
        params["grain_type"],
        params["a_cm"],
        radiation_model=params.get("radiation_model", "Mathis"),
        ion_species=params.get("ion_species", []),
        debug=False,
    )

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    Zs, P, rates, zmean, zsigma = equilibrium_charge_for_grain(
        params["G0"],
        params["ne"],
        params["T"],
        params["grain_type"],
        params["a_cm"],
        radiation_model=params.get("radiation_model", "Mathis"),
        ion_species=params.get("ion_species", []),
        debug=False,
    )
    pr.disable()
    wall_s = time.perf_counter() - t0

    # Gather top cumulative functions.
    stream_cum = io.StringIO()
    ps_cum = pstats.Stats(pr, stream=stream_cum).sort_stats("cumtime")
    ps_cum.print_stats(top_n)

    # Gather top self-time functions.
    stream_self = io.StringIO()
    ps_self = pstats.Stats(pr, stream=stream_self).sort_stats("tottime")
    ps_self.print_stats(top_n)

    # Extract a few key cumulative timings programmatically.
    key_funcs = [
        ("dust_charging.py", "compute_equilibrium_charge_distribution_vectorized"),
        ("dust_photoelectric_heating.py", "compute_photoelectric_heating_rate"),
        ("dust_charging.py", "compute_Rpe_vectorized"),
        ("dust_charging.py", "find_Zref_and_bounds_optimized"),
    ]
    key_totals = {}
    for suffix, func_name in key_funcs:
        total = 0.0
        calls = 0
        for (file_name, _line, name), stat in ps_cum.stats.items():
            _cc, nc, _tt, ct, _callers = stat
            if file_name.endswith(suffix) and name == func_name:
                total += float(ct)
                calls += int(nc)
        key_totals[f"{suffix}:{func_name}"] = {"cumtime_s": total, "calls": calls}

    return {
        "case": {
            "grain_type": params["grain_type"],
            "a_cm": params["a_cm"],
            "G0": params["G0"],
            "ne": params["ne"],
            "T": params["T"],
            "radiation_model": params.get("radiation_model", "Mathis"),
        },
        "result": {
            "N_Z": int(len(Zs)),
            "Zmean": float(zmean),
            "Zsigma": float(zsigma),
            "Gamma_total": float(rates.get("Gamma_total", 0.0)),
            "Recomb_total": float(rates.get("Recomb_total", 0.0)),
            "efficiency": float(rates.get("efficiency", 0.0)),
        },
        "timing": {
            "wall_s": float(wall_s),
            "profiler_total_s": float(pstats.Stats(pr).total_tt),
        },
        "key_cumulative": key_totals,
        "top_cumulative_text": stream_cum.getvalue(),
        "top_self_text": stream_self.getvalue(),
    }


def main() -> None:
    # A compact set of representative cases across material and charging regime.
    cases = [
        {
            "grain_type": "graphite",
            "a_cm": 1.0e-6,
            "G0": 0.01,
            "ne": 0.007,
            "T": 33.0,
            "radiation_model": "Mathis",
            "ion_species": [
                {"n": 3e-3, "T": 33.0, "m": 1.6726219e-24, "z": 1},
                {"n": 0.0042, "T": 33.0, "m": 12.0 * 1.66053906660e-24, "z": 1},
            ],
        },
        {
            "grain_type": "graphite",
            "a_cm": 5.0e-7,
            "G0": 1.0,
            "ne": 0.03,
            "T": 100.0,
            "radiation_model": "Mathis",
            "ion_species": [{"n": 1e-2, "T": 100.0, "m": 1.6726219e-24, "z": 1}],
        },
        {
            "grain_type": "silicate",
            "a_cm": 5.0e-7,
            "G0": 1.0,
            "ne": 0.03,
            "T": 100.0,
            "radiation_model": "Mathis",
            "ion_species": [{"n": 1e-2, "T": 100.0, "m": 1.6726219e-24, "z": 1}],
        },
    ]

    report = {
        "generated_at_unix_s": time.time(),
        "cwd": os.getcwd(),
        "python_executable": os.environ.get("PYTHON_EXECUTABLE", ""),
        "cases": [],
    }

    print("Running dust charging profiler cases...")
    for i, case in enumerate(cases, start=1):
        print(
            f"[{i}/{len(cases)}] {case['grain_type']}, a={case['a_cm']:.3e} cm, "
            f"G0={case['G0']}, ne={case['ne']}, T={case['T']}"
        )
        case_report = _profile_case(case)
        report["cases"].append(case_report)
        print(
            f"  -> wall={case_report['timing']['wall_s']:.3f}s, "
            f"N_Z={case_report['result']['N_Z']}, "
            f"Zmean={case_report['result']['Zmean']:.3f}, "
            f"Zsigma={case_report['result']['Zsigma']:.3f}"
        )

    out_path = _output_path()
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
