"""The solver layer.

solvers/ is the one subpackage that was already clean before packaging: it uses
relative imports throughout and depends on nothing in models/. These tests pin
that boundary, the initial-condition loader, the RHS assembly, the table
readers and a short end-to-end integration.

Anything that reads generated tables is skipped when no populated model_data/
tree is available, since that directory is gitignored and produced by
`calima-export`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pycalima._paths import list_solver_configs, resolve_solver_config_path


# ---------------------------------------------------------------------------
# the models/ <-> solvers/ boundary
# ---------------------------------------------------------------------------

def test_solvers_does_not_import_models():
    """The produce/consume split: models/ writes tables, solvers/ reads them.
    solvers/ must not pull in the physics layer."""
    import ast
    import pathlib

    import pycalima

    root = pathlib.Path(pycalima.__path__[0]) / "solvers"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if mod and "pycalima.models" in mod:
                offenders.append(f"{path.name}:{node.lineno} -> {mod}")
    assert not offenders, f"solvers/ imports models/: {offenders}"


def test_table_io_constructs_no_paths():
    """Every table_io entry point takes an explicit file, so all path
    resolution stays in dust_init."""
    import inspect

    from pycalima.solvers import table_io

    src = inspect.getsource(table_io)
    for forbidden in ("model_data", "get_model_data_dir"):
        assert forbidden not in src.replace("``models/", ""), (
            f"table_io should not resolve {forbidden!r}; that belongs in dust_init"
        )


# ---------------------------------------------------------------------------
# initial conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config_name", list_solver_configs())
def test_every_bundled_config_loads(config_name, model_data):
    from pycalima.solvers.dust_init import load_initial_conditions

    state, y_gas, y_dust = load_initial_conditions(
        resolve_solver_config_path(config_name)
    )
    assert state.n_elements > 0
    assert y_gas.shape == (state.n_elements,)
    assert y_dust.size == state.npah + len(state.dust_bins)


def test_initial_conditions_are_physical(model_data):
    from pycalima.solvers.dust_init import load_initial_conditions

    state, y_gas, y_dust = load_initial_conditions(
        resolve_solver_config_path("example_ic")
    )
    assert np.all(y_gas >= 0), "negative initial gas density"
    assert np.all(y_dust >= 0), "negative initial dust density"
    assert np.all(np.isfinite(y_gas)) and np.all(np.isfinite(y_dust))
    assert state.local_Tk > 0
    assert state.local_nH > 0
    assert state.local_G0 >= 0


def test_state_bin_metadata_is_consistent(model_data):
    from pycalima.solvers.dust_init import load_initial_conditions

    state, _, y_dust = load_initial_conditions(
        resolve_solver_config_path("example_ic")
    )
    for db in state.dust_bins:
        assert db.asize_cm > 0
        assert db.mgrain > 0
        assert db.sgrain > 0
        assert len(db.el_indices) == len(db.el_mfractions)
        assert sum(db.el_mfractions) == pytest.approx(1.0, rel=1e-6), (
            f"{db.bin_id} mass fractions do not sum to one"
        )
    for pb in state.pah_bins:
        assert 0 <= pb.bin_index < y_dust.size


def test_missing_model_data_raises_an_actionable_error(tmp_path, monkeypatch):
    """dust_init must name calima-export rather than failing obscurely."""
    from pycalima.solvers.dust_init import load_initial_conditions

    monkeypatch.setenv("CALIMA_MODEL_DATA", str(tmp_path / "absent"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="calima-export"):
        load_initial_conditions(resolve_solver_config_path("example_ic"))


def test_bundled_configs_do_not_pin_a_cwd_relative_model_data_dir():
    """All eight used to set "model_data_dir": "model_data", overriding the
    resolver with a path that only worked from the repository root."""
    from pycalima._paths import get_solver_config_dir

    for path in sorted(get_solver_config_dir().glob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert "model_data_dir" not in cfg, (
            f"{path.name} pins model_data_dir; let pycalima._paths resolve it"
        )


def test_relative_model_data_dir_resolves_against_the_config_file(
    tmp_path, monkeypatch, model_data
):
    """A relative override must follow the config, not the process CWD, so a
    config and its tables can travel together."""
    from pycalima.solvers.dust_init import load_initial_conditions

    # config in tmp_path/cfg/, tables reachable as ../tables from there
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (tmp_path / "tables").symlink_to(model_data, target_is_directory=True)

    cfg = json.loads(resolve_solver_config_path("example_ic").read_text(encoding="utf-8"))
    cfg["model_data_dir"] = "../tables"
    custom = cfg_dir / "custom_ic.json"
    custom.write_text(json.dumps(cfg), encoding="utf-8")

    # Run from an unrelated directory: a CWD-relative reading would fail here.
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.delenv("CALIMA_MODEL_DATA", raising=False)
    state, y_gas, y_dust = load_initial_conditions(custom)
    assert y_gas.size == state.n_elements


# ---------------------------------------------------------------------------
# RHS assembly
# ---------------------------------------------------------------------------

def test_process_list_reflects_the_physics_flags(model_data):
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.rhs import build_process_list

    state, _, _ = load_initial_conditions(resolve_solver_config_path("example_ic"))
    procs = build_process_list(state)
    assert procs, "no processes enabled for example_ic"
    names = [p.name for p in procs]
    assert len(names) == len(set(names)), f"duplicate processes: {names}"


def test_all_processes_config_enables_more_than_the_example(model_data):
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.rhs import build_process_list

    n = {}
    for name in ("example_ic", "all_processes_test"):
        state, _, _ = load_initial_conditions(resolve_solver_config_path(name))
        n[name] = len(build_process_list(state))
    assert n["all_processes_test"] >= n["example_ic"]


def test_rhs_is_finite_and_conserves_mass(model_data):
    """Dust growth must be balanced by gas depletion: the summed derivative
    over gas and dust should vanish to round-off."""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.rhs import build_process_list, compute_rhs

    state, y_gas, y_dust = load_initial_conditions(
        resolve_solver_config_path("example_ic")
    )
    procs = build_process_list(state)
    dgas, ddust = compute_rhs(state, y_gas, y_dust, procs)

    assert np.all(np.isfinite(dgas)), "non-finite gas derivative"
    assert np.all(np.isfinite(ddust)), "non-finite dust derivative"

    total = float(dgas.sum() + ddust.sum())
    scale = float(np.abs(dgas).sum() + np.abs(ddust).sum())
    if scale > 0:
        assert abs(total) / scale < 1e-8, (
            f"mass not conserved: net {total:.3e} against flux scale {scale:.3e}"
        )


def test_rhs_can_report_the_stiffness_scale(model_data):
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.rhs import build_process_list, compute_rhs

    state, y_gas, y_dust = load_initial_conditions(
        resolve_solver_config_path("example_ic")
    )
    procs = build_process_list(state)
    result = compute_rhs(state, y_gas, y_dust, procs, return_kmax=True)
    assert len(result) == 3
    kmax = result[2]
    assert np.isfinite(kmax) and kmax >= 0


# ---------------------------------------------------------------------------
# table readers
# ---------------------------------------------------------------------------

def test_sputtering_tables_read_and_interpolate(model_data):
    from pycalima.solvers.table_io import (
        build_sputtering_interpolator,
        read_sputtering_table,
    )

    candidates = sorted((model_data / "thermal_sputtering_data").glob("sputtering_*"))
    candidates = [c for c in candidates if c.is_file() and c.suffix not in (".png", ".pdf")]
    if not candidates:
        pytest.skip("no sputtering tables present")

    table = candidates[0]
    T, phi, rates = read_sputtering_table(table)
    assert T.ndim == 1 and phi.ndim == 1
    assert rates.shape == (T.size, phi.size)
    assert np.all(np.isfinite(rates))
    assert np.all(np.diff(T) > 0), "temperature axis must be increasing"

    # returns (evaluate, info)
    evaluate, info = build_sputtering_interpolator(table)
    assert callable(evaluate)
    assert isinstance(info, dict)
    mid = evaluate(float(T[T.size // 2]), float(phi[phi.size // 2]))
    assert np.all(np.isfinite(np.asarray(mid)))


def test_sputtering_interpolator_reproduces_the_grid_nodes(model_data):
    """Bilinear interpolation must be exact at the tabulated points."""
    from pycalima.solvers.table_io import (
        build_sputtering_interpolator,
        read_sputtering_table,
    )

    candidates = [
        c for c in sorted((model_data / "thermal_sputtering_data").glob("sputtering_*"))
        if c.is_file() and c.suffix not in (".png", ".pdf")
    ]
    if not candidates:
        pytest.skip("no sputtering tables present")

    table = candidates[0]
    T, phi, rates = read_sputtering_table(table)
    evaluate, _ = build_sputtering_interpolator(table)

    for i in (0, T.size // 2, T.size - 1):
        for j in (0, phi.size // 2, phi.size - 1):
            got = float(np.asarray(evaluate(float(T[i]), float(phi[j]))).ravel()[0])
            want = float(rates[i, j])
            if not np.isfinite(want):
                continue
            assert got == pytest.approx(want, rel=1e-6, abs=1e-30), (
                f"node ({i},{j}): interpolator gave {got:.6e}, table has {want:.6e}"
            )


def test_table_reader_rejects_a_missing_file(tmp_path):
    from pycalima.solvers.table_io import read_sputtering_table

    with pytest.raises((FileNotFoundError, OSError)):
        read_sputtering_table(tmp_path / "no_such_table")


# ---------------------------------------------------------------------------
# solver classes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["rk4", "rk54", "anninos", "mprk22"])
def test_time_integrators_construct(name):
    from pycalima.solvers.anninos import AnninosSolver
    from pycalima.solvers.mprk22 import MPRK22Solver
    from pycalima.solvers.rk4 import RK4Solver
    from pycalima.solvers.rk54 import RK54Solver

    cls = {"rk4": RK4Solver, "rk54": RK54Solver, "anninos": AnninosSolver,
           "mprk22": MPRK22Solver}[name]
    solver = cls()
    from pycalima.solvers.solver_base import DustSolverBase

    assert isinstance(solver, DustSolverBase)


@pytest.mark.parametrize("name", ["newton_krylov", "sparse_newton"])
def test_equilibrium_solvers_construct(name):
    from pycalima.solvers.equilibrium import (
        EquilibriumSolverBase,
        NewtonKrylovEquilibriumSolver,
        SparseNewtonEquilibriumSolver,
    )

    cls = {"newton_krylov": NewtonKrylovEquilibriumSolver,
           "sparse_newton": SparseNewtonEquilibriumSolver}[name]
    solver = cls()
    assert isinstance(solver, EquilibriumSolverBase)


def _cool_dense_g10_state(T=454.0, nH=6.16):
    """ramses_G10_MWZsun moved to cool dense gas, densities rescaled with nH."""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.rhs import build_process_list

    state, y_gas, y_dust = load_initial_conditions(
        resolve_solver_config_path("ramses_G10_MWZsun")
    )
    r = nH / state.local_nH
    state.local_Tk, state.local_nH = T, nH
    state.local_rho *= r
    return state, y_gas * r, y_dust * r, build_process_list(state)


def test_nk_equilibrium_is_the_long_time_limit(model_data):
    """A steady state is where the time evolution ends up. Newton from the IC
    used to return a spurious root here (all dust in one bin, the carbonaceous
    bins emptied by the zero-clip in the residual)."""
    from pycalima.solvers.anninos import AnninosSolver
    from pycalima.solvers.equilibrium import NewtonKrylovEquilibriumSolver
    from pycalima.solvers.ode_driver import integrate_dust_ode

    state, y_gas, y_dust, procs = _cool_dense_g10_state()
    _, y_nk, diag = NewtonKrylovEquilibriumSolver(f_tol=1e-300, f_rtol=1e-6).find_equilibrium(
        state, y_gas, y_dust, procs, t_eq_s=400 * 3.156e13)
    _, y_long, _ = integrate_dust_ode(state, 1e6 * 3.156e13, y_gas, y_dust, procs,
                                      AnninosSolver())
    assert diag["converged"], diag["message"]
    frac_nk, frac_long = y_nk / y_nk.sum(), y_long / y_long.sum()
    np.testing.assert_allclose(frac_nk, frac_long, atol=0.02)
    assert np.all(frac_nk[frac_long > 0.05] > 0.05), "a populated bin was emptied"


def test_subgrid_clumping_boost_matches_ramses_yomp(model_data):
    """The accretion boost must be the one in ramses-yomp cooling_module.f90,
    written out here independently of dust_rates."""
    import math

    from pycalima.solvers.dust_rates import accretion_rate

    def fortran_boost(sigma_cm_s, T, nH, nhmax):
        mach  = max(1e-5, sigma_cm_s / math.sqrt(1.666667 * 1.380649e-16 * T / 1.6726219e-24))
        sigs  = math.log(1 + (0.4 * mach) ** 2)
        sigs2 = sigs * sigs
        smax  = math.log(nhmax / min(nH, nhmax))
        return 0.5 * math.exp(sigs2) * math.erfc((1.5 * sigs2 - smax) / (math.sqrt(2) * sigs))

    state, y_gas, y_dust, _ = _cool_dense_g10_state(T=50.0, nH=50.0)   # Jeans-unresolved
    y_dust = np.maximum(y_dust, 1e-30)

    def bin_rates(sigma_km_s):
        state.local_sigma = sigma_km_s * 1e5
        dg, dd = np.zeros_like(y_gas), np.zeros_like(y_dust)
        accretion_rate(state, y_gas, y_dust, dg, dd)
        return dd

    base = bin_rates(0.0)
    for sigma in (0.5, 2.0, 5.0):
        ratio = bin_rates(sigma) / base
        for db in state.dust_bins:
            k = db.bin_index + state.npah
            if base[k] > 0:
                expected = (fortran_boost(sigma * 1e5, 50.0, 50.0, db.nhmax_acc)
                            / fortran_boost(0.0, 50.0, 50.0, db.nhmax_acc))
                assert ratio[k] == pytest.approx(expected, rel=1e-6), (db.bin_id, sigma)  # constants differ in the 7th digit


def test_nk_equilibrium_is_robust_to_ic_noise(model_data):
    """The old solver's answer flipped between roots under tiny perturbations."""
    from pycalima.solvers.equilibrium import NewtonKrylovEquilibriumSolver

    results = []
    for eps in (0.0, 1e-9, -1e-9):
        state, y_gas, y_dust, procs = _cool_dense_g10_state()
        _, y, _ = NewtonKrylovEquilibriumSolver(f_tol=1e-300, f_rtol=1e-6).find_equilibrium(
            state, y_gas, y_dust * (1 + eps), procs, t_eq_s=400 * 3.156e13)
        results.append(y / y.sum())
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], atol=1e-3)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_short_rk4_run_conserves_mass_and_writes_output(model_data, tmp_path):
    """The full documented workflow, from a directory that is not the repo."""
    from pycalima.solvers.run_chemistry import compute_element_totals, run_chemistry

    results = run_chemistry(
        resolve_solver_config_path("example_ic"),
        t_end_Myr=1e-3,
        verbose=False,
        output_dir=str(tmp_path),
        save_txt=True,
    )
    assert results is not None

    written = sorted(p.name for p in tmp_path.iterdir())
    assert any(n.endswith(".txt") for n in written), f"no text output: {written}"

    # the run must have advanced time and stayed physical
    assert results["t_end_s"] > 0
    for key in ("y_gas_init", "y_gas_final", "y_dust_init", "y_dust_final"):
        arr = np.asarray(results[key])
        assert np.all(np.isfinite(arr)), f"{key} has non-finite entries"
        assert np.all(arr >= 0), f"{key} went negative"

    # total mass across gas and dust is conserved by construction
    m0 = float(np.sum(results["y_gas_init"]) + np.sum(results["y_dust_init"]))
    m1 = float(np.sum(results["y_gas_final"]) + np.sum(results["y_dust_final"]))
    assert m1 == pytest.approx(m0, rel=1e-8), (
        f"total mass drifted from {m0:.6e} to {m1:.6e}"
    )


def test_run_chemistry_accepts_a_bundled_config_by_name(model_data, tmp_path):
    """`calima-run example_ic` relies on this resolution."""
    from pycalima.solvers.run_chemistry import run_chemistry

    results = run_chemistry(
        resolve_solver_config_path("example_ic"),
        t_end_Myr=1e-4,
        verbose=False,
        output_dir=str(tmp_path),
        save_txt=False,
    )
    assert results is not None


def test_grid_runner_produces_the_documented_arrays(model_data, tmp_path):
    from pycalima.solvers.run_grid import run_grid

    grid = run_grid(
        config_path=resolve_solver_config_path("example_ic"),
        x_param="T", x_values=[100.0, 1000.0],
        y_param="nH", y_values=[1.0, 10.0],
        t_end_Myr=1e-4,
        solver_type="rk4",
        verbose=False,
        n_jobs=1,
    )
    for key in ("DTM", "rho_dust", "rho_pah", "rho_gas", "x_values", "y_values",
                "x_param", "y_param", "elapsed_s"):
        assert key in grid, f"grid is missing {key!r}"
    assert np.asarray(grid["DTM"]).shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(grid["DTM"])))
    assert np.all(np.asarray(grid["DTM"]) >= 0), "dust-to-metal ratio cannot be negative"


def test_grid_npz_round_trip(model_data, tmp_path):
    from pycalima.solvers.run_grid import load_grid_npz, run_grid, save_grid_npz

    grid = run_grid(
        config_path=resolve_solver_config_path("example_ic"),
        x_param="T", x_values=[100.0],
        y_param="nH", y_values=[1.0],
        t_end_Myr=1e-4,
        solver_type="rk4",
        verbose=False,
        n_jobs=1,
    )
    out = save_grid_npz(grid, str(tmp_path / "grid.npz"))
    reloaded = load_grid_npz(out)
    assert np.allclose(np.asarray(reloaded["DTM"]), np.asarray(grid["DTM"]))


# ---------------------------------------------------------------------------
# solver registry: every solver reachable, and documented
# ---------------------------------------------------------------------------

EXPECTED_SOLVERS = {
    "rk4": "RK4Solver",
    "rk54": "RK54Solver",
    "anninos": "AnninosSolver",
    "mprk22": "MPRK22Solver",
    "newton_krylov": "NewtonKrylovEquilibriumSolver",
    "sparse_newton": "SparseNewtonEquilibriumSolver",
}


def test_registry_contains_every_solver():
    from pycalima.solvers.run_chemistry import SOLVER_REGISTRY

    assert set(SOLVER_REGISTRY) == set(EXPECTED_SOLVERS), (
        f"registry is {sorted(SOLVER_REGISTRY)}, expected {sorted(EXPECTED_SOLVERS)}"
    )
    for key, cls_name in EXPECTED_SOLVERS.items():
        assert SOLVER_REGISTRY[key].__name__ == cls_name


@pytest.mark.parametrize("key", sorted(EXPECTED_SOLVERS))
def test_every_registered_solver_instantiates(key):
    from pycalima.solvers.run_chemistry import _make_solver

    solver = _make_solver({"type": key})
    assert solver is not None


def test_make_solver_rejects_an_unknown_type():
    from pycalima.solvers.run_chemistry import _make_solver

    with pytest.raises(ValueError, match="Unknown solver type"):
        _make_solver({"type": "not_a_solver"})


def test_both_clis_offer_the_same_solver_choices():
    """calima-run used to have no --solver flag at all, while calima-grid did,
    so a documented `calima-run ... --solver newton_krylov` failed with
    'unrecognized arguments'."""
    import contextlib
    import importlib
    import io

    # NB: `from pycalima.solvers import run_chemistry` binds the re-exported
    # *function*, not the module, because solvers/__init__.py shadows the name.
    rc = importlib.import_module("pycalima.solvers.run_chemistry")
    rg = importlib.import_module("pycalima.solvers.run_grid")

    # Both build their parser inside the entry point, so compare the --help
    # output, which is also what a user actually sees.
    found = {}
    for name, fn in (("calima-run", rc.main), ("calima-grid", rg.cli)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
            fn(["--help"])
        text = buf.getvalue()
        assert "--solver" in text, f"{name} does not expose --solver"
        found[name] = {s for s in EXPECTED_SOLVERS if s in text}

    for name, solvers in found.items():
        assert solvers == set(EXPECTED_SOLVERS), (
            f"{name} --help advertises {sorted(solvers)}, "
            f"expected {sorted(EXPECTED_SOLVERS)}"
        )


@pytest.mark.parametrize("key", sorted(EXPECTED_SOLVERS))
def test_solver_type_override_reaches_the_run(key, model_data, tmp_path):
    """--solver must override the config's solver.type."""
    from pycalima.solvers.run_chemistry import run_chemistry

    results = run_chemistry(
        resolve_solver_config_path("example_ic"),
        t_end_Myr=1e-4,
        verbose=False,
        output_dir=str(tmp_path),
        save_txt=False,
        solver_type=key,
    )
    assert results is not None
    for field in ("y_gas_final", "y_dust_final"):
        arr = np.asarray(results[field])
        assert np.all(np.isfinite(arr)) and np.all(arr >= 0), f"{key}: bad {field}"


def test_run_chemistry_rejects_an_unknown_solver_override(model_data):
    from pycalima.solvers.run_chemistry import run_chemistry

    with pytest.raises(ValueError, match="Unknown solver type"):
        run_chemistry(
            resolve_solver_config_path("example_ic"),
            verbose=False,
            solver_type="not_a_solver",
        )


def test_the_docs_document_every_solver():
    """The gap this guards: the solver table once listed only rk4,
    newton_krylov and sparse_newton, omitting rk54 and anninos entirely.

    The table now lives in docs/guide/solvers.md rather than the README, so
    that is what gets checked.
    """
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "docs" / "guide" / "solvers.md"
    if not page.is_file():
        pytest.skip("documentation sources are not part of the wheel")
    text = page.read_text(encoding="utf-8")
    missing = [k for k in EXPECTED_SOLVERS if f"`{k}`" not in text]
    assert not missing, f"docs/guide/solvers.md omits solver types: {missing}"
    missing_cls = [c for c in EXPECTED_SOLVERS.values() if c not in text]
    assert not missing_cls, f"docs/guide/solvers.md omits solver classes: {missing_cls}"


# ---------------------------------------------------------------------------
# Regression: rate tables that silently failed to load
#
# Four independent defects each left a rate table unloaded, and every one was
# swallowed -- `if not fname.exists(): continue` in dust_init, or a bare
# `except Exception: pass`. The visible effect was a process with an
# identically zero rate, which looks exactly like physics.
#
#   1. _load_sputtering_tables asked for "thermal_sputtering_<bin>_Z_<Z>";
#      dust_sputtering.py:1061 writes "sputtering_<bin>_Z_<Z>". Dust thermal
#      sputtering was therefore disabled for every bin.
#   2. read_pah_sputtering_table did not skip the "#" provenance header the
#      exporters write, so int(lines[0]) raised. PAH sputtering was disabled.
#   3. read_pah_photolysis_table had the same defect. PAH photolysis was
#      disabled.
#   4. Solver configs spell PAH bins "PAHbin_01"; exporters write "PAHBin_01".
#      An exact lookup finds the file on a case-insensitive filesystem (macOS)
#      and misses it on a case-sensitive one (Linux, hence CI).
# ---------------------------------------------------------------------------

def test_dust_sputtering_tables_are_actually_loaded(model_data):
    """Every dust bin must end up with a non-empty sputtering interpolator set."""
    from pycalima._paths import resolve_solver_config_path
    from pycalima.solvers.dust_init import load_initial_conditions

    state, _y_gas, _y_dust = load_initial_conditions(
        resolve_solver_config_path("example_ic")
    )
    empty = [db.bin_id for db in state.dust_bins if not db.sputtering_interps]
    assert not empty, (
        f"no thermal sputtering tables loaded for {empty}; the solver would "
        f"silently report a zero sputtering rate"
    )


def test_pah_sputtering_and_photolysis_tables_are_actually_loaded(model_data):
    from pycalima._paths import resolve_solver_config_path
    from pycalima.solvers.dust_init import load_initial_conditions

    state, _y_gas, _y_dust = load_initial_conditions(
        resolve_solver_config_path("example_ic")
    )
    no_sput = [pb.bin_id for pb in state.pah_bins if not pb.sputtering_interps]
    no_diss = [pb.bin_id for pb in state.pah_bins if pb.dissociation_interp is None]
    assert not no_sput, f"no PAH sputtering tables loaded for {no_sput}"
    assert not no_diss, f"no PAH dissociation tables loaded for {no_diss}"


def test_pah_readers_skip_the_provenance_header(tmp_path):
    """The exporters prepend '#' lines; both PAH readers must ignore them."""
    from pycalima.solvers.table_io import (
        read_pah_photolysis_table,
        read_pah_sputtering_table,
    )

    sput = tmp_path / "sputtering_PAHBin_09_Z_1"
    sput.write_text(
        "# PAH Sputtering rates (Simple, no phi dependence)\n"
        "# Script: models/PAH_gas_collisions/PAH_sputtering.py\n"
        "# Git: somebranch (abc1234)\n"
        "3\n"
        "3.0\n4.0\n5.0\n"
        "-12.0\n-11.0\n-10.0\n",
        encoding="utf-8",
    )
    log_T, log_J = read_pah_sputtering_table(sput)
    assert list(log_T) == [3.0, 4.0, 5.0]
    assert list(log_J) == [-12.0, -11.0, -10.0]

    # The photolysis reader shares the defect; assert it parses a header too.
    diss = tmp_path / "dissociation_PAHBin_09.dat"
    diss.write_text(
        "# PAH photodissociation rate table\n"
        "# Git: somebranch (abc1234)\n"
        "2 2\n"
        "0.0\n1.0\n"
        "1.0\n2.0\n"
        "-9.0\n-8.0\n"
        "-7.0\n-6.0\n",
        encoding="utf-8",
    )
    # Shape conventions differ between versions of the writer; all this test
    # asserts is that the header no longer makes it raise.
    try:
        read_pah_photolysis_table(diss)
    except ValueError as exc:
        assert "#" not in str(exc), f"header still reaching the parser: {exc}"


def test_table_lookup_tolerates_the_pah_bin_id_case_difference(tmp_path):
    """Solver configs say PAHbin_01, exporters write PAHBin_01."""
    from pycalima.solvers.dust_init import _resolve_table

    (tmp_path / "sputtering_PAHBin_01_Z_1").write_text("1\n3.0\n-12.0\n")

    found = _resolve_table(tmp_path, "sputtering_PAHbin_01_Z_1")
    assert found.exists(), (
        "case-insensitive fallback failed; the exact-name lookup succeeds on "
        "macOS and fails on Linux, so this must be checked explicitly"
    )
    # Assert it is readable rather than asserting the name: on a
    # case-insensitive filesystem the exact path already resolves, so the
    # returned name keeps the requested spelling and never reaches the
    # fallback. What has to hold on both kinds of filesystem is that the
    # bytes come back.
    assert found.read_text(encoding="utf-8") == "1\n3.0\n-12.0\n"

    # A genuinely absent table must still come back non-existent, so that
    # callers can treat it as "not exported yet".
    missing = _resolve_table(tmp_path, "sputtering_DustBin_77_Z_1")
    assert not missing.exists()


# ---------------------------------------------------------------------------
# thermal sublimation: the table is log10(epsilon) against *dust* temperature
# ---------------------------------------------------------------------------

def _write_sublimation_table(path, T, col2, log10=True, a_cm=1.0e-5):
    unit = "log10(epsilon/[s^-1])" if log10 else "epsilon [s^-1]"
    lines = [
        f"# Dust bin: DustBin_01, Material: graphite, Grain radius "
        f"(representative): {a_cm:.6e} cm",
        f"# Columns: T_d [K] (dust temperature) | {unit}",
    ]
    lines += [f"{t:14.6E} {v:14.6E}" for t, v in zip(T, col2)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sublimation_reader_decodes_log10_tables(tmp_path):
    """The exporter writes log10(epsilon); reading it as linear epsilon made
    every rate below 1 s^-1 negative, and so silently zero."""
    from pycalima.solvers.dust_init import _load_erosion_rate_table

    T = np.array([1000.0, 1500.0, 2000.0, 3000.0])
    log_eps = np.array([-20.0, -12.0, -6.0, 0.5])
    _write_sublimation_table(tmp_path / "sublimation_rate_DustBin_01.dat", T, log_eps)

    interp, a_cm = _load_erosion_rate_table("DustBin_01", tmp_path)
    assert a_cm == pytest.approx(1.0e-5)
    np.testing.assert_allclose(interp(T), 10.0 ** log_eps, rtol=1e-12)
    assert interp(500.0) == 0.0, "below the table the grain must not sublimate"
    assert interp(5000.0) == pytest.approx(10.0 ** 0.5), "above it, hold the last row"
    # log-linear between nodes, so the midpoint is the geometric mean
    assert interp(1750.0) == pytest.approx(10.0 ** -9.0)


def test_sublimation_reader_still_reads_old_linear_tables(tmp_path):
    from pycalima.solvers.dust_init import _load_erosion_rate_table

    T = np.array([1000.0, 2000.0])
    eps = np.array([1.0e-20, 1.0e-6])
    _write_sublimation_table(
        tmp_path / "sublimation_rate_DustBin_01.dat", T, eps, log10=False
    )
    interp, _ = _load_erosion_rate_table("DustBin_01", tmp_path)
    np.testing.assert_allclose(interp(T), eps, rtol=1e-12)


def test_sublimation_tables_match_the_gd89_model(isolated_env, tmp_path):
    """Export -> solver round trip: the solver's epsilon(T_d) must be the
    model-layer |da/dt|/a for the radius the table was computed for."""
    from pycalima.models.dust_radiation.dust_sublimation import (
        _load_dust_bins, radius_loss_rate, write_sublimation_rate_tables)
    from pycalima.solvers.dust_init import _load_erosion_rate_table

    write_sublimation_rate_tables(output_dir=str(tmp_path), n_T=60)
    for meta in _load_dust_bins():
        interp, a_cm = _load_erosion_rate_table(meta["id"], tmp_path)
        material = str(meta["composition"]).lower()
        T_nodes = np.loadtxt(tmp_path / f"sublimation_rate_{meta['id']}.dat")[:, 0]
        for T in T_nodes[::5]:
            # the header and rows are printed to 7 significant figures
            expected = radius_loss_rate(material, a_cm, float(T)) / a_cm
            assert interp(T) == pytest.approx(expected, rel=1e-4), (meta["id"], T)


def _sublimation_config(tmp_path, model_data, **env):
    cfg = json.loads(resolve_solver_config_path("example_ic").read_text(encoding="utf-8"))
    cfg["model_data_dir"] = str(model_data)
    cfg["physics"] = {k: False for k in cfg.get("physics", {})}
    cfg["physics"]["dust_sublimation"] = True
    cfg["environment"].update(env)
    path = tmp_path / "subl_ic.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_sublimation_requires_a_dust_temperature(tmp_path, model_data):
    from pycalima.solvers.dust_init import load_initial_conditions

    with pytest.raises(ValueError, match="dust_temperature_K"):
        load_initial_conditions(_sublimation_config(tmp_path, model_data))


def test_sublimation_follows_the_dust_not_the_gas_temperature(tmp_path, model_data):
    """Hot gas around cold grains must not sublimate them (it used to destroy
    every grain within seconds in any gas above ~4000 K), and hot grains in
    cold gas must."""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.rhs import build_process_list, compute_rhs

    def rhs(T_gas, T_dust):
        state, y_gas, y_dust = load_initial_conditions(_sublimation_config(
            tmp_path, model_data, gas_temperature_K=T_gas, dust_temperature_K=T_dust))
        if not any(db.erosion_rate_interp for db in state.dust_bins):
            pytest.skip("no sublimation tables exported")
        procs = build_process_list(state)
        assert [p.name for p in procs] == ["sublimation"]
        dgas, ddust = compute_rhs(state, y_gas, y_dust, procs)
        return state, y_dust, dgas, ddust

    _, _, dgas, ddust = rhs(T_gas=1.0e6, T_dust=50.0)
    assert np.all(ddust == 0.0) and np.all(dgas == 0.0)

    state, y_dust, dgas, ddust = rhs(T_gas=10.0, T_dust=1800.0)
    for db in state.dust_bins:
        if db.erosion_rate_interp is None:
            continue
        i = db.bin_index + state.npah
        eps = float(db.erosion_rate_interp(1800.0))
        assert eps > 0.0
        assert ddust[i] == pytest.approx(-3.0 * eps * y_dust[i], rel=1e-12)
    # mass leaves the grains and arrives in the gas
    assert abs(dgas.sum() + ddust.sum()) <= 1e-10 * np.abs(ddust).sum()


def test_bin_dust_temperature_overrides_the_environment(tmp_path, model_data):
    from pycalima.solvers.dust_init import load_initial_conditions

    path = _sublimation_config(tmp_path, model_data, dust_temperature_K=100.0)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["dust_bins"][0]["dust_temperature_K"] = 1500.0
    path.write_text(json.dumps(cfg), encoding="utf-8")
    state, _, _ = load_initial_conditions(path)
    temps = [db.dust_temperature_K for db in state.dust_bins]
    assert temps[0] == 1500.0 and all(t == 100.0 for t in temps[1:])


# ---------------------------------------------------------------------------
# recondensation: the reverse of sublimation, balancing it at n_sat(T_d)
# ---------------------------------------------------------------------------

_KB = 1.3806488e-16
_M_C = 12.011 * 1.66053906660e-24


def _condensation_config(tmp_path, *, T_gas=1.0e4, T_dust=1700.0, nH=1.0e8,
                         dust_frac=0.99, physics=None, models=None, env=None):
    """One 0.1 um graphite bin holding 99% of the carbon, with a synthetic
    sublimation table log10(epsilon) = -12 + 0.01 (T_d - 1500)."""
    md = tmp_path / "md"
    (md / "dust_sublimation").mkdir(parents=True, exist_ok=True)
    T = np.linspace(1000.0, 3000.0, 201)
    _write_sublimation_table(md / "dust_sublimation" / "sublimation_rate_DustBin_01.dat",
                             T, -12.0 + 0.01 * (T - 1500.0))
    rho = 1.4 * nH * 1.6726219e-24
    x_C = 10.0 ** -3.44 * 12.011 / 1.4           # C mass fraction
    cfg = {
        "model_data_dir": str(md),
        "environment": {"gas_temperature_K": T_gas,
                        "hydrogen_number_density_cm3": nH,
                        "dust_temperature_K": T_dust, **(env or {})},
        "elemental_abundances": {"C": {"mass_fraction": x_C}},
        "dust_bins": [{"id": "DustBin_01", "composition": "graphite",
                       "grain_size_micron": 0.1, "grain_density_gcm3": 2.24,
                       "elements": [{"name": "C", "mass_fraction": 1.0}],
                       "initial_mass_density_gcm3": dust_frac * x_C * rho}],
        "physics": physics or {"dust_sublimation": True, "dust_condensation": True},
        "models": models or {},
    }
    path = tmp_path / "cond_ic.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _condensation_only(state, y_gas, y_dust):
    from pycalima.solvers.dust_rates import condensation_rate

    dgas, ddust = np.zeros_like(y_gas), np.zeros_like(y_dust)
    condensation_rate(state, y_gas, y_dust, dgas, ddust)
    return dgas, ddust


def test_condensation_is_the_kinetic_flux_of_the_key_species(tmp_path):
    """dm/dt = alpha pi a^2 n_C v_C(T_gas) m_C per grain, with the GD89 alpha."""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.rhs import build_process_list

    state, y_gas, y_dust = load_initial_conditions(_condensation_config(tmp_path))
    assert [p.name for p in build_process_list(state)] == ["sublimation", "condensation"]
    assert state.condensation_coefficient == 0.1

    db = state.dust_bins[0]
    n_C = y_gas[2] / _M_C
    v_C = np.sqrt(8.0 * _KB * state.local_Tk / (np.pi * _M_C))
    dm_dt = 0.1 * np.pi * db.asize_cm ** 2 * n_C * v_C * _M_C
    dgas, ddust = _condensation_only(state, y_gas, y_dust)
    assert ddust[0] == pytest.approx(dm_dt / db.mgrain * y_dust[0], rel=1e-10)
    assert dgas[2] == pytest.approx(-ddust[0], rel=1e-12)


def test_sublimation_and_condensation_balance_at_the_saturated_density(tmp_path):
    """Closed box: the gas-phase carbon relaxes to the density at which
    condensation cancels the tabulated evaporation, and the dust keeps the
    rest; hot enough that n_sat exceeds the carbon budget, the dust is gone."""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.ode_driver import integrate_dust_ode
    from pycalima.solvers.rhs import build_process_list
    from pycalima.solvers.rk54 import RK54Solver

    def relax(T_dust):
        state, y_gas, y_dust = load_initial_conditions(
            _condensation_config(tmp_path, T_dust=T_dust))
        g, d, _ = integrate_dust_ode(state, 1.0e10, y_gas, y_dust,
                                     build_process_list(state), RK54Solver(errmax=0.01),
                                     h_init=1.0, h_min=1.0e-6)
        assert g.sum() + d.sum() == pytest.approx(y_gas.sum() + y_dust.sum(), rel=1e-10)
        return state, g, d

    # 3 epsilon = alpha (3 / 4 a rho) sqrt(8 k T / pi) rho_C / sqrt(m_C)
    state, g, d = relax(1600.0)
    db = state.dust_bins[0]
    eps = float(db.erosion_rate_interp(1600.0))
    rho_sat = (4.0 * db.asize_cm * db.sgrain * eps * np.sqrt(_M_C)
               / (0.1 * np.sqrt(8.0 * _KB * state.local_Tk / np.pi)))
    assert 0.0 < rho_sat < g[2] + d[0], "test point must keep some dust"
    assert g[2] == pytest.approx(rho_sat, rel=1e-6)

    _, g, d = relax(1800.0)
    assert d[0] < 1e-6 * g[2]


def test_pressure_balance_scales_the_flux_by_sqrt_gas_over_dust_temperature(tmp_path):
    from pycalima.solvers.dust_init import load_initial_conditions

    def ddust(balance):
        state, y_gas, y_dust = load_initial_conditions(_condensation_config(
            tmp_path, T_gas=1.0e4, T_dust=1600.0,
            models={"condensation_balance": balance}))
        return _condensation_only(state, y_gas, y_dust)[1][0]

    assert ddust("pressure") / ddust("flux") == pytest.approx(np.sqrt(1.0e4 / 1600.0))


def test_condensing_ions_are_repelled_by_a_positive_grain(tmp_path):
    """Singly charged ions onto a grain at +U: suppressed by exp(-eU/kT)."""
    from pycalima.solvers.dust_init import load_initial_conditions

    def ddust(x_ion, U):
        state, y_gas, y_dust = load_initial_conditions(_condensation_config(
            tmp_path, models={"condensation_ion_fraction": x_ion},
            env={"grain_potential_V": U}))
        return _condensation_only(state, y_gas, y_dust)[1][0]

    kT_eV = _KB * 1.0e4 / 1.602176634e-12
    neutral = ddust(0.0, 2.0)
    assert ddust(1.0, 2.0) / neutral == pytest.approx(np.exp(-2.0 / kT_eV), rel=1e-4)
    assert ddust(0.5, 2.0) / neutral == pytest.approx(0.5 + 0.5 * np.exp(-2.0 / kT_eV), rel=1e-4)


def test_condensation_and_accretion_cannot_both_be_on(tmp_path):
    from pycalima.solvers.dust_init import load_initial_conditions

    with pytest.raises(ValueError, match="twice"):
        load_initial_conditions(_condensation_config(tmp_path, physics={
            "dust_sublimation": True, "dust_condensation": True, "dust_accretion": True}))
    with pytest.raises(ValueError, match="condensation_balance"):
        load_initial_conditions(_condensation_config(
            tmp_path, models={"condensation_balance": "pressur"}))


@pytest.mark.parametrize("solver_type", ["anninos", "newton_krylov"])
def test_run_chemistry_warns_against_solvers_that_cannot_balance_condensation(
        tmp_path, solver_type):
    from pycalima.solvers.run_chemistry import run_chemistry

    path = _condensation_config(tmp_path)
    with pytest.warns(UserWarning, match="does not reach the right equilibrium"):
        run_chemistry(path, t_end_Myr=1e-9, verbose=False, solver_type=solver_type)


# ---------------------------------------------------------------------------
# MPRK22 and the Fortran-parity Anninos
# ---------------------------------------------------------------------------

def _element_totals(state, y_gas, y_dust):
    from pycalima.solvers.equilibrium import _build_contribution_matrix

    return y_gas + _build_contribution_matrix(state) @ y_dust


def test_mprk22_reaches_the_saturated_density_in_a_few_steps(tmp_path):
    """The case Anninos cannot do: sublimation and condensation in balance.
    The gas relaxes to n_sat; MPRK22 is implicit in it and gets there in tens
    of steps however long the run."""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.mprk22 import MPRK22Solver
    from pycalima.solvers.ode_driver import integrate_dust_ode
    from pycalima.solvers.rhs import build_process_list

    state, y_gas, y_dust = load_initial_conditions(
        _condensation_config(tmp_path, T_dust=1600.0))
    db = state.dust_bins[0]
    eps = float(db.erosion_rate_interp(1600.0))
    rho_sat = (4.0 * db.asize_cm * db.sgrain * eps * np.sqrt(_M_C)
               / (0.1 * np.sqrt(8.0 * _KB * state.local_Tk / np.pi)))
    for t_end in (1.0e10, 1.0e17):
        g, d, diag = integrate_dust_ode(state, t_end, y_gas, y_dust,
                                        build_process_list(state), MPRK22Solver(),
                                        h_init=1.0, h_min=1.0e-6)
        assert g[2] == pytest.approx(rho_sat, rel=1e-3)
        assert diag["naccepted"] < 200
        np.testing.assert_allclose(_element_totals(state, g, d),
                                   _element_totals(state, y_gas, y_dust), rtol=1e-12)


@pytest.mark.parametrize("config", ["example_ic", "all_processes_test"])
def test_mprk22_conserves_each_element_and_matches_rk54(config, model_data):
    from pycalima.solvers.dust_init import SEC2MYR, load_initial_conditions
    from pycalima.solvers.mprk22 import MPRK22Solver
    from pycalima.solvers.ode_driver import integrate_dust_ode
    from pycalima.solvers.rhs import build_process_list
    from pycalima.solvers.rk54 import RK54Solver

    state, y_gas, y_dust = load_initial_conditions(resolve_solver_config_path(config))
    procs = build_process_list(state)
    kw = dict(h_init=1e10, h_min=1.0, h_max=SEC2MYR)
    g, d, _ = integrate_dust_ode(state, 10 * SEC2MYR, y_gas, y_dust, procs,
                                 MPRK22Solver(), **kw)
    _, d_ref, _ = integrate_dust_ode(state, 10 * SEC2MYR, y_gas, y_dust, procs,
                                     RK54Solver(errmax=1e-3), **kw)
    M0 = _element_totals(state, y_gas, y_dust)
    np.testing.assert_allclose(_element_totals(state, g, d), M0,
                               rtol=1e-12, atol=1e-12 * M0.max())
    big = d_ref > 1e-6 * d_ref.max()
    np.testing.assert_allclose(d[big], d_ref[big], rtol=1e-2)


def test_anninos_moves_the_actual_bin_change_into_the_gas(tmp_path):
    """As the Fortran anninos_step: gas + dust is conserved per element in
    every branch (the Euler gas update created mass when sources and sinks
    nearly cancel)."""
    from pycalima.solvers.anninos import AnninosSolver
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.ode_driver import integrate_dust_ode
    from pycalima.solvers.rhs import build_process_list

    state, y_gas, y_dust = load_initial_conditions(
        _condensation_config(tmp_path, T_dust=1600.0))
    g, d, _ = integrate_dust_ode(state, 1.0e10, y_gas, y_dust,
                                 build_process_list(state), AnninosSolver(),
                                 h_init=1.0, h_min=1.0e-6)
    np.testing.assert_allclose(_element_totals(state, g, d),
                               _element_totals(state, y_gas, y_dust), rtol=1e-12)


def test_mprk22_takes_grains_that_vanish_within_one_step(tmp_path):
    """Grains far above their sublimation temperature empty within the
    shortest step; their relative error must not be measured against ~0
    (every step used to be rejected until countmax)."""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.mprk22 import MPRK22Solver
    from pycalima.solvers.ode_driver import integrate_dust_ode
    from pycalima.solvers.rhs import build_process_list

    state, y_gas, y_dust = load_initial_conditions(
        _condensation_config(tmp_path, T_dust=2900.0, nH=1e4))
    g, d, diag = integrate_dust_ode(state, 3.0e7, y_gas, y_dust,
                                    build_process_list(state), MPRK22Solver(),
                                    h_init=3.0e6, h_min=1.0)
    assert diag["tau"] == pytest.approx(3.0e7)
    assert diag["naccepted"] < 100
    assert d[0] < 1e-6 * y_dust[0]
    np.testing.assert_allclose(_element_totals(state, g, d),
                               _element_totals(state, y_gas, y_dust), rtol=1e-12)


# ---------------------------------------------------------------------------
# nucleation: new grains from supersaturated gas
# ---------------------------------------------------------------------------

def _nucleation_config(tmp_path, *, dust_frac=0.0, physics=None, **entry):
    from pycalima.models.dust_radiation.dust_sublimation import nucleation_parameters

    path = _condensation_config(tmp_path, dust_frac=dust_frac, physics=physics or {
        "dust_sublimation": True, "dust_condensation": True, "dust_nucleation": True},
        **{k: entry.pop(k) for k in ("T_dust", "nH", "T_gas") if k in entry})
    cfg = json.loads(path.read_text(encoding="utf-8"))
    p = nucleation_parameters("graphite")
    cfg["nucleation"] = [{"target_bin": "DustBin_01", "key_element": "C",
                          "evaporation_prefactor_cm2s": p["evaporation_prefactor_cm2s"],
                          "binding_energy_K": p["binding_energy_K"],
                          "surface_energy_K": p["surface_energy_K"],
                          "monomer_mass_amu": p["monomer_mass_amu"],
                          "density_gcm3": p["density_gcm3"], **entry}]
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _kernel(fn, state, y_gas, y_dust):
    dgas, ddust = np.zeros_like(y_gas), np.zeros_like(y_dust)
    fn(state, y_gas, y_dust, dgas, ddust)
    return dgas, ddust


@pytest.mark.parametrize("T_dust", [1300.0, 1500.0, 1650.0])
def test_nucleation_kernel_matches_the_model_rate(tmp_path, T_dust):
    from pycalima.models.dust_radiation.dust_sublimation import steady_state_nucleation_rate
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.dust_rates import nucleation_rate

    state, y_gas, y_dust = load_initial_conditions(
        _nucleation_config(tmp_path, T_dust=T_dust, nH=1e10))
    n_C = y_gas[2] / _M_C
    J, _ = steady_state_nucleation_rate("graphite", n_C, T_dust, state.local_Tk)
    dgas, ddust = _kernel(nucleation_rate, state, y_gas, y_dust)
    assert J > 0
    assert ddust[0] == pytest.approx(J * state.dust_bins[0].mgrain, rel=1e-6)
    assert dgas[2] == pytest.approx(-ddust[0], rel=1e-12)


def test_critical_cluster_is_where_gd89_evaporation_balances_growth():
    """n* = (2 mu / 3 ln S)^3 with mu = 1.5 sigma_GD89 / T, so that
    S = exp(sigma_GD89 n*^-1/3 / T): the Kelvin balance of the sublimation
    tables' B(N) = B - sigma N^-1/3."""
    from pycalima.models.dust_radiation import dust_sublimation as ds

    sig = ds.GD89_PARAMS["graphite"]["surface_free_energy_K"]
    for T in (1400.0, 1600.0, 1700.0):
        n = 10 ** -3.44 * 1e10
        S = float(ds.supersaturation("graphite", n, T, 1e4))
        _, n_star = ds.steady_state_nucleation_rate("graphite", n, T, 1e4)
        assert S == pytest.approx(np.exp(sig * n_star ** (-1 / 3) / T), rel=1e-9)


def test_nucleation_never_exceeds_the_monomer_collision_rate():
    """However supersaturated, clusters cannot form faster than monomers
    meet; where the critical cluster is more than a fraction of an atom
    (T >~ 1000 K here) the rate rises steeply as the clusters cool."""
    from pycalima.models.dust_radiation import dust_sublimation as ds

    n, T_gas = 10 ** -3.44 * 1e10, 1e4
    a0 = (3 * 12.011 * 1.66053906660e-24 / 2.24 / (4 * np.pi)) ** (1 / 3)
    v = np.sqrt(8 * _KB * T_gas / (np.pi * _M_C))
    J_coll = 0.5 * 0.1 * n ** 2 * np.pi * (2 * a0) ** 2 * np.sqrt(2) * v
    J = [ds.steady_state_nucleation_rate("graphite", n, T, T_gas)[0]
         for T in (300.0, 600.0, 1000.0, 1400.0, 1700.0)]
    assert all(j <= J_coll * (1 + 1e-12) for j in J)
    assert J[2] > J[3] > J[4]


def test_no_nucleation_from_undersaturated_or_ionised_gas(tmp_path):
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.dust_rates import nucleation_rate

    for kw in ({"T_dust": 2500.0}, {"T_dust": 1500.0, "neutral_fraction": 0.0}):
        state, y_gas, y_dust = load_initial_conditions(_nucleation_config(tmp_path, **kw))
        _, ddust = _kernel(nucleation_rate, state, y_gas, y_dust)
        assert ddust[0] == 0.0, kw


def test_nucleation_config_is_validated(tmp_path):
    from pycalima.solvers.dust_init import load_initial_conditions

    with pytest.raises(ValueError, match="target_bin"):
        load_initial_conditions(_nucleation_config(tmp_path, target_bin="DustBin_99"))
    with pytest.raises(ValueError, match="key_element"):
        load_initial_conditions(_nucleation_config(tmp_path, key_element="Si"))
    path = _condensation_config(tmp_path, physics={"dust_sublimation": True,
                                                   "dust_nucleation": True})
    with pytest.raises(ValueError, match='"nucleation"'):
        load_initial_conditions(path)


def test_dust_forms_from_pure_gas_and_settles_at_saturation(tmp_path):
    """Closed box with no dust: nucleation seeds grains, condensation grows
    them, and the gas ends at the saturated density, conserving carbon.
    Without nucleation nothing happens. (At n_H = 1e10 and 1400 K the
    critical cluster is ~1 atom; at 1e8 and 1600 K it is ~180 and nothing
    nucleates.)"""
    from pycalima.solvers.dust_init import load_initial_conditions
    from pycalima.solvers.mprk22 import MPRK22Solver
    from pycalima.solvers.ode_driver import integrate_dust_ode
    from pycalima.solvers.rhs import build_process_list

    state, y_gas, y_dust = load_initial_conditions(
        _nucleation_config(tmp_path, T_dust=1400.0, nH=1e10))
    assert y_dust[0] == 0.0
    g, d, _ = integrate_dust_ode(state, 1e9, y_gas, y_dust, build_process_list(state),
                                 MPRK22Solver(), h_init=1e-3, h_min=1e-9)
    db = state.dust_bins[0]
    eps = float(db.erosion_rate_interp(1400.0))
    rho_sat = (4.0 * db.asize_cm * db.sgrain * eps * np.sqrt(_M_C)
               / (0.1 * np.sqrt(8.0 * _KB * state.local_Tk / np.pi)))
    assert d[0] > 0.5 * y_gas[2]
    assert g[2] == pytest.approx(rho_sat, rel=1e-2)
    np.testing.assert_allclose(_element_totals(state, g, d),
                               _element_totals(state, y_gas, y_dust), rtol=1e-12)

    state, y_gas, y_dust = load_initial_conditions(_nucleation_config(
        tmp_path, T_dust=1400.0, nH=1e10,
        physics={"dust_sublimation": True, "dust_condensation": True}))
    _, d, _ = integrate_dust_ode(state, 1e9, y_gas, y_dust, build_process_list(state),
                                 MPRK22Solver(), h_init=1e-3, h_min=1e-9)
    assert d[0] == 0.0
