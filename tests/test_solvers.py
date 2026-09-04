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

@pytest.mark.parametrize("name", ["rk4", "rk54", "anninos"])
def test_time_integrators_construct(name):
    from pycalima.solvers.anninos import AnninosSolver
    from pycalima.solvers.rk4 import RK4Solver
    from pycalima.solvers.rk54 import RK54Solver

    cls = {"rk4": RK4Solver, "rk54": RK54Solver, "anninos": AnninosSolver}[name]
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


def test_readme_documents_every_solver():
    """The gap this section exists to prevent: the README's solver table listed
    only rk4, newton_krylov and sparse_newton."""
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    if not readme.is_file():
        pytest.skip("README.md not present (installed copy)")
    text = readme.read_text(encoding="utf-8")
    missing = [k for k in EXPECTED_SOLVERS if f"`{k}`" not in text]
    assert not missing, f"README does not document solver types: {missing}"
    missing_cls = [c for c in EXPECTED_SOLVERS.values() if c not in text]
    assert not missing_cls, f"README does not name solver classes: {missing_cls}"
