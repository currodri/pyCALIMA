"""Importing pycalima must not touch the filesystem or global matplotlib state.

Two module-scope ``os.makedirs`` calls used to run at *import* time and write
into the package's own directory, which raises PermissionError on a read-only
install prefix. Eleven modules also set ``text.usetex = True`` at import,
silently reconfiguring matplotlib for the whole process and breaking every
later ``savefig`` on a machine with no LaTeX. These tests keep both shut.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import pycalima

# Modules whose import legitimately requires an optional extra or external data.
KNOWN_OPTIONAL = {
    "pycalima.models.tools.eq_analysis",  # needs yt: pip install 'pycalima[sim]'
}


def _all_modules() -> list[str]:
    return sorted(
        m.name
        for m in pkgutil.walk_packages(pycalima.__path__, "pycalima.")
        if m.name not in KNOWN_OPTIONAL
    )


ALL_MODULES = _all_modules()


def test_module_discovery_is_sane():
    """Guard against the walk silently finding nothing."""
    assert len(ALL_MODULES) > 100, f"only found {len(ALL_MODULES)} modules"


@pytest.mark.parametrize("name", ALL_MODULES)
def test_import_is_side_effect_free(name, tmp_path, monkeypatch):
    """Importing from an empty CWD must succeed and write nothing.

    ``monkeypatch.chdir`` proves the module has no CWD dependence; the final
    assertion proves it performed no import-time writes.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delenv("CALIMA_DATA", raising=False)
    monkeypatch.delenv("CALIMA_MODEL_DATA", raising=False)
    monkeypatch.delenv("CALIMA_RESULTS", raising=False)

    importlib.import_module(name)

    leftovers = sorted(p.name for p in tmp_path.iterdir())
    assert not leftovers, f"{name} wrote {leftovers} into the CWD at import time"


def test_no_module_scope_mkdir():
    """No module executes mkdir/makedirs at import time.

    Checked statically rather than by import, so it holds for every module at
    once and cannot be masked by an earlier import in the same process.
    """
    import ast
    from pathlib import Path

    class ModuleScope(ast.NodeVisitor):
        """Visit only statements that run at import time."""

        def __init__(self):
            self.hits: list[int] = []

        # do not descend into callables or classes
        def visit_FunctionDef(self, node):
            pass

        def visit_AsyncFunctionDef(self, node):
            pass

        def visit_ClassDef(self, node):
            pass

        def visit_If(self, node):
            """Skip `if __name__ == "__main__":`, which does not run on import."""
            test = node.test
            is_main_guard = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and any(
                    isinstance(c, ast.Constant) and c.value == "__main__"
                    for c in test.comparators
                )
            )
            if is_main_guard:
                for stmt in node.orelse:
                    self.visit(stmt)
                return
            self.generic_visit(node)

        def visit_Call(self, node):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in ("makedirs", "mkdir"):
                self.hits.append(node.lineno)
            self.generic_visit(node)

    offenders = []
    root = Path(pycalima.__path__[0])
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        visitor = ModuleScope()
        for stmt in tree.body:
            visitor.visit(stmt)
        offenders += [f"{path.relative_to(root)}:{ln}" for ln in visitor.hits]

    assert not offenders, "import-time mkdir/makedirs: " + ", ".join(offenders)


def test_importing_physics_does_not_enable_latex():
    """rcParams must survive importing the modules that used to mutate it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (
        "pycalima.models.dust_radiation.dust_oppacity",
        "pycalima.models.PAH_radiation.pah_oppacity",
        "pycalima.models.PAH_charge.PAH_photoelectric_heating",
        "pycalima.models.dust_charge.dust_charging",
    ):
        importlib.import_module(name)

    assert plt.rcParams["text.usetex"] is False, (
        "importing a physics module set text.usetex=True globally; style must "
        "be opt-in via pycalima.plotting_style.use_calima_style()"
    )
