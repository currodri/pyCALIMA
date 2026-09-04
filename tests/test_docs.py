"""Structural checks on the documentation sources.

These do not build the site -- that is CI's job, and it takes minutes. What
they check is the two things that rot silently: a new module that lands in
neither the API reference nor the scripts index, and content that must never
be published finding its way into docs/.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
PKG = REPO / "src" / "pycalima"

pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="documentation sources are not part of the wheel"
)

# Mirrors docs/_ext/calima_scripts.py. A module matching this is a figure,
# diagnostic or ad-hoc script and belongs in the scripts index, not the API.
SCRIPT_RE = re.compile(
    r"^(compare_|reproduce_|diagnose_|plot_|scan_|profile_|benchmark_|check_|"
    r"make_|run_eqtemp|fit_|download_|dustregime_|test_|run_acetylene|export_)"
)

# galaxysam/SUMMARY.md carries an unfilled "- Python conversion: [Your name]"
# placeholder; optical_props/semenov_2003/Readme.txt carries a third party's
# personal email address. Neither may be published.
FORBIDDEN_SUBSTRINGS = ["[Your name]"]
# Deliberately general: the address actually at risk here is
# dima@astro.uni-jena.de, which a hyphen-free domain pattern misses.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}")
# Addresses that may legitimately appear (the maintainer's own, and
# obvious placeholders in examples).
EMAIL_ALLOWED = ("currodri@uchicago.edu", "you@example.com",
                 "user@example.com", "noreply@")


def _library_modules() -> set[str]:
    out = set()
    for path in PKG.rglob("*.py"):
        if path.name == "__init__.py" or SCRIPT_RE.match(path.stem):
            continue
        rel = path.relative_to(PKG).with_suffix("")
        out.add("pycalima." + str(rel).replace("/", "."))
    return out


def _documented_modules() -> set[str]:
    out = set()
    for page in (DOCS / "api").glob("*.md"):
        for line in page.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token.startswith("pycalima.") and " " not in token:
                out.add(token)
    return out


def test_api_pages_exist():
    pages = list((DOCS / "api").glob("*.md"))
    assert len(pages) >= 15, f"only {len(pages)} api pages found"


def test_every_library_module_is_in_the_api_reference():
    """A new physics module must be added to an api/*.md list.

    If this fails, either add the module to the relevant page or -- if it is a
    script rather than API -- give it a name SCRIPT_RE recognises, which also
    routes it to docs/reference/scripts.md automatically.
    """
    missing = sorted(_library_modules() - _documented_modules())
    # galaxysam/ROUTINES_MAPPING is an IDL-to-Python migration map, not API.
    missing = [m for m in missing if not m.endswith("ROUTINES_MAPPING")]
    assert not missing, "modules absent from docs/api/: " + ", ".join(missing)


def test_api_reference_lists_no_module_that_does_not_exist():
    stale = sorted(_documented_modules() - _library_modules())
    # export_all_grain_data is deliberately documented: it backs calima-export.
    stale = [m for m in stale if not m.endswith("export_all_grain_data")]
    assert not stale, "docs/api/ lists nonexistent modules: " + ", ".join(stale)


def test_docs_sources_carry_no_forbidden_content():
    offenders = []
    for path in DOCS.rglob("*.md"):
        if "_build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO)}: {needle!r}")
        for match in EMAIL_RE.finditer(text):
            address = match.group(0)
            if any(ok in address for ok in EMAIL_ALLOWED):
                continue
            offenders.append(f"{path.relative_to(REPO)}: email {address!r}")
    assert not offenders, "; ".join(offenders)


def test_no_relative_anchor_links_survive():
    """README-style `](#anchor)` links do not work across Sphinx pages."""
    offenders = []
    for path in DOCS.rglob("*.md"):
        if "_build" in path.parts:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            if "](#" in line:
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert not offenders, "relative anchors: " + ", ".join(offenders)


def _outside_fences(text: str):
    """Yield (line_number, line) for lines outside fenced code blocks.

    MyST does not parse math inside a fence, so a bare $VAR there is safe --
    and the RAMSES notebooks' error messages legitimately contain one.
    """
    in_fence = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", ":::")):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield number, line


def test_env_vars_are_backticked():
    """`dollarmath` turns two bare $VARs on one line into a math span.

    The variable names are silently swallowed, with no warning, so this is
    checked rather than left to review.
    """
    pattern = re.compile(r"(?:^|[^`$])\$CALIMA_[A-Z_]+")
    offenders = []
    for path in DOCS.rglob("*.md"):
        if "_build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in _outside_fences(text):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert not offenders, "unbackticked $CALIMA_*: " + ", ".join(offenders)


def test_warning_budget_is_a_plain_integer():
    budget = (DOCS / "warning-budget.txt").read_text(encoding="utf-8").strip()
    assert budget.isdigit(), f"warning-budget.txt holds {budget!r}"


def test_the_five_console_scripts_expose_a_parser():
    """sphinx-argparse renders _build_parser(); a rename would silently empty
    the CLI reference rather than fail the build."""
    import importlib

    targets = [
        ("pycalima._paths", "calima-paths"),
        ("pycalima._datasets", "calima-fetch-data"),
        ("pycalima.models.export_all_grain_data", "calima-export"),
        ("pycalima.solvers.run_chemistry", "calima-run"),
        ("pycalima.solvers.run_grid", "calima-grid"),
    ]
    for module_name, prog in targets:
        module = importlib.import_module(module_name)
        builder = getattr(module, "_build_parser", None)
        assert builder is not None, f"{module_name} has no _build_parser"
        assert builder().prog == prog, f"{module_name}: prog is not {prog}"
