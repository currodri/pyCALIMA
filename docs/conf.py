"""Sphinx configuration for pyCALIMA.

Every non-default setting here works around something measured in this
codebase, and the reason is recorded inline so that nobody "tidies" it away.

* 13 of 18 ``__init__.py`` files are empty, and ``src/pycalima/__init__.py``
  deliberately re-exports nothing -- its docstring explains that three live
  circular dependencies are resolved only by deferred function-local imports.
  So ``automodule:: pycalima :members:`` documents nothing useful, and the API
  reference is driven by the explicit module lists in ``api/*.md``. Never by
  discovery, and never by sphinx-apidoc.
* 58 modules are script-like (``if __name__ == "__main__"``), 34 of them
  figure-reproduction scripts. They are simply absent from those lists.
* 66 non-standard dashed section headings across 42 distinct names appear in
  docstrings ("Usage" x14, "Modified variables" x6, ...).
  ``napoleon_custom_sections`` renders each as a rubric instead of raising
  "Unexpected section title" inside autodoc.
* 78 single-backtick spans are meant as literals, so ``default_role =
  "literal"`` fixes all of them at no cost.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment hygiene. Must run before autodoc imports anything.
# ---------------------------------------------------------------------------
# Autodoc imports ~65 physics modules, nearly all of which import pyplot. Pin
# a non-interactive backend for THIS process only.
#
# Do NOT set $MPLBACKEND here. myst-nb executes notebooks in a subprocess
# kernel that inherits the environment, and with MPLBACKEND=Agg the inline
# backend's automatic figure display never runs -- measured: all 11 figures in
# calima_dust_pah_processes.ipynb vanish, silently, with no warning and no
# error. matplotlib.use() affects only the Sphinx process.
import matplotlib

matplotlib.use("Agg")

# Two modules touch the filesystem at import. Give the build a throwaway
# writable root so nothing can land in the checkout. setdefault, so that CI's
# CALIMA_MODEL_DATA (needed to execute the notebooks) still wins -- the finer
# grained variable beats CALIMA_DATA in pycalima._paths.
os.environ.setdefault(
    "CALIMA_DATA", str(Path(tempfile.gettempdir()) / "calima-docs-build")
)

HERE = Path(__file__).parent.resolve()
REPO = HERE.parent
sys.path.insert(0, str(HERE / "_ext"))
# Redundant under `pip install -e .`, but makes a bare `sphinx-build docs
# docs/_build/html` work in a fresh clone.
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
project = "pyCALIMA"
author = "Curro Rodriguez Montero"
project_copyright = "2026, Curro Rodriguez Montero"

try:
    from importlib.metadata import version as _dist_version

    release = _dist_version("pycalima")
except Exception:  # not installed as a distribution
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    # myst_nb provides the MyST markdown parser AND notebook execution. Do not
    # also list myst_parser: myst_nb loads it, and registering the .md parser
    # twice is an error.
    "myst_nb",
    "sphinxarg.ext",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_design",
    # Local, in docs/_ext. Both emit generated tables, so the pages that use
    # them cannot drift from the package.
    "calima_registry",   # .. calima-registry::  from data/registry.toml
    "calima_scripts",    # .. calima-scripts::   the modules the API omits
]

# Deliberately NOT enabled:
#
# sphinx.ext.autosectionlabel
#     The migrated narrative pages share heading names ("Usage" alone appears
#     14 times in docstrings). autosectionlabel would emit a duplicate-label
#     warning per collision and make {ref} targets ambiguous, including the
#     solver-types target the README depends on. Explicit MyST targets
#     -- (solver-types)= -- instead.
#
# sphinx.ext.viewcode
#     Would inline all 62,010 lines of src/pycalima as highlighted HTML,
#     including dust_photoelectric_heating.py (4124 lines) and dust_oppacity.py
#     (3345). linkcode below points at GitHub instead: same utility, none of
#     the page weight.
#
# sphinx_autodoc_typehints
#     Only 23% of 1301 functions are annotated, and it resolves annotations
#     with get_type_hints(), which raises on the TYPE_CHECKING-only AmesPAHdb
#     annotation in models/PAH_photophysics/pah_mol_data.py:26 (41 files use
#     `from __future__ import annotations`, so every annotation is a string
#     needing resolution). Revisit above ~70% coverage.

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
# Order matters. autosummary writes its generated stubs using the FIRST key
# here, and the stub template is reStructuredText, so ".rst" must come first --
# otherwise the stubs are written as .md, parsed as MyST, and every
# `.. automodule::` in them silently degrades to plain text.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
root_doc = "index"
language = "en"

# Required for the :template: option in api/*.md to resolve
# "autosummary/module.rst"; without it autosummary falls back to its built-in
# template, which omits :members:.
templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "README.md",
    # Never publish these, even if a stray copy appears under docs/.
    # galaxysam/SUMMARY.md carries an unfilled "[Your name]" placeholder and
    # semenov_2003/Readme.txt a third party's personal email address.
    "**/SUMMARY.md",
    "**/MIGRATION.md",
    "**/VALIDATION.md",
    "**/YIELD_FILES_MANIFEST.md",
]

# 78 single-backtick spans across the codebase are meant as literals, not as
# cross-references. Explicit roles (:func:, :mod:, :class:) are unaffected.
default_role = "literal"

nitpicky = False
nitpick_ignore_regex = [
    ("py:class", r"unyt\..*"),
    ("py:class", r"yt\..*"),
    ("py:class", r"amespahdbpythonsuite\..*"),
    ("py:class", r"numpy\.typing\..*"),
    ("py:class", r"optional"),
    ("py:class", r"array_like"),
    ("py:class", r"array-like"),
    ("py:class", r"callable"),
    ("py:class", r"file-like"),
    ("py:class", r"path-like"),
    # Prose types written in numpydoc Parameters blocks. These are English,
    # not importable objects, and there are 26 "ndarray" and 11 "shape"
    # among them; nothing is gained by resolving them.
    ("py:class", r"ndarray"),
    ("py:class", r"scalar"),
    ("py:class", r"dict-like"),
    ("py:obj", r"ndarray"),
    ("py:obj", r"shape"),
    ("py:obj", r"Choices"),
    ("py:obj", r"Path"),
    ("py:obj", r"default.*"),
    ("py:obj", r"[a-z_]+_new"),
]

# ---------------------------------------------------------------------------
# Napoleon
# ---------------------------------------------------------------------------
napoleon_google_docstring = True  # 11 files use `Args:` (110 occurrences)
napoleon_numpy_docstring = True   # the house style: 221 dashed sections
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True   # renders numpydoc Parameters types, which is why
                            # sphinx-autodoc-typehints adds little here
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# Every name below is a dashed heading that actually occurs in this codebase
# and would otherwise be an "Unexpected section title" error inside autodoc.
# Regenerate with the snippet in docs/development/docs.md.
napoleon_custom_sections = [
    "JSON schema",
    "Usage",                                        # x14
    "Modified variables",                           # x6
    "Main entry point",                             # x3
    "Algorithm",                                    # x3
    "File format",                                  # x2
    "Python API",                                   # x2
    "Models for grain relative velocity",
    "ODE state layout",
    "Table file format",
    "Panels produced",
    "Command-line usage",
    "Outputs saved automatically after every run",
    "Gas conservation",
    "Step-size control",
    "Usage (CLI)",
    "Public API",
    "GD89 model",
    "Example usage",
    "Model parameters",
    "Strategy",
    "Output",
    "Workflow",
    "Why discrepancies exist",
    "Quantities compared",
    "Data files",
    "Physical assumptions",
    "Pipeline",
    "Important caveat",
    "Parameters (packed as tuple)",
    "Species grid",
    "Processes",
    "Two solver modes",
    "Quick-start",
    "Catalog management",
    "Low-level helpers",
    "Andrews (2016) parameter table",
    "Fields",
    "Andrews data files",
    "Public functions",
    "Physical context",
    "Functions",
    "Utilities",
]

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    # 40% of public functions are undocumented, and they cluster
    # (models/dust_charge/shared_physics.py is 0/34, grain_distributions.py
    # 7/58). Listing them signature-only is more useful than hiding them: it
    # makes the coverage gap visible, and it keeps the autosummary tables in
    # _templates/autosummary/module.rst from linking to nothing.
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__,__dict__,__module__,__annotations__",
}
autodoc_inherit_docstrings = True

# Render annotations as the source strings they already are; never evaluate.
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Several kernels take array or dict defaults whose reprs are large and
# unstable across numpy versions. Show the source text instead.
autodoc_preserve_defaults = True

# Only genuinely absent at build time. numba is NOT mocked: all three import
# sites are already try/except-guarded with working pure-Python fallbacks, and
# mocking it would document the JIT branch rather than the real one. yt is NOT
# mocked either -- the docs extra installs pycalima[all].
autodoc_mock_imports = [
    "amespahdbpythonsuite",  # TYPE_CHECKING-only, pah_mol_data.py:26
    "uclchem",               # notebooks only; no PyPI wheel exists
]

autosummary_generate = True
autosummary_generate_overwrite = True
# Critical: keeps the 59 re-exports in models/PAH_photophysics/__init__.py from
# being duplicated onto that package's page.
autosummary_imported_members = False
# Honour __all__ in the three curated islands (solvers, models/dust_shielding,
# galaxysam).
autosummary_ignore_module_all = False

# ---------------------------------------------------------------------------
# MyST / myst-nb
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "dollarmath",
    "amsmath",
    "substitution",
    "tasklist",
    "attrs_inline",
    "html_image",
]
# sphinxcontrib-mermaid registers a directive, not a fence handler, so without
# this a ```mermaid fence in an included README is treated as a code block with
# an unknown Pygments lexer. solvers/ramses_source/README.md has one.
myst_fence_as_directive = ["mermaid"]

myst_heading_anchors = 3   # gives migrated README headings stable anchors, so
                           # old README anchor links keep landing correctly
myst_dmath_double_inline = False

# GUARD: with dollarmath enabled, two unbackticked $CALIMA_* on one line parse
# as a math span and the variable names are silently swallowed. All of them are
# backticked today; the CI grep gate keeps it that way. Do not drop that gate.

myst_substitutions = {
    "release": release,
    "repo_url": "https://github.com/currodri/pyCALIMA",
}

nb_execution_mode = "cache"
nb_execution_cache_path = str(HERE / "_build" / ".jupyter_cache")
# Measured locally: calima_tutorial 79s, calima_dust_pah_processes 3s. The
# generous ceiling is for slower CI runners.
nb_execution_timeout = 1800
nb_execution_raise_on_error = True   # a silently broken tutorial is worse than
                                     # a red build
nb_execution_show_tb = True
# The two notebooks needing user-supplied multi-GB RAMSES snapshots via
# $CALIMA_SIM_DIR live here with outputs baked in. A directory glob rather than
# two filenames, so a typo cannot fall through to executing them.
nb_execution_excludepatterns = ["tutorials/rendered/*"]
nb_merge_streams = True
nb_output_stderr = "remove-warn"

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
intersphinx_timeout = 15

# ---------------------------------------------------------------------------
# extlinks / bibtex
# ---------------------------------------------------------------------------
extlinks = {
    "ads": ("https://ui.adsabs.harvard.edu/abs/%s/abstract", "%s"),
    "doi": ("https://doi.org/%s", "doi:%s"),
    "ghfile": ("https://github.com/currodri/pyCALIMA/blob/main/%s", "%s"),
}
extlinks_detect_hardcoded_links = True

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "plain"
# Docstrings are not bibtex-aware: there are zero numpydoc References sections
# and 153 author-year strings with inconsistent spellings. refs.bib serves the
# narrative pages only.

# ---------------------------------------------------------------------------
# MathJax
# ---------------------------------------------------------------------------
# LaTeX appears in exactly one module (solvers/dust_rates.py: 5 `.. math::`,
# 8 `:math:`), correctly escaped. Everything else writes formulae as
# column-aligned unicode, which renders as HTML text and needs no MathJax.
mathjax3_config = {
    "tex": {"inlineMath": [["\\(", "\\)"]], "processEscapes": True},
    "chtml": {"displayAlign": "left", "displayIndent": "2em"},
}

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = f"pyCALIMA {version}"
# MANIFEST.in prunes assets/, so the logo never ships in the sdist. Pointing at
# it relative to confdir means Sphinx copies it into _static at build time and
# there is no second committed copy to keep in sync.
html_logo = "../assets/CALIMA_logo1.png"
html_favicon = "../assets/CALIMA_logo1.png"
html_static_path = ["_static"]
html_show_sourcelink = True
html_copy_source = False
html_baseurl = "https://currodri.github.io/pyCALIMA/"
html_theme_options = {
    "source_repository": "https://github.com/currodri/pyCALIMA/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}

# ---------------------------------------------------------------------------
# linkcode -> GitHub
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "main"


_LINKCODE_SHA = _git_sha()


def linkcode_resolve(domain, info):
    """Map a documented object to its line range on GitHub.

    Deliberately tolerant. Some objects are reached through the three curated
    ``__init__.py`` files and some are wrapped by decorators, so ``inspect``
    fails for a minority. A failure means no source link, never a build error.
    """
    if domain != "py" or not info.get("module"):
        return None
    try:
        mod = sys.modules.get(info["module"])
        if mod is None:
            return None
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        filename = Path(inspect.getsourcefile(obj)).resolve()
        _, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None
    try:
        rel = filename.relative_to(REPO)
    except ValueError:
        return None
    return (
        f"https://github.com/currodri/pyCALIMA/blob/{_LINKCODE_SHA}/"
        f"{rel.as_posix()}#L{lineno}"
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def _skip_private_members(app, what, name, obj, skip, options):
    """Keep the underscore-prefixed top-level *modules*, drop private members.

    ``pycalima._paths`` and ``pycalima._datasets`` are documented deliberately:
    the README already directs users at them and they back two console
    scripts. Their private helpers are not.
    """
    if name.startswith("_") and not name.startswith("__"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_private_members)
    return {"version": release, "parallel_read_safe": True}
