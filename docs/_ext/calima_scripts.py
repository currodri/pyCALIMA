"""``.. calima-scripts::`` -- index the modules the API reference omits.

Of 116 non-``__init__`` modules, 29 are figure-reproduction or diagnostic
scripts and 12 are batch exporters. None belongs in the API reference, but they
should not vanish either: this directive walks the package and lists whatever
the API pages do not claim, so the listing cannot drift.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Kept in step with the module lists in docs/api/*.md by the test in
# tests/test_docs.py, which fails if a module is in neither place.
FIGURE_RE = re.compile(
    r"^(compare_|reproduce_|diagnose_|plot_|scan_|profile_|benchmark_|check_|"
    r"make_|run_eqtemp|fit_|download_|dustregime_|test_|run_acetylene)"
)


def _package_root(app) -> Path:
    return Path(app.confdir).parent / "src" / "pycalima"


def _summary(path: Path) -> str:
    """First line of the module docstring, or a dash."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return "-"
    doc = ast.get_docstring(tree)
    if not doc:
        return "-"
    return doc.strip().splitlines()[0].strip() or "-"


def _has_main(path: Path) -> bool:
    return bool(
        re.search(r"^if __name__ == ", path.read_text(encoding="utf-8", errors="ignore"), re.M)
    )


class CalimaScripts(Directive):
    has_content = False
    option_spec = {}

    def run(self):
        app = self.state.document.settings.env.app
        root = _package_root(app)
        if not root.is_dir():
            logger.warning("calima-scripts: %s not found", root)
            return []

        # Whatever the API pages already list is not a script.
        documented = set()
        for page in (Path(app.srcdir) / "api").glob("*.md"):
            for line in page.read_text(encoding="utf-8").splitlines():
                token = line.strip()
                if token.startswith("pycalima."):
                    documented.add(token)

        rows = []
        for path in sorted(root.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            module = "pycalima." + str(
                path.relative_to(root).with_suffix("")
            ).replace("/", ".")
            if module in documented:
                continue
            stem = path.stem
            kind = ("exporter" if stem.startswith("export_")
                    else "script" if FIGURE_RE.match(stem)
                    else "other")
            rows.append((module, kind, _summary(path), _has_main(path)))

        if not rows:
            return [nodes.paragraph(text="Every module is in the API reference.")]

        table = nodes.table(classes=["colwidths-auto"])
        group = nodes.tgroup(cols=4)
        table += group
        for _ in range(4):
            group += nodes.colspec(colwidth=1)
        head = nodes.thead()
        group += head
        head += self._row(["Module", "Kind", "Summary", "Runnable"])
        body = nodes.tbody()
        group += body
        for module, kind, summary, has_main in rows:
            body += self._row([module, kind, summary, "yes" if has_main else ""],
                              literal_first=True)
        return [table]

    @staticmethod
    def _row(values, literal_first=False):
        row = nodes.row()
        for index, value in enumerate(values):
            cell = nodes.entry()
            para = nodes.paragraph()
            if literal_first and index == 0:
                para += nodes.literal(text=str(value))
            else:
                para += nodes.Text(str(value))
            cell += para
            row += cell
        return row


def setup(app):
    app.add_directive("calima-scripts", CalimaScripts)
    return {"version": "1.0", "parallel_read_safe": True}
