"""``.. calima-registry::`` -- tabulate the bundled dataset registry.

The registry is the authority on where every reference dataset came from, so
the provenance page is generated from it at build time and cannot drift out of
step with what the package actually ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 11):
    import tomllib
else:  # the package declares tomli for this case
    import tomli as tomllib

COLUMNS = [
    ("name", "Dataset"),
    ("kind", "Kind"),
    ("dest", "Location"),
    ("size", "Size"),
    ("url_status", "URL"),
    ("citation", "Source"),
]


def _registry_path(app) -> Path:
    return Path(app.confdir).parent / "src" / "pycalima" / "data" / "registry.toml"


def _human_size(entry) -> str:
    n = entry.get("size_bytes")
    if not n:
        return "-"
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


class CalimaRegistry(Directive):
    has_content = False

    def run(self):
        app = self.state.document.settings.env.app
        path = _registry_path(app)
        if not path.is_file():
            logger.warning("calima-registry: %s not found", path)
            return [nodes.paragraph(text=f"registry not found: {path}")]

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        entries = data.get("dataset") or data.get("datasets") or {}
        if isinstance(entries, dict):
            rows = [dict(v, name=k) for k, v in entries.items()]
        else:
            rows = list(entries)
        rows.sort(key=lambda r: (r.get("kind", ""), r.get("name", "")))

        table = nodes.table(classes=["colwidths-auto"])
        group = nodes.tgroup(cols=len(COLUMNS))
        table += group
        for _ in COLUMNS:
            group += nodes.colspec(colwidth=1)

        head = nodes.thead()
        group += head
        head += self._row([label for _key, label in COLUMNS])

        body = nodes.tbody()
        group += body
        for entry in rows:
            body += self._row([
                entry.get("name", "?"),
                entry.get("kind", "?"),
                entry.get("dest", "-"),
                _human_size(entry),
                entry.get("url_status", "-"),
                entry.get("citation", "-"),
            ])
        return [table]

    @staticmethod
    def _row(values):
        row = nodes.row()
        for value in values:
            cell = nodes.entry()
            cell += nodes.paragraph(text=str(value))
            row += cell
        return row


def setup(app):
    app.add_directive("calima-registry", CalimaRegistry)
    return {"version": "1.0", "parallel_read_safe": True}
