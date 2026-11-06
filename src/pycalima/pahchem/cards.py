"""Method cards: the recorded inputs of published PAH models.

Each ``method_cards/<name>.toml`` lists, value by value, the inputs of one
paper with their source and a confidence grade (A experiment ... E
assumption). See docs/physics/pahchem-spec.md.
"""

from __future__ import annotations

from importlib import resources

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


def load_card(name: str) -> dict:
    """The parsed method card ``method_cards/<name>.toml``."""
    path = resources.files("pycalima.pahchem").joinpath("method_cards", f"{name}.toml")
    with path.open("rb") as fh:
        return tomllib.load(fh)
