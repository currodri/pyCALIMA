"""pahchem: method cards and the gates of the benchmark ladder.

A method card records the inputs of a published PAH model. The invariants here
keep the cards honest: every number names where in the paper it comes from,
and every card cites a bibliography entry that exists.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

CARDS = sorted(
    p for p in resources.files("pycalima.pahchem").joinpath("method_cards").iterdir()
    if p.name.endswith(".toml")
)
REFS_BIB = Path(__file__).resolve().parents[1] / "docs" / "refs.bib"
REQUIRED_SECTIONS = {"card", "species", "dissociation", "targets"}
NUMERIC_KEYS = {"value", "E_act", "efficiency"}


def _load(card):
    with card.open("rb") as fh:
        return tomllib.load(fh)


def _inline_tables(node, path=()):
    """Yield (path, table) for every table holding a number to be sourced."""
    if isinstance(node, dict):
        if NUMERIC_KEYS & node.keys():
            yield path, node
        for key, value in node.items():
            yield from _inline_tables(value, path + (key,))


def test_there_are_method_cards():
    assert {c.name for c in CARDS} >= {"andrews16.toml", "montillaud13.toml"}


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_card_has_required_sections(card):
    data = _load(card)
    assert REQUIRED_SECTIONS <= data.keys(), REQUIRED_SECTIONS - data.keys()
    assert data["card"]["citation"]


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_card_bib_key_is_in_the_bibliography(card):
    key = _load(card)["card"]["bib_key"]
    assert re.search(r"@\w+\{" + re.escape(key) + ",", REFS_BIB.read_text())


@pytest.mark.parametrize("card", CARDS, ids=lambda c: c.name)
def test_every_number_has_a_source_and_confidence(card):
    tables = list(_inline_tables(_load(card)))
    assert tables
    for path, table in tables:
        where = ".".join(path)
        assert table.get("source"), f"{card.name}: {where} has no source"
        assert table.get("confidence") in {"A", "B", "C", "D", "E"}, (
            f"{card.name}: {where} has no A-E confidence")
