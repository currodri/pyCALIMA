"""Observed local conditions as inputs of the PAH network and its emission.

pahchem does not compute gas densities, temperatures, electron densities or
fields: they come from the observations' own analyses, recorded with sources
in ``environments/<region>.toml``. Each quantity is a value or a [low, high]
range with a ``source``, an A-E ``confidence`` and ``stated`` (whether a paper
gives it for that place, or the range is pahchem's assumption).

``bracketing_environments`` turns a template's ranges into the two
``baseline.Environment`` corners that bracket the PAH ionisation and
dehydrogenation: least ionising (lowest G0, highest n_H and n_e) and most
ionising (highest G0, lowest n_H and n_e). The electron density is ``n_e``
where it is measured, else ``x_e`` times n_H. A G0 entry may name its
``definition`` (``G0_DEFINITIONS``); it is converted to the Habing integral
for the region's field shape.
"""

from __future__ import annotations

from importlib import resources

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    import tomli as tomllib


def load(region):
    """The parsed conditions card ``environments/<region>.toml``."""
    with resources.files("pycalima.pahchem").joinpath("environments", f"{region}.toml").open("rb") as fh:
        return tomllib.load(fh)


def bounds(entry):
    """(low, high) of a quantity given as ``value`` or ``range``."""
    if "range" in entry:
        lo, hi = entry["range"]
        return float(lo), float(hi)
    return float(entry["value"]), float(entry["value"])


#: G0 definitions a card entry may carry (``definition``); pahchem's own is the Habing 6-13.6 eV integral
G0_DEFINITIONS = {"habing_6_13.6": "Habing: int_6^13.6 E F(E) dE / 1.6e-3 erg cm^-2 s^-1 (pahchem, fields.py)",
                  "fnu_1030A": "F_nu at the O I pump lines 1027/1040 A over 8.7e-19 erg cm^-2 s^-1 Hz^-1 (Peeters+24 App. A)"}


def g0_conversion(entry, teff_k):
    """Stated G0 over pahchem's Habing G0 for a card entry, given the field shape (Kurucz ``teff_k``)."""
    from pycalima.pahchem import fields

    d = entry.get("definition", "habing_6_13.6")
    if d not in G0_DEFINITIONS:
        raise ValueError(f"unknown G0 definition {d!r}; known: {sorted(G0_DEFINITIONS)}")
    return 1.0 if d == "habing_6_13.6" else fields.fnu_g0_ratio(fields.kurucz(teff_k, 1.0))


def bracketing_environments(region, template):
    """{"least ionising", "most ionising"}: Environment corners of the template's input ranges."""
    from pycalima.pahchem.baseline import Environment

    card = load(region)
    t = card[template]
    teff = int(card["region"]["field_teff_k"]["value"])
    g0, nh = bounds(t["g0"]), bounds(t["n_h"])
    g0 = tuple(v / g0_conversion(t["g0"], teff) for v in g0)
    temp = bounds(t["t_gas"])[0]
    if "n_e" in t:
        ne_least, ne_most = bounds(t["n_e"])[1], bounds(t["n_e"])[0]
    else:
        xe = bounds(t["x_e"])
        ne_least, ne_most = xe[1] * nh[1], xe[0] * nh[0]
    return {"least ionising": Environment(f"{template} (least ionising)", g0[0], nh[1], ne_least, temp, teff),
            "most ionising": Environment(f"{template} (most ionising)", g0[1], nh[0], ne_most, temp, teff)}
