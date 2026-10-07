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
where it is measured, else ``x_e`` times n_H.
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


def bracketing_environments(region, template):
    """{"least ionising", "most ionising"}: Environment corners of the template's input ranges."""
    from pycalima.pahchem.baseline import Environment

    card = load(region)
    t = card[template]
    teff = int(card["region"]["field_teff_k"]["value"])
    g0, nh = bounds(t["g0"]), bounds(t["n_h"])
    temp = bounds(t["t_gas"])[0]
    if "n_e" in t:
        ne_least, ne_most = bounds(t["n_e"])[1], bounds(t["n_e"])[0]
    else:
        xe = bounds(t["x_e"])
        ne_least, ne_most = xe[1] * nh[1], xe[0] * nh[0]
    return {"least ionising": Environment(f"{template} (least ionising)", g0[0], nh[1], ne_least, temp, teff),
            "most ionising": Environment(f"{template} (most ionising)", g0[1], nh[0], ne_most, temp, teff)}
