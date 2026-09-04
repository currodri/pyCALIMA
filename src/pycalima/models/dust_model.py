"""
MODELLING DUST DISTRIBUTION AND EVOLUTION

These functions allow for the computation of intrinstic dust
size distributions and how different processes affect their
formation, size and properties.

By: Curro Rodriguez (currodri@gmail.com)
"""

import numpy as np
import re

from pycalima.models.grain_distributions import (
    Classical_LogNormal_Distribution,
    LogNormal_Distribution,
    PowerLaw_ExpCutoff_Distribution,
)
from pycalima.models.grain_size_config import (
    build_lognormal_distribution,
    get_bin_by_rank,
    get_parameter_array,
    get_bin_index,
    load_grain_size_config,
)


def _load_parameter_arrays():
    return {
        "basic_a0": get_parameter_array("a0", "basic"),
        "basic_amin": get_parameter_array("amin", "basic"),
        "basic_amax": get_parameter_array("amax", "basic"),
        "basic_sigma": get_parameter_array("sigma", "basic"),
        "basic_s": get_parameter_array("s", "basic"),
        "shattering_a0": get_parameter_array("a0", "shattering"),
        "shattering_amin": get_parameter_array("amin", "shattering"),
        "shattering_amax": get_parameter_array("amax", "shattering"),
        "shattering_sigma": get_parameter_array("sigma", "shattering"),
        "shattering_s": get_parameter_array("s", "shattering"),
    }


def reload_grain_size_model(config_path=None):
    """Reload JSON configuration and refresh module-level legacy arrays."""
    load_grain_size_config(config_path=config_path, reload=True)
    arrays = _load_parameter_arrays()
    globals().update(arrays)


def build_distribution(species, model_name="basic"):
    """Build a LogNormal distribution from JSON settings.

    Args:
        species: Bin id (e.g. "graphite_bin_0") or integer index.
        model_name: Parameter set name in JSON ("basic" or "shattering").
    """
    return build_lognormal_distribution(species=species, model_name=model_name)


def build_distribution_for(composition, bin_rank=0, is_pah=False, model_name="basic"):
    """Build a LogNormal distribution by composition and bin rank.

    Args:
        composition: "graphite" or "silicate".
        bin_rank: Integer rank of the bin for that composition.
        is_pah: Whether to select a PAH bin.
        model_name: Parameter set in JSON ("basic" or "shattering").
    """
    meta = get_bin_by_rank(composition=composition, bin_rank=bin_rank, is_pah=is_pah)
    return build_distribution(meta["id"], model_name=model_name)


def build_distribution_from_dust_type(dust_type, model_name="basic"):
    """Build a LogNormal distribution from a dust token.

    Supported generic formats:
    - `graphite_bin_<N>`
    - `silicate_bin_<N>`
    - `pah_bin_<N>`
    - `pah_ion_bin_<N>` / `pah_neutral_bin_<N>`
    """
    token = str(dust_type).lower()

    m = re.search(r"bin[_-]?(\d+)", token)
    bin_rank = int(m.group(1)) if m else 0

    if "pah" in token:
        composition = "graphite"
        is_pah = True
    elif "sil" in token:
        composition = "silicate"
        is_pah = False
    else:
        composition = "graphite"
        is_pah = False

    return build_distribution_for(
        composition=composition,
        bin_rank=bin_rank,
        is_pah=is_pah,
        model_name=model_name,
    )


# Load defaults on import and expose historical variable names used by many modules.
globals().update(_load_parameter_arrays())
