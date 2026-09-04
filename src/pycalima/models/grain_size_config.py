"""Utilities to read grain size distribution settings from JSON.

This module centralizes grain size parameters used across `models/`.
"""

import json
import warnings
from pathlib import Path

import numpy as np

from pycalima import _paths

# get_git_info is re-exported: the exporters import it from this module. The
# implementation lives in pycalima._provenance, which queries git against the
# package's own tree rather than the caller's CWD, and falls back to the
# setuptools_scm-stamped distribution version when there is no .git.
from pycalima._provenance import get_git_info, provenance_string  # noqa: F401

_CONFIG_CACHE = None
_CURRENT_CONFIG_PATH = None


def _default_config_path():
    """Bundled default grain-size configuration."""
    return _paths.resolve_grain_config_path(None)


def get_repo_root():
    """Deprecated. Use :mod:`pycalima._paths` instead.

    This used to return the repository root, from which callers built
    ``external_data/``, ``optical_props/`` and ``model_data/`` paths. Once
    installed there is no single such root: read-only reference data lives
    inside the package while generated data lives in a writable user
    directory. A single retarget therefore cannot be correct for all callers,
    so this shim returns the *read-only* reference-data root -- right for
    ``external_data`` and ``optical_props``, and deliberately wrong for
    ``model_data``, where ``_paths._assert_writable_target`` will then raise
    rather than let anything write into site-packages.

    Replace with::

        get_repo_root() / "external_data"  ->  _paths.get_external_data_path()
        get_repo_root() / "optical_props"  ->  _paths.get_optical_props_path()
        get_repo_root() / "model_data"     ->  get_model_data_dir()

    .. deprecated:: 0.1
    """
    warnings.warn(
        "grain_size_config.get_repo_root() is deprecated; use "
        "pycalima._paths.get_external_data_path() / get_optical_props_path() / "
        "grain_size_config.get_model_data_dir() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _paths.get_data_root()


def get_optical_props_path(*parts):
    """Path inside the bundled ``optical_props/`` tree.

    ``get_optical_props_path()`` still returns the directory itself, which is
    what the existing callers expect.
    """
    return _paths.get_optical_props_path(*parts)


def get_external_data_path(*parts):
    """Path inside the bundled ``external_data/`` tree."""
    return _paths.get_external_data_path(*parts)


def get_model_data_dir(model_name=None, *, base=None, create=False):
    """Return ``model_data/<model_name>/`` for the active configuration.

    Parameters
    ----------
    model_name : str, optional
        Override the ``model_name`` key of the active configuration.
    base : str or Path, optional
        Explicit ``model_data`` directory; bypasses environment and CWD
        resolution.
    create : bool, default False
        Create the directory if missing. False by default so that importing a
        module never writes to disk.

    Notes
    -----
    Reads ``model_name`` from the cached parsed configuration. The
    pre-packaging implementation re-read and re-parsed the JSON file on every
    call -- bypassing ``_CONFIG_CACHE`` entirely -- inside export loops.
    """
    if model_name is None:
        model_name = load_grain_size_config().get("model_name")
    return _paths.get_model_data_dir(model_name, base=base, create=create)


def set_config_path(config_path):
    """Set the configuration file path globally.

    Call this at the start of a script to use a specific JSON configuration.
    All subsequent calls to load_grain_size_config() will use this path.

    Parameters
    ----------
    config_path : str or Path or None
        A path to a JSON configuration file, or a bundled short name such as
        ``"ramses4bin"``, ``"4C6Si"``, ``"test"`` or ``"default"``. None
        restores the bundled default.

    Notes
    -----
    The path is resolved to an absolute file immediately, so a later
    ``os.chdir`` cannot invalidate it and two spellings of the same file no
    longer produce two cache entries.
    """
    global _CURRENT_CONFIG_PATH, _CONFIG_CACHE
    _CURRENT_CONFIG_PATH = (
        _paths.resolve_grain_config_path(config_path) if config_path else None
    )
    _CONFIG_CACHE = None  # Clear cache to force reload
    return _CURRENT_CONFIG_PATH


def get_config_path():
    """Get the currently active configuration path."""
    return _CURRENT_CONFIG_PATH or _default_config_path()


def _as_float_array(values, key, expected_len):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"'{key}' must be a 1D list.")
    if arr.size != expected_len:
        raise ValueError(
            f"'{key}' has length {arr.size}, expected {expected_len}."
        )
    return arr


def load_grain_size_config(config_path=None, reload=False):
    global _CONFIG_CACHE

    # Use provided path, or current global path, or default. Resolving here
    # normalises the cache key, so "models/x.json" and "/abs/models/x.json" no
    # longer produce two entries for one file.
    if config_path is not None:
        path = _paths.resolve_grain_config_path(config_path)
    else:
        path = get_config_path()

    if _CONFIG_CACHE is not None and not reload and _CONFIG_CACHE["__path__"] == str(path):
        return _CONFIG_CACHE

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    bins = raw.get("bins")
    if not bins or not isinstance(bins, list):
        raise ValueError("'bins' must be a non-empty list.")

    ids = []
    metadata = []
    for i, item in enumerate(bins):
        if not isinstance(item, dict):
            raise ValueError("Each entry in 'bins' must be an object.")
        bin_id = item.get("id")
        if not bin_id or not isinstance(bin_id, str):
            raise ValueError(f"bins[{i}].id must be a non-empty string.")
        composition = str(item.get("composition", "")).lower()
        if composition not in ("graphite", "silicate"):
            raise ValueError(
                f"bins[{i}].composition must be 'graphite' or 'silicate'."
            )
        is_pah = bool(item.get("is_pah", False))
        bin_rank = int(item.get("bin_rank", 0))
        ids.append(bin_id)
        metadata.append(
            {
                "id": bin_id,
                "composition": composition,
                "is_pah": is_pah,
                "bin_rank": bin_rank,
                "index": i,
            }
        )

    n_species = len(ids)

    export_parameters = raw.get("export_parameters", {})
    if export_parameters is None:
        export_parameters = {}
    if not isinstance(export_parameters, dict):
        raise ValueError("'export_parameters' must be an object if provided.")

    parsed = {
        "bins": tuple(metadata),
        "bin_ids": tuple(ids),
        "bin_to_index": {name: i for i, name in enumerate(ids)},
        "export_parameters": dict(export_parameters),
        # Cached so get_model_data_dir() need not re-read the JSON per call.
        "model_name": raw.get("model_name"),
        "__path__": str(path),
    }

    for model_name in ("basic", "shattering"):
        section = raw.get(model_name)
        if section is None:
            raise ValueError(f"Missing '{model_name}' section in {path}.")

        parsed[model_name] = {
            "a0": _as_float_array(section.get("a0"), f"{model_name}.a0", n_species),
            "amin": _as_float_array(section.get("amin"), f"{model_name}.amin", n_species),
            "amax": _as_float_array(section.get("amax"), f"{model_name}.amax", n_species),
            "sigma": _as_float_array(section.get("sigma"), f"{model_name}.sigma", n_species),
            "s": _as_float_array(section.get("s"), f"{model_name}.s", n_species),
        }

    _CONFIG_CACHE = parsed
    return parsed


def get_parameter_array(parameter_name, model_name="basic"):
    cfg = load_grain_size_config()
    if model_name not in cfg:
        raise KeyError(f"Unknown model set '{model_name}'.")
    if parameter_name not in cfg[model_name]:
        raise KeyError(f"Unknown parameter '{parameter_name}'.")
    return cfg[model_name][parameter_name].copy()


def get_species_index(species_name):
    return get_bin_index(species_name)


def get_bin_index(bin_id):
    cfg = load_grain_size_config()
    try:
        return cfg["bin_to_index"][bin_id]
    except KeyError as exc:
        available = ", ".join(cfg["bin_ids"])
        raise KeyError(f"Unknown bin '{bin_id}'. Available: {available}") from exc


def get_bins(composition=None, is_pah=None):
    cfg = load_grain_size_config()
    out = []
    for item in cfg["bins"]:
        if composition is not None and item["composition"] != str(composition).lower():
            continue
        if is_pah is not None and item["is_pah"] != bool(is_pah):
            continue
        out.append(dict(item))
    return out


def get_bin_by_rank(composition, bin_rank=0, is_pah=False):
    candidates = sorted(
        get_bins(composition=composition, is_pah=is_pah),
        key=lambda x: x["bin_rank"],
    )
    if not candidates:
        raise KeyError(
            f"No bins found for composition='{composition}' and is_pah={is_pah}."
        )
    rank = int(bin_rank)
    for c in candidates:
        if c["bin_rank"] == rank:
            return c
    raise KeyError(
        f"No bin found for composition='{composition}', is_pah={is_pah}, bin_rank={rank}."
    )


def get_lognormal_parameters(species, model_name="basic"):
    cfg = load_grain_size_config()

    if isinstance(species, str):
        idx = get_bin_index(species)
    else:
        idx = int(species)

    if model_name not in cfg:
        raise KeyError(f"Unknown model set '{model_name}'.")

    section = cfg[model_name]
    return {
        "a0": float(section["a0"][idx]),
        "amin": float(section["amin"][idx]),
        "amax": float(section["amax"][idx]),
        "sigma": float(section["sigma"][idx]),
        "s": float(section["s"][idx]),
        "index": idx,
    }


def build_lognormal_distribution(species, model_name="basic", distribution_class=None):
    if distribution_class is None:
        from pycalima.models.grain_distributions import LogNormal_Distribution

        distribution_class = LogNormal_Distribution

    p = get_lognormal_parameters(species, model_name=model_name)
    return distribution_class(p["a0"], p["amin"], p["amax"], p["sigma"], p["s"])


def get_export_parameters(section=None, defaults=None):
    """Return export parameter settings from the active configuration.

    Parameters
    ----------
    section : str, optional
        Top-level key under ``export_parameters`` (for example,
        ``dust_photoelectric_heating``). If not provided, returns the whole
        export parameter mapping.
    defaults : dict, optional
        Default values to merge with config values (config wins).

    Returns
    -------
    dict
        Export parameters for the requested section.
    """
    cfg = load_grain_size_config()
    export_cfg = dict(cfg.get("export_parameters", {}))

    if section is None:
        section_cfg = export_cfg
    else:
        section_cfg = export_cfg.get(section, {})
        if section_cfg is None:
            section_cfg = {}
        if not isinstance(section_cfg, dict):
            raise ValueError(
                f"export_parameters.{section} must be an object if provided."
            )
        section_cfg = dict(section_cfg)

    if defaults is None:
        return section_cfg

    merged = dict(defaults)
    merged.update(section_cfg)
    return merged


def get_header_lines(title, script_name, bin_info=None, val_desc=None, num_lines=None):
    """Generate standardized comment header lines for rate tables."""
    from datetime import datetime

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    headers = [
        f"# {title}",
        f"# Script: {script_name}",
        f"# Date: {date_str}",
        f"# Code: {provenance_string()}",
    ]
    if bin_info:
        headers.append(f"# {bin_info}")
    else:
        headers.append("# Bin info: none")
        
    if val_desc:
        headers.append(f"# {val_desc}")
    else:
        headers.append("# Values: log10(rate)")

    if num_lines is not None:
        # Enforce exact line count if requested
        if len(headers) < num_lines:
            while len(headers) < num_lines:
                headers.append("#")
        elif len(headers) > num_lines:
            headers = headers[:num_lines]
            
    return headers

