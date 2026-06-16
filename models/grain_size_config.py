"""Utilities to read grain size distribution settings from JSON.

This module centralizes grain size parameters used across `models/`.
"""

import json
from pathlib import Path

import numpy as np

_CONFIG_CACHE = None
_CURRENT_CONFIG_PATH = None


def _default_config_path():
    return Path(__file__).with_name("grain_size_distribution.json")


def get_repo_root():
    """
    Get the root directory of the CALIMA repository.
    
    Returns
    -------
    Path
        Path to the repository root (parent of the 'models' directory).
    """
    return Path(__file__).parent.parent


def get_optical_props_path():
    """
    Get the path to the optical_props directory.
    
    Returns
    -------
    Path
        Path to optical_props folder in the repo root.
    """
    return get_repo_root() / "optical_props"


def set_config_path(config_path):
    """
    Set the configuration file path globally.
    
    Call this at the start of a script to use a specific JSON configuration.
    All subsequent calls to load_grain_size_config() will use this path.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the JSON configuration file.
    """
    global _CURRENT_CONFIG_PATH, _CONFIG_CACHE
    _CURRENT_CONFIG_PATH = Path(config_path) if config_path else None
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

    # Use provided path, or current global path, or default
    if config_path is not None:
        path = Path(config_path)
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
        from models.grain_distributions import LogNormal_Distribution

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


def get_git_info():
    """Retrieve git branch and short commit hash."""
    import subprocess
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        commit_short = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return branch, commit_short
    except Exception:
        return 'unknown', 'unknown'


def get_header_lines(title, script_name, bin_info=None, val_desc=None, num_lines=None):
    """Generate standardized comment header lines for rate tables."""
    from datetime import datetime
    branch, commit = get_git_info()
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    headers = [
        f"# {title}",
        f"# Script: {script_name}",
        f"# Date: {date_str}",
        f"# Git: {branch} ({commit})",
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

