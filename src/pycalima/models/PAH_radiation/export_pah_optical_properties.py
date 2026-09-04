"""
PAH OPTICAL PROPERTIES BATCH EXPORTER

This script is a convenience wrapper that exports optical properties for all PAH bins
defined in the grain size configuration to model_data/optical_properties/.

The computation depends only on PAH size and composition from the configuration.

By: Curro Rodriguez (currodri@gmail.com)
"""

import os
import argparse
from pathlib import Path

from pycalima.models.PAH_radiation.pah_oppacity import export_pah_optical_properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export optical properties for all PAH bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    args = parser.parse_args()
    
    export_pah_optical_properties(config_path=args.config)
