"""
Galaxy Semi-Analytic Model (SAM) Evolution Module

This module provides tools for computing galactic chemical evolution using 
different stellar yield models (Kobayashi, LC18, Karakas, etc.) and simulating
galaxy properties evolution over cosmic time.

Components:
- yield_models: Reading and interpolating different yield tables
- galaxy_sam: Main galaxy SAM evolution simulation
- imf: Initial Mass Function utilities
- sn1a: Type Ia supernova calculations
- constants: Physical constants and default parameters
- plotting: Visualization tools for yields and evolution results
"""

from . import yield_models
from . import galaxy_sam
from . import imf
from . import sn1a
from . import constants
from . import plotting

__version__ = "0.1.0"
__all__ = [
    "yield_models",
    "galaxy_sam",
    "imf",
    "sn1a",
    "constants",
    "plotting",
]
