"""
DUST EMISSION

In this script there are tools to test the emission of dust from
the modelling used in Dusty-PRISM. This considers boths the emission
from quasi-steady temperature large grains as well as the stochastic
emission from small grains and PAHs.

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import some libraries
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})
import re
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution
from dust_oppacity import dust_efficiencies,pah_efficiencies
from PAHs_model import Draine_1978_isrf

# Functions
